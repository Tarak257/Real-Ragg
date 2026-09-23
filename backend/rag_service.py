import os
import sys
import tempfile
import time
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from rag_engine.chunking import CHUNKING_STRATEGIES, create_chunks
from rag_engine.query_rewriter import REWRITER_MODES, rewrite_query
from rag_engine.reranker import HybridReranker
from ingest import ingest_document

from backend.models import (
    QueryRequest,
    QueryResponse,
    SourcePassage,
    PipelineDebugInfo,
    IngestResponse,
    StatusResponse
)

load_dotenv()

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

class RAGService:
    def __init__(self):
        self.chunks: List[Document] = []
        self.current_doc_name: str = "myres.pdf"
        self.chunk_strategy: str = "Recursive Character"
        self.reranker: Optional[HybridReranker] = None
        self._initialize_default_doc()

    def _get_embeddings(self):
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    def load_vectorstore(self) -> Chroma:
        embeddings = self._get_embeddings()
        return Chroma(
            collection_name="rag_documents",
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH
        )

    def _initialize_default_doc(self):
        default_pdf = os.path.join(os.path.dirname(__file__), "..", "documents", "myres.pdf")
        if os.path.exists(default_pdf):
            try:
                vstore, chunks = ingest_document(
                    file_path=default_pdf,
                    chroma_path=CHROMA_PATH,
                    strategy=self.chunk_strategy,
                    chunk_size=1000,
                    chunk_overlap=200,
                    clear_existing=True
                )
                self.chunks = chunks
                self.current_doc_name = "myres.pdf"
                self.reranker = HybridReranker(chunks)
                print(f"✅ Default document {default_pdf} loaded ({len(chunks)} chunks).")
            except Exception as e:
                print(f"Notice during default document load: {e}")

    def get_status(self) -> StatusResponse:
        return StatusResponse(
            status="success",
            active_document=self.current_doc_name,
            chunk_strategy=self.chunk_strategy,
            total_chunks=len(self.chunks)
        )

    def process_document(
        self,
        file_bytes: bytes,
        filename: str,
        strategy: str = "Recursive Character",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> IngestResponse:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".pdf", ".txt"]:
            raise ValueError("Only PDF (.pdf) and Text (.txt) files are supported.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            vstore, chunks = ingest_document(
                file_path=tmp_path,
                chroma_path=CHROMA_PATH,
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                clear_existing=True
            )
            self.chunks = chunks
            self.current_doc_name = filename
            self.chunk_strategy = strategy
            self.reranker = HybridReranker(chunks)

            return IngestResponse(
                status="success",
                message=f"Successfully ingested {filename}",
                document_name=filename,
                chunk_strategy=strategy,
                total_chunks=len(chunks)
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def invoke_llm_with_fallback(
        self,
        prompt_input,
        primary_model: str = "gemini-3.6-flash",
        temperature: float = 0.0
    ) -> Tuple[str, str]:
        candidate_models = [primary_model] + [
            m for m in ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
            if m != primary_model
        ]
        last_err = None
        for m_name in candidate_models:
            try:
                llm_inst = ChatGoogleGenerativeAI(model=m_name)
                resp = llm_inst.invoke(prompt_input)
                
                if isinstance(resp.content, list):
                    extracted = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(getattr(item, "text", item))
                        for item in resp.content
                    )
                else:
                    extracted = str(resp.content)
                return extracted, m_name
            except Exception as e:
                last_err = e
                print(f"Notice: Model {m_name} failed with: {e}. Retrying fallback...")
                continue

        raise last_err

    def query(self, req: QueryRequest) -> QueryResponse:
        start_time = time.time()
        question = req.question.strip()
        if not question:
            raise ValueError("Query question cannot be empty.")

        vector_store = self.load_vectorstore()
        llm = ChatGoogleGenerativeAI(model=req.model_choice)

        # Step 1: Query Expansion
        rewrite_result = rewrite_query(question, mode=req.query_mode, llm=llm)
        search_queries = rewrite_result.get("search_queries", [question])

        # Step 2: Hybrid Retrieval
        if req.use_hybrid and self.reranker:
            candidates = self.reranker.hybrid_retrieve(
                vector_store=vector_store,
                search_queries=search_queries,
                top_k=req.retrieval_k
            )
        else:
            candidates = []
            for sq in search_queries:
                results = vector_store.similarity_search_with_relevance_scores(sq, k=req.retrieval_k)
                for rank, (doc, score) in enumerate(results):
                    candidates.append({
                        "rank": rank + 1,
                        "document": doc,
                        "rrf_score": round(float(score), 4)
                    })

        # Step 3: Semantic Reranking
        if req.use_reranker and self.reranker:
            final_retrieved = self.reranker.rerank_with_llm(
                query=question,
                candidates=candidates,
                llm=llm,
                top_k=req.rerank_k
            )
        else:
            final_retrieved = candidates[:req.rerank_k]

        # Step 4: Context assembly & LLM generation
        context_blocks = []
        for idx, item in enumerate(final_retrieved):
            doc = item["document"]
            parent_ctx = doc.metadata.get("parent_content")
            if parent_ctx:
                context_blocks.append(f"--- Passage {idx+1} (Parent Context) ---\n{parent_ctx}")
            else:
                context_blocks.append(f"--- Passage {idx+1} ---\n{doc.page_content}")

        combined_context = "\n\n".join(context_blocks)

        prompt_template = ChatPromptTemplate.from_template("""
You are an expert AI documentation assistant.
Answer the user's question clearly, thoroughly, and accurately based ONLY on the provided context below.
If the information is not present in the context, explicitly state: "I don't know based on the provided document."

Context Passages:
{context}

User Question:
{question}

Formulate a concise, well-structured response with key bullet points where applicable.
""")

        final_prompt = prompt_template.invoke({
            "context": combined_context,
            "question": question
        })

        try:
            answer_text, actual_model_used = self.invoke_llm_with_fallback(
                final_prompt,
                primary_model=req.model_choice,
                temperature=req.temperature
            )
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                answer_text = (
                    "⚠️ API Quota Exceeded (429 Rate Limit).\n\n"
                    "Google Gemini free tier quota limit was reached. Please wait ~30 seconds before retrying."
                )
            else:
                answer_text = f"⚠️ Error generating answer: {err_str}"
            actual_model_used = req.model_choice

        elapsed_time = round(time.time() - start_time, 2)

        # Prepare source passages response
        sources_list: List[SourcePassage] = []
        for idx, item in enumerate(final_retrieved):
            doc = item["document"]
            sources_list.append(
                SourcePassage(
                    rank=idx + 1,
                    content=doc.page_content,
                    page=doc.metadata.get("page", 1),
                    relevance_score=item.get("relevance_score"),
                    rrf_score=item.get("rrf_score")
                )
            )

        pipeline_debug = PipelineDebugInfo(
            query_mode=req.query_mode,
            original_query=question,
            expanded_queries=search_queries,
            hyde_document=rewrite_result.get("hyde_document"),
            step_back_query=rewrite_result.get("step_back_query"),
            candidate_count=len(candidates),
            reranked_count=len(final_retrieved)
        )

        return QueryResponse(
            status="success",
            question=question,
            answer=answer_text,
            model_used=actual_model_used,
            elapsed_time_seconds=elapsed_time,
            sources=sources_list,
            pipeline_debug=pipeline_debug
        )


# Global service instance
rag_service_instance = RAGService()
