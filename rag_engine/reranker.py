import math
import re
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class HybridReranker:
    def __init__(self, chunks: List[Document]):
        self.chunks = chunks
        self.bm25 = None
        self._build_bm25_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def _build_bm25_index(self):
        if not self.chunks:
            return
        corpus = [self._tokenize(doc.page_content) for doc in self.chunks]
        self.bm25 = BM25Okapi(corpus)

    def hybrid_retrieve(
        self,
        vector_store,
        search_queries: List[str],
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval using ChromaDB vector search + BM25 keyword search,
        fused via Reciprocal Rank Fusion (RRF).
        """
        if not self.chunks:
            return []

        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # 1. Vector Search across all expanded queries
        for query in search_queries:
            results = vector_store.similarity_search_with_relevance_scores(query, k=top_k)
            for rank, (doc, score) in enumerate(results):
                content_key = doc.page_content.strip()
                doc_map[content_key] = doc
                # RRF score
                rrf_val = 1.0 / (60 + (rank + 1))
                doc_scores[content_key] = doc_scores.get(content_key, 0.0) + rrf_val

        # 2. BM25 Search across all search queries
        if self.bm25:
            for query in search_queries:
                tokens = self._tokenize(query)
                bm25_scores = self.bm25.get_scores(tokens)
                
                # Top BM25 indices
                top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
                for rank, idx in enumerate(top_indices):
                    if idx < len(self.chunks):
                        doc = self.chunks[idx]
                        content_key = doc.page_content.strip()
                        doc_map[content_key] = doc
                        rrf_val = 1.0 / (60 + (rank + 1))
                        doc_scores[content_key] = doc_scores.get(content_key, 0.0) + rrf_val

        # Sort combined results by RRF score
        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        
        candidates = []
        for idx, key in enumerate(sorted_keys[:top_k]):
            candidates.append({
                "rank": idx + 1,
                "document": doc_map[key],
                "rrf_score": round(doc_scores[key], 4)
            })
            
        return candidates

    def rerank_with_llm(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        llm: ChatGoogleGenerativeAI,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Uses LLM as a Semantic Cross-Encoder to score candidate relevance (0-100)
        and re-order context passages in a SINGLE batched LLM API call to conserve API quota.
        """
        if not candidates:
            return []

        # If only 1 candidate, no complex reranking needed
        if len(candidates) == 1:
            cand = candidates[0].copy()
            cand["relevance_score"] = 90
            return [cand]

        passages_formatted = []
        for idx, cand in enumerate(candidates):
            text_snippet = cand["document"].page_content[:400].replace("\n", " ")
            passages_formatted.append(f"Passage [{idx + 1}]: {text_snippet}")
        
        combined_passages = "\n\n".join(passages_formatted)

        prompt = ChatPromptTemplate.from_template("""
Score the relevance of each passage below to the question on an integer scale from 0 to 100 (where 100 means highly relevant and 0 means completely irrelevant).

Question: {question}

Passages to score:
{passages}

Output ONLY a comma-separated list of integer scores corresponding to each Passage in sequential order (e.g. 85, 90, 45, 60). Do NOT output any other text or explanation.
""")

        scores = []
        try:
            formatted = prompt.format(question=query, passages=combined_passages)
            
            # Invoke with candidate model fallback list
            raw_text = None
            fallback_list = [getattr(llm, "model", "gemini-3.6-flash")] + [
                m for m in ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
                if m != getattr(llm, "model", "")
            ]
            for m_name in fallback_list:
                try:
                    active_llm = ChatGoogleGenerativeAI(model=m_name)
                    resp = active_llm.invoke(formatted)
                    if isinstance(resp.content, list):
                        raw_text = "".join(
                            item.get("text", "") if isinstance(item, dict) else str(getattr(item, "text", item))
                            for item in resp.content
                        )
                    else:
                        raw_text = str(resp.content)
                    break
                except Exception as ex:
                    continue
                    
            if not raw_text:
                raise RuntimeError("All LLM reranking attempts failed.")
            
            # Extract all integers from response
            extracted = [int(s) for s in re.findall(r'\b\d{1,3}\b', raw_text)]
            
            # Validate extracted scores count
            if len(extracted) == len(candidates):
                scores = [max(0, min(100, val)) for val in extracted]
            else:
                # If partial matches found, take first N or pad with fallbacks
                for i in range(len(candidates)):
                    if i < len(extracted):
                        scores.append(max(0, min(100, extracted[i])))
                    else:
                        scores.append(max(30, 80 - i * 10))
        except Exception as e:
            print(f"Notice: Batched LLM reranking encountered error: {e}. Falling back to RRF scores.")
            # Fallback score based on original RRF rank
            scores = [max(30, 90 - i * 10) for i in range(len(candidates))]


        reranked = []
        for idx, cand in enumerate(candidates):
            cand_copy = cand.copy()
            cand_copy["relevance_score"] = scores[idx] if idx < len(scores) else (90 - idx * 10)
            reranked.append(cand_copy)

        # Sort by relevance_score descending
        reranked.sort(key=lambda x: x["relevance_score"], reverse=True)
        return reranked[:top_k]

