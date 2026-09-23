from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., description="User's query string")
    chunk_strategy: Optional[str] = Field("Recursive Character", description="Chunking strategy name")
    query_mode: Optional[str] = Field("Multi-Query Expansion", description="Query rewriting mode")
    use_hybrid: Optional[bool] = Field(True, description="Enable BM25 + Vector hybrid search")
    use_reranker: Optional[bool] = Field(True, description="Enable LLM cross-encoder reranker")
    model_choice: Optional[str] = Field("gemini-3.6-flash", description="Gemini model name")
    temperature: Optional[float] = Field(0.0, description="LLM sampling temperature")
    retrieval_k: Optional[int] = Field(6, description="Candidate retrieval count")
    rerank_k: Optional[int] = Field(3, description="Top reranked count")


class SourcePassage(BaseModel):
    rank: int
    content: str
    page: int
    relevance_score: Optional[float] = None
    rrf_score: Optional[float] = None


class PipelineDebugInfo(BaseModel):
    query_mode: str
    original_query: str
    expanded_queries: List[str]
    hyde_document: Optional[str] = None
    step_back_query: Optional[str] = None
    candidate_count: int
    reranked_count: int


class QueryResponse(BaseModel):
    status: str = "success"
    question: str
    answer: str
    model_used: str
    elapsed_time_seconds: float
    sources: List[SourcePassage]
    pipeline_debug: PipelineDebugInfo


class IngestResponse(BaseModel):
    status: str = "success"
    message: str
    document_name: str
    chunk_strategy: str
    total_chunks: int


class StatusResponse(BaseModel):
    status: str = "success"
    active_document: str
    chunk_strategy: str
    total_chunks: int
