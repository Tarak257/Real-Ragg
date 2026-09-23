import os
import sys
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models import (
    QueryRequest,
    QueryResponse,
    IngestResponse,
    StatusResponse
)
from backend.rag_service import rag_service_instance

app = FastAPI(
    title="Build_RAG Engine API",
    description="Multi-Strategy RAG Engine Backend API powering Document Q&A",
    version="2.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "Build_RAG Engine"}


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    return rag_service_instance.get_status()


@app.post("/api/upload", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_strategy: str = Form("Recursive Character"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    try:
        content = await file.read()
        return rag_service_instance.process_document(
            file_bytes=content,
            filename=file.filename,
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        return rag_service_instance.query(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve static web frontend if directory exists
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def serve_index():
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Build_RAG API is running. Add index.html to static/ directory."}
