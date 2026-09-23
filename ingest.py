import os
import sys
import shutil
from typing import List, Tuple
from dotenv import load_dotenv

# Ensure stdout supports UTF-8 on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_engine.chunking import create_chunks, CHUNKING_STRATEGIES

load_dotenv()

PDF_PATH = "documents/myres.pdf"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_documents"

def load_document(file_path: str) -> List[Document]:
    """Loads PDF or TXT document into LangChain Document objects."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
    return loader.load()

def ingest_document(
    file_path: str = PDF_PATH,
    chroma_path: str = CHROMA_PATH,
    strategy: str = "Recursive Character",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    clear_existing: bool = True
) -> Tuple[Chroma, List[Document]]:
    """
    Ingests document, applies specified chunking strategy, embeds, and stores in ChromaDB.
    Returns (vector_store, chunks).
    """
    print(f"📄 Loading document: {file_path}")
    documents = load_document(file_path)
    print(f"✅ Loaded {len(documents)} pages/sections.")

    # Apply selected chunking strategy
    chunks = create_chunks(
        documents,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print(f"✂️ Created {len(chunks)} chunks using strategy: '{strategy}' (Size: {chunk_size}, Overlap: {chunk_overlap})")

    # Clear existing collection if requested
    if clear_existing:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            temp_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=chroma_path
            )
            temp_store.delete_collection()
            print(f"🧹 Cleared existing Chroma collection '{COLLECTION_NAME}'")
        except Exception as e:
            print(f"Notice clearing collection: {e}")

    # Create Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # Store in ChromaDB
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=chroma_path
    )

    if chunks:
        vector_store.add_documents(chunks)
        print("🚀 Documents successfully stored in ChromaDB!")

    return vector_store, chunks

if __name__ == "__main__":
    if os.path.exists(PDF_PATH):
        ingest_document(PDF_PATH, strategy="Recursive Character")
    else:
        print(f"Sample file {PDF_PATH} not found. Ready for Streamlit dynamic ingestion.")