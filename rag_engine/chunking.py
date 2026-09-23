import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

CHUNKING_STRATEGIES = [
    "Recursive Character",
    "Fixed Size",
    "Sentence-Based",
    "Paragraph / Semantic",
    "Parent-Child (Hierarchical)"
]

def create_chunks(
    documents: List[Document],
    strategy: str = "Recursive Character",
    chunk_size: int =   512,
    chunk_overlap: int = 75
) -> List[Document]:
    """ 
    Splits documents into text chunks according to the specified chunking strategy.
    Adds metadata tags indicating chunk index, strategy, and parent relations.
    """
    if not documents:
        return []

    chunks: List[Document] = []

    if strategy == "Fixed Size":
        splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = splitter.split_documents(documents)

    elif strategy == "Sentence-Based":
        # Group sentences until chunk_size is reached
        for doc_idx, doc in enumerate(documents):
            sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)
            current_chunk = ""
            start_sentence_idx = 0
            
            for s_idx, sentence in enumerate(sentences):
                if len(current_chunk) + len(sentence) + 1 <= chunk_size or not current_chunk:
                    current_chunk = (current_chunk + " " + sentence).strip()
                else:
                    meta = doc.metadata.copy()
                    meta.update({
                        "chunk_strategy": strategy,
                        "chunk_id": len(chunks)
                    })
                    chunks.append(Document(page_content=current_chunk, metadata=meta))
                    
                    # Apply overlap by taking trailing sentences
                    overlap_text = ""
                    for back_idx in range(max(start_sentence_idx, s_idx - 2), s_idx):
                        if len(overlap_text) + len(sentences[back_idx]) <= chunk_overlap:
                            overlap_text = (overlap_text + " " + sentences[back_idx]).strip()
                    
                    current_chunk = (overlap_text + " " + sentence).strip()
                    start_sentence_idx = s_idx
            
            if current_chunk:
                meta = doc.metadata.copy()
                meta.update({
                    "chunk_strategy": strategy,
                    "chunk_id": len(chunks)
                })
                chunks.append(Document(page_content=current_chunk, metadata=meta))

    elif strategy == "Paragraph / Semantic":
        for doc in documents:
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', doc.page_content) if p.strip()]
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 <= chunk_size or not current_chunk:
                    current_chunk = (current_chunk + "\n\n" + para).strip()
                else:
                    meta = doc.metadata.copy()
                    meta.update({
                        "chunk_strategy": strategy,
                        "chunk_id": len(chunks)
                    })
                    chunks.append(Document(page_content=current_chunk, metadata=meta))
                    current_chunk = para
            
            if current_chunk:
                meta = doc.metadata.copy()
                meta.update({
                    "chunk_strategy": strategy,
                    "chunk_id": len(chunks)
                })
                chunks.append(Document(page_content=current_chunk, metadata=meta))

    elif strategy == "Parent-Child (Hierarchical)":
        # Parent chunk size is larger (e.g., 2x chunk_size)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(chunk_size * 2, 1500),
            chunk_overlap=chunk_overlap * 2
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        parent_docs = parent_splitter.split_documents(documents)
        for parent_idx, parent_doc in enumerate(parent_docs):
            child_docs = child_splitter.split_documents([parent_doc])
            for child_doc in child_docs:
                meta = child_doc.metadata.copy()
                meta.update({
                    "chunk_strategy": strategy,
                    "parent_id": parent_idx,
                    "parent_content": parent_doc.page_content,
                    "chunk_id": len(chunks)
                })
                chunks.append(Document(page_content=child_doc.page_content, metadata=meta))

    else:
        # Default: Recursive Character
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(documents)

    # Attach common metadata tags
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_strategy"] = strategy
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks
