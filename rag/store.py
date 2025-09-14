from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
import shutil

from langchain_ollama import OllamaEmbeddings
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # fallback if package not available
    RecursiveCharacterTextSplitter = None  # type: ignore

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    Chroma = None  # type: ignore


DB_DIR = "rag_db"


def _split_text(text: str) -> List[str]:
    if RecursiveCharacterTextSplitter is None:
        # naive fallback
        chunk_size = 800
        chunk_overlap = 100
        out = []
        i = 0
        n = len(text)
        while i < n:
            out.append(text[i:i + chunk_size])
            i += chunk_size - chunk_overlap
        return out
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return splitter.split_text(text)


def _ensure_db_dir() -> str:
    os.makedirs(DB_DIR, exist_ok=True)
    return DB_DIR


def get_vectorstore(base_url: str, embed_model: str = "nomic-embed-text"):
    """Return a Chroma vector store with Ollama embeddings (persistent)."""
    if Chroma is None:
        raise RuntimeError("Chroma is not installed. Add 'chromadb' and 'langchain-community' to requirements.")
    _ensure_db_dir()
    embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)
    vs = Chroma(collection_name="docs", persist_directory=DB_DIR, embedding_function=embeddings)
    return vs


def reset_vectorstore() -> None:
    """Delete the persistent vector DB directory and recreate it empty."""
    try:
        shutil.rmtree(DB_DIR, ignore_errors=True)
    except Exception:
        pass
    os.makedirs(DB_DIR, exist_ok=True)


def delete_by_filename(base_url: str, filename: str, embed_model: str = "nomic-embed-text") -> int:
    """Delete all chunks where metadata filename matches. Returns number of deleted items (best effort)."""
    if not filename:
        return 0
    vs = get_vectorstore(base_url, embed_model)
    # LangChain Chroma supports where filter deletion
    try:
        vs.delete(where={"filename": filename})
        try:
            vs.persist()
        except Exception:
            pass
        # Chroma does not return count; we cannot easily know exact number here
        return 1
    except Exception:
        return 0


def add_document(base_url: str, text: str, metadata: Optional[Dict[str, Any]] = None, embed_model: str = "nomic-embed-text") -> int:
    """Chunk and add a single text document. Returns number of chunks added."""
    if not text:
        return 0
    vs = get_vectorstore(base_url, embed_model)
    text = (text or "").strip()
    if not text:
        return 0
    chunks = _split_text(text)
    if not chunks:
        # Fallback: index whole text as one chunk
        chunks = [text]
    metadatas = [metadata or {} for _ in chunks]
    vs.add_texts(chunks, metadatas=metadatas)
    try:
        vs.persist()
    except Exception:
        pass
    return len(chunks)


def add_documents(base_url: str, items: List[Dict[str, Any]], embed_model: str = "nomic-embed-text") -> int:
    """Add multiple documents. items = [{"text": str, "metadata": {..}}]. Returns total chunks."""
    total = 0
    for it in items:
        total += add_document(base_url, it.get("text", ""), it.get("metadata"), embed_model)
    return total


def similarity_search(base_url: str, query: str, k: int = 5, embed_model: str = "nomic-embed-text") -> List[Dict[str, Any]]:
    """Return list of {"page_content": str, "metadata": {...}} for top-k matches."""
    vs = get_vectorstore(base_url, embed_model)
    docs = vs.similarity_search(query, k=k)
    out = []
    for d in docs:
        out.append({"page_content": d.page_content, "metadata": getattr(d, "metadata", {}) or {}})
    return out


def list_filenames(base_url: str, embed_model: str = "nomic-embed-text") -> List[str]:
    """Return a sorted list of distinct filenames currently in the index (best effort)."""
    vs = get_vectorstore(base_url, embed_model)
    try:
        raw = vs._collection.get(include=["metadatas"])  # type: ignore[attr-defined]
        metas = raw.get("metadatas", []) if isinstance(raw, dict) else []
        names = []
        for m in metas:
            if isinstance(m, dict) and m.get("filename"):
                names.append(str(m["filename"]))
        return sorted(sorted(set(names)))
    except Exception:
        return []


def similarity_search_filtered(
    base_url: str,
    query: str,
    k: int = 5,
    *,
    filename: Optional[str] = None,
    embed_model: str = "nomic-embed-text",
) -> List[Dict[str, Any]]:
    """Similarity search with optional filename filter.
    Falls back to client-side filter if vectorstore doesn't support server-side filters.
    """
    vs = get_vectorstore(base_url, embed_model)
    docs = []
    if filename:
        try:
            docs = vs.similarity_search(query, k=k, filter={"filename": filename})  # type: ignore[arg-type]
        except Exception:
            # fallback: get more docs and filter in Python
            cand = vs.similarity_search(query, k=max(k * 5, 20))
            docs = [d for d in cand if (getattr(d, "metadata", {}) or {}).get("filename") == filename][:k]
    else:
        docs = vs.similarity_search(query, k=k)
    out: List[Dict[str, Any]] = []
    for d in docs:
        out.append({"page_content": d.page_content, "metadata": getattr(d, "metadata", {}) or {}})
    return out
