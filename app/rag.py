"""
RAG orchestration layer.

Ties together PDF extraction, chunking, embedding, Pinecone storage,
retrieval, and Groq generation. Also maintains a small local JSON
registry of document metadata (name, chunk count, upload time, size)
so the frontend can list and delete documents without needing to
know anything about Pinecone internals.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app import groq_client, pinecone_db, web_search
from app.chunking import chunk_pages
from app.config import get_settings
from app.embeddings import embed_query, embed_texts
from app.models import ChatMode, ChatMessage, DocumentInfo, SourceChunk, WebSource
from app.pdf_processor import extract_text_from_pdf, PDFProcessingError
from app.web_search import WebSearchError

logger = logging.getLogger("rag_chatbot.rag")

_registry_lock = threading.Lock()
_REGISTRY_PATH = Path("data/documents/registry.json")


class RAGError(Exception):
    """Raised for any user-facing RAG pipeline failure."""


# --------------------------------------------------------------------------
# Document registry (local metadata store; the vectors themselves live in
# Pinecone — this only tracks what the frontend needs to display/manage).
# --------------------------------------------------------------------------

def _load_registry() -> dict[str, dict]:
    if not _REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read document registry, starting fresh: %s", exc)
        return {}


def _save_registry(registry: dict[str, dict]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def list_documents() -> list[DocumentInfo]:
    with _registry_lock:
        registry = _load_registry()
    return [
        DocumentInfo(
            document_id=doc_id,
            document_name=meta["document_name"],
            num_chunks=meta["num_chunks"],
            uploaded_at=meta["uploaded_at"],
            size_bytes=meta["size_bytes"],
        )
        for doc_id, meta in registry.items()
    ]


def find_duplicate_document(document_name: str, size_bytes: int) -> DocumentInfo | None:
    """Detect an existing document with the same name and size (safe re-indexing check)."""
    for doc in list_documents():
        if doc.document_name == document_name and doc.size_bytes == size_bytes:
            return doc
    return None


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------

def ingest_pdf(file_bytes: bytes, original_filename: str, replace_existing: bool = False) -> DocumentInfo:
    """
    Full ingestion pipeline: extract -> clean & chunk -> embed -> upsert
    to Pinecone -> register locally.
    """
    settings = get_settings()

    duplicate = find_duplicate_document(original_filename, len(file_bytes))
    if duplicate and not replace_existing:
        raise RAGError(
            f"A document named '{original_filename}' with the same content is already "
            "indexed. Delete it first or re-upload with replace enabled to re-index."
        )
    if duplicate and replace_existing:
        delete_document(duplicate.document_id)

    try:
        pages = extract_text_from_pdf(file_bytes)
    except PDFProcessingError as exc:
        raise RAGError(str(exc)) from exc

    chunks = chunk_pages(pages, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    if not chunks:
        raise RAGError("No usable text chunks could be produced from this document.")

    document_id = uuid.uuid4().hex[:16]
    chunk_ids = [f"chunk-{c.chunk_index}" for c in chunks]
    chunk_texts = [c.text for c in chunks]
    chunk_pages_list = [c.page for c in chunks]

    try:
        vectors = embed_texts(chunk_texts)
    except Exception as exc:  # noqa: BLE001
        raise RAGError(f"Failed to generate embeddings: {exc}") from exc

    try:
        pinecone_db.upsert_chunks(
            document_id=document_id,
            document_name=original_filename,
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
            chunk_vectors=vectors,
            chunk_pages=chunk_pages_list,
        )
    except Exception as exc:  # noqa: BLE001
        raise RAGError(f"Failed to store document in the vector database: {exc}") from exc

    doc_info = DocumentInfo(
        document_id=document_id,
        document_name=original_filename,
        num_chunks=len(chunks),
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        size_bytes=len(file_bytes),
    )

    with _registry_lock:
        registry = _load_registry()
        registry[document_id] = doc_info.model_dump()
        _save_registry(registry)

    logger.info("Ingested document '%s' (%s) with %d chunks.", original_filename, document_id, len(chunks))
    return doc_info


def delete_document(document_id: str) -> bool:
    with _registry_lock:
        registry = _load_registry()
        if document_id not in registry:
            return False
        del registry[document_id]
        _save_registry(registry)

    try:
        pinecone_db.delete_document(document_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to delete vectors for document '%s': %s", document_id, exc)
        raise RAGError(f"Failed to remove document vectors: {exc}") from exc

    return True


# --------------------------------------------------------------------------
# Retrieval + answer generation
# --------------------------------------------------------------------------

def _history_to_dicts(history: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in history]


def retrieve_context(question: str, document_id: str | None = None) -> list[pinecone_db.RetrievedChunk]:
    settings = get_settings()
    query_vector = embed_query(question)
    matches = pinecone_db.query_similar_chunks(
        query_vector=query_vector,
        top_k=settings.top_k,
        document_id=document_id,
    )
    return [m for m in matches if m.score >= settings.similarity_threshold]


def _to_source_chunks(matches: list[pinecone_db.RetrievedChunk]) -> list[SourceChunk]:
    sources = []
    for match in matches:
        snippet = match.text[:280] + ("..." if len(match.text) > 280 else "")
        sources.append(
            SourceChunk(
                document_id=match.document_id,
                document_name=match.document_name,
                chunk_id=match.chunk_id,
                page=match.page,
                score=round(match.score, 4),
                snippet=snippet,
            )
        )
    return sources


def _to_web_sources(results: list[web_search.WebResult]) -> list[WebSource]:
    return [WebSource(title=r.title, url=r.url, snippet=r.snippet) for r in results]


def answer_question(
    question: str,
    mode: ChatMode,
    history: list[ChatMessage],
    document_id: str | None = None,
) -> tuple[str, ChatMode, list[SourceChunk], list[WebSource], bool | None]:
    """
    Route the question to General AI, RAG, or Web Search mode and
    produce an answer.

    Returns (answer, mode_actually_used, sources, web_sources, found_in_documents).
    """
    history_dicts = _history_to_dicts(history)

    if mode == ChatMode.GENERAL:
        answer = groq_client.generate_general_answer(question, history_dicts)
        return answer, ChatMode.GENERAL, [], [], None

    if mode == ChatMode.WEB:
        try:
            results = web_search.search_web(question, max_results=get_settings().web_search_max_results)
        except WebSearchError as exc:
            return str(exc), ChatMode.WEB, [], [], None

        if not results:
            answer = "I couldn't find any current web results for that question. Try rephrasing it."
            return answer, ChatMode.WEB, [], [], None

        context = "\n\n---\n\n".join(f"[{r.title}]({r.url})\n{r.snippet}" for r in results)
        answer = groq_client.generate_web_answer(question, context, history_dicts)
        return answer, ChatMode.WEB, [], _to_web_sources(results), None

    if mode == ChatMode.AUTO:
        has_documents = len(list_documents()) > 0
        wants_documents = has_documents and groq_client.classify_intent_needs_documents(question)
        if not wants_documents:
            answer = groq_client.generate_general_answer(question, history_dicts)
            return answer, ChatMode.GENERAL, [], [], None
        mode = ChatMode.RAG

    # RAG mode (explicit or resolved from AUTO)
    matches = retrieve_context(question, document_id=document_id)
    sources = _to_source_chunks(matches)

    if not matches:
        answer = (
            "I couldn't find relevant information about that in the uploaded "
            "documents. Please try rephrasing your question, upload a document "
            "that covers this topic, or switch to General AI mode."
        )
        return answer, ChatMode.RAG, [], [], False

    context = "\n\n---\n\n".join(
        f"[Source: {m.document_name}, page {m.page or 'n/a'}]\n{m.text}" for m in matches
    )
    answer = groq_client.generate_rag_answer(question, context, history_dicts)
    return answer, ChatMode.RAG, sources, [], True
