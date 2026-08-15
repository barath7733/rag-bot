"""
FastAPI application entry point.

Exposes endpoints for chat (general + RAG), document upload/list/delete,
and a health check. Serves the static frontend from /static and /templates.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import image_gen, rag
from app.config import configure_logging, get_settings, validate_required_settings
from app.embeddings import get_embedding_dimension
from app.image_gen import ImageGenerationError
from app.models import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocumentInfo,
    ErrorResponse,
    HealthResponse,
    ImageGenerateRequest,
    ImageGenerateResponse,
    UploadResponse,
)
from app.pinecone_db import is_index_ready
from app.rag import RAGError
from app.web_search import WebSearchError

configure_logging()
logger = logging.getLogger("rag_chatbot.main")

settings = get_settings()

app = FastAPI(
    title="General-Purpose AI Assistant + RAG Document Intelligence",
    description="A chatbot that answers general questions via Groq and grounds document-related "
                "questions in retrieved context using Pinecone vector search.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

ALLOWED_CONTENT_TYPES = {"application/pdf"}
MAX_UPLOAD_BYTES = settings.max_upload_size_mb * 1024 * 1024


@app.on_event("startup")
async def on_startup() -> None:
    warnings = validate_required_settings()
    for warning in warnings:
        logger.warning(warning)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path("data/documents").mkdir(parents=True, exist_ok=True)
    logger.info("Application startup complete. Groq model=%s, Embedding model=%s",
                settings.groq_model, settings.embedding_model)


# --------------------------------------------------------------------------
# Error handling — never leak stack traces or secrets to the client.
# --------------------------------------------------------------------------

@app.exception_handler(RAGError)
async def rag_error_handler(request: Request, exc: RAGError) -> JSONResponse:
    return JSONResponse(status_code=422, content=ErrorResponse(error=str(exc)).model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="An internal error occurred. Please try again.",
        ).model_dump(),
    )


# --------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


# --------------------------------------------------------------------------
# Health / status
# --------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    warnings = validate_required_settings()
    embedding_dimension: int | None = None
    try:
        embedding_dimension = get_embedding_dimension()
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Embedding model could not be loaded: {exc}")

    pinecone_ready = False
    if settings.pinecone_api_key and embedding_dimension is not None:
        pinecone_ready = is_index_ready()

    return HealthResponse(
        status="ok" if not warnings else "degraded",
        groq_configured=bool(settings.groq_api_key),
        pinecone_configured=bool(settings.pinecone_api_key),
        embedding_model=settings.embedding_model,
        embedding_dimension=embedding_dimension,
        pinecone_index_ready=pinecone_ready,
        warnings=warnings,
    )


@app.get("/api/features")
async def features() -> dict:
    """Lightweight capability flags the frontend uses to enable/disable UI features."""
    return {
        "web_search_enabled": bool(settings.tavily_api_key),
        "image_generation_enabled": True,  # Pollinations.ai requires no key
    }


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        answer, mode_used, sources, web_sources, found = rag.answer_question(
            question=payload.question,
            mode=payload.mode,
            history=payload.history,
            document_id=payload.document_id,
        )
    except (RuntimeError, WebSearchError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(
        answer=answer,
        mode_used=mode_used,
        sources=sources,
        web_sources=web_sources,
        found_in_documents=found,
    )


# --------------------------------------------------------------------------
# Image generation
# --------------------------------------------------------------------------

@app.post("/api/image/generate", response_model=ImageGenerateResponse)
async def generate_image(payload: ImageGenerateRequest) -> ImageGenerateResponse:
    try:
        url = image_gen.generate_image(payload.prompt, width=payload.width, height=payload.height)
    except ImageGenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ImageGenerateResponse(image_url=url, prompt=payload.prompt)


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------

@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    replace_existing: bool = Form(default=False),
) -> UploadResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Sanitize filename to avoid path traversal / unsafe characters.
    safe_name = Path(file.filename or f"document-{uuid.uuid4().hex[:8]}.pdf").name

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the maximum allowed size of {settings.max_upload_size_mb} MB.",
        )

    try:
        doc_info = rag.ingest_pdf(file_bytes, safe_name, replace_existing=replace_existing)
    except RAGError:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return UploadResponse(document=doc_info, message=f"'{safe_name}' processed and indexed successfully.")


@app.get("/api/documents", response_model=list[DocumentInfo])
async def get_documents() -> list[DocumentInfo]:
    return rag.list_documents()


@app.delete("/api/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str) -> DeleteResponse:
    try:
        deleted = rag.delete_document(document_id)
    except RAGError:
        raise
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DeleteResponse(document_id=document_id, deleted=True, message="Document deleted successfully.")
