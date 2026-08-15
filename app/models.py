"""
Pydantic models used for request validation and response shaping.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ChatMode(str, Enum):
    """Explicit mode the frontend can request. AUTO lets the backend decide."""

    AUTO = "auto"
    GENERAL = "general"
    RAG = "rag"
    WEB = "web"


class ChatMessage(BaseModel):
    """A single turn in the conversation, used for context passed to Groq."""

    role: str = Field(..., description="Either 'user' or 'assistant'.")
    content: str


class ChatRequest(BaseModel):
    """Incoming payload for the /api/chat endpoint."""

    question: str = Field(..., min_length=1, max_length=4000)
    mode: ChatMode = Field(default=ChatMode.AUTO)
    document_id: Optional[str] = Field(
        default=None,
        description="If set, restrict RAG retrieval to this single document.",
    )
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class SourceChunk(BaseModel):
    """A single retrieved source used to ground a RAG answer."""

    document_id: str
    document_name: str
    chunk_id: str
    page: Optional[int] = None
    score: float
    snippet: str


class ChatResponse(BaseModel):
    """Response returned by both General AI and RAG modes."""

    answer: str
    mode_used: ChatMode
    sources: list[SourceChunk] = Field(default_factory=list)
    web_sources: list[WebSource] = Field(default_factory=list)
    found_in_documents: Optional[bool] = None


class WebSource(BaseModel):
    """A single web result used to ground a Web Search mode answer."""

    title: str
    url: str
    snippet: str


class ImageGenerateRequest(BaseModel):
    """Incoming payload for the /api/image/generate endpoint."""

    prompt: str = Field(..., min_length=1, max_length=2000)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)


class ImageGenerateResponse(BaseModel):
    image_url: str
    prompt: str


class DocumentInfo(BaseModel):
    """Metadata describing an uploaded/indexed document."""

    document_id: str
    document_name: str
    num_chunks: int
    uploaded_at: str
    size_bytes: int


class UploadResponse(BaseModel):
    """Response returned after a document has been processed and indexed."""

    document: DocumentInfo
    message: str


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    groq_configured: bool
    pinecone_configured: bool
    embedding_model: str
    embedding_dimension: Optional[int] = None
    pinecone_index_ready: bool
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
