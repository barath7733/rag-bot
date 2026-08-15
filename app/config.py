"""
Central application configuration.

All configuration is loaded from environment variables (via a `.env`
file in development, or real environment variables in production).
Nothing sensitive is hard-coded, and nothing sensitive is ever sent
to the frontend.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env as early as possible so BaseSettings can see the values.
load_dotenv()

logger = logging.getLogger("rag_chatbot")


class Settings(BaseSettings):
    """
    Strongly-typed application settings.

    Every value is read from the environment. Sensible defaults are
    provided only for non-sensitive, purely operational values
    (chunk size, top-k, etc.) — never for API keys.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Groq (LLM) -------------------------------------------------
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    # --- Pinecone (vector database) ---------------------------------
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="rag-chatbot-index", alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")

    # --- Embeddings ---------------------------------------------------
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")

    # --- Web search (current/real-time info) --------------------------
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    web_search_max_results: int = Field(default=5, alias="WEB_SEARCH_MAX_RESULTS")

    # --- Image generation ----------------------------------------------
    image_gen_model: str = Field(default="flux", alias="IMAGE_GEN_MODEL")

    # --- Chunking / retrieval ----------------------------------------
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=5, alias="TOP_K")
    similarity_threshold: float = Field(default=0.30, alias="SIMILARITY_THRESHOLD")

    # --- Uploads --------------------------------------------------
    max_upload_size_mb: int = Field(default=20, alias="MAX_UPLOAD_SIZE_MB")
    upload_dir: str = Field(default="data/pdfs", alias="UPLOAD_DIR")

    # --- App / server ---------------------------------------------
    app_env: str = Field(default="development", alias="APP_ENV")
    allowed_origins: str = Field(default="*", alias="ALLOWED_ORIGINS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_must_be_smaller_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 800)
        if v >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return v

    @property
    def cors_origins(self) -> list[str]:
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()


def configure_logging() -> None:
    """Configure root logging once, without ever logging secrets."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def validate_required_settings() -> list[str]:
    """
    Return a list of human-readable warnings for missing critical
    configuration. Called at startup so misconfiguration is obvious
    immediately rather than surfacing as a confusing runtime error.
    """
    settings = get_settings()
    warnings: list[str] = []
    if not settings.groq_api_key:
        warnings.append("GROQ_API_KEY is not set — General AI mode and RAG answer generation will fail.")
    if not settings.pinecone_api_key:
        warnings.append("PINECONE_API_KEY is not set — Document/RAG mode will fail.")
    if not settings.tavily_api_key:
        warnings.append("TAVILY_API_KEY is not set — Web Search mode will fail.")
    return warnings
