"""
Embedding provider with OpenAI or free local fallback.

- OpenAI: text-embedding-3-small (needs API key + small credit balance)
- Local: Chroma default ONNX MiniLM (no key, good for assessment)
"""

from __future__ import annotations

import logging
from typing import Protocol

from legal_rag import config as cfg

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    @property
    def name(self) -> str: ...


class OpenAIEmbeddingProvider:
    def __init__(self) -> None:
        if not cfg.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
        from openai import OpenAI

        self._client = OpenAI(api_key=cfg.OPENAI_API_KEY)
        self._model = cfg.EMBEDDING_MODEL

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class ChromaDefaultEmbeddingProvider:
    """Free local embeddings via Chroma (ONNX MiniLM — no API key)."""

    def __init__(self) -> None:
        from chromadb.utils import embedding_functions

        self._ef = embedding_functions.DefaultEmbeddingFunction()

    @property
    def name(self) -> str:
        return "chroma-default:onnx-minilm"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._ef(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._ef([text])[0]


def get_embedding_provider() -> EmbeddingProvider:
    """
    auto: OpenAI if key present, else local Chroma embeddings.
    openai: force OpenAI (error without key)
    local: force local embeddings
    """
    backend = cfg.EMBEDDING_BACKEND.lower()
    if backend == "openai":
        return OpenAIEmbeddingProvider()
    if backend == "local":
        return ChromaDefaultEmbeddingProvider()
    # auto
    if cfg.OPENAI_API_KEY:
        try:
            logger.info("Using OpenAI embeddings (%s)", cfg.EMBEDDING_MODEL)
            return OpenAIEmbeddingProvider()
        except (ImportError, ValueError) as exc:
            logger.warning("OpenAI embeddings unavailable (%s); using local", exc)
    logger.info("Using free local embeddings (Chroma MiniLM)")
    return ChromaDefaultEmbeddingProvider()
