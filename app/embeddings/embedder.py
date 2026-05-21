"""
Embedding model wrapper for Rift.

Provides a thin abstraction over HuggingFace sentence-transformer embeddings
so the rest of the application can swap models via configuration only.
"""

from __future__ import annotations

import logging

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Factory for HuggingFace embedding instances."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        """
        Initialise the embedding model wrapper.

        Args:
            model_name: HuggingFace model identifier.  Defaults to the
                value in ``config.settings.EMBEDDING_MODEL``.
        """
        self._model_name = model_name

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Return a ready-to-use :class:`HuggingFaceEmbeddings` instance.

        The model is downloaded and cached automatically by the
        ``sentence-transformers`` library on first use.

        Returns:
            A configured :class:`HuggingFaceEmbeddings` object.
        """
        logger.info("Loading embedding model: %s", self._model_name)
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self._model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{self._model_name}': {exc}"
            ) from exc

        logger.info("Embedding model loaded successfully.")
        return embeddings
