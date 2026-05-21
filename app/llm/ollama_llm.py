"""
Ollama LLM wrapper for Rift.

Provides a factory for creating a LangChain-compatible Ollama LLM instance
configured via the application settings.
"""

from __future__ import annotations

import logging

from langchain_ollama import ChatOllama

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


class LLMModel:
    """Factory for Ollama-backed LLM instances."""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        """
        Initialise the LLM factory.

        Args:
            model: Ollama model name (e.g. ``"mistral"``).
            base_url: Ollama server URL.
        """
        self._model = model
        self._base_url = base_url

    def get_llm(self) -> ChatOllama:
        """
        Return a ready-to-use :class:`ChatOllama` instance.

        Returns:
            A configured :class:`ChatOllama` object.

        Raises:
            RuntimeError: If the Ollama server is unreachable or the
                requested model is not available.
        """
        logger.info(
            "Initialising Ollama LLM (model=%s, base_url=%s).",
            self._model,
            self._base_url,
        )

        try:
            llm = ChatOllama(
                model=self._model,
                base_url=self._base_url,
                temperature=0.0,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise Ollama LLM "
                f"(model={self._model}, url={self._base_url}): {exc}"
            ) from exc

        logger.info("Ollama LLM initialised successfully.")
        return llm
