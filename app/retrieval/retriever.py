"""
Document retriever for Rift.

Wraps a ChromaDB vector store as a LangChain retriever with configurable
top-k similarity search.
"""

from __future__ import annotations

import logging

from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever

from config.settings import TOP_K_RESULTS

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Create a LangChain-compatible retriever from a Chroma vector store."""

    def __init__(self, top_k: int = TOP_K_RESULTS) -> None:
        """
        Initialise the retriever factory.

        Args:
            top_k: Number of most-similar chunks to return per query.
        """
        self._top_k = top_k

    def get_retriever(self, vectorstore: Chroma) -> BaseRetriever:
        """
        Return a retriever backed by *vectorstore*.

        Args:
            vectorstore: A :class:`Chroma` vector store instance.

        Returns:
            A LangChain :class:`BaseRetriever` configured for similarity
            search with ``top_k`` results.
        """
        logger.info("Creating retriever with top_k=%d.", self._top_k)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self._top_k},
        )

        return retriever
