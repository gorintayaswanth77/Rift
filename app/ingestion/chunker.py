"""
Document chunker using LangChain's RecursiveCharacterTextSplitter.

Splits loaded documents into smaller, overlapping chunks suitable for
embedding and retrieval while preserving metadata.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Split documents into overlapping chunks for vector storage."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        """
        Initialise the chunker.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between
                consecutive chunks.
        """
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of Documents into smaller chunks.

        Each resulting chunk inherits the metadata of its parent document
        with an additional ``chunk_index`` field.

        Args:
            documents: Documents produced by :class:`DocumentLoader`.

        Returns:
            A flat list of chunked :class:`Document` objects.
        """
        if not documents:
            logger.warning("No documents provided to chunk.")
            return []

        chunks = self._splitter.split_documents(documents)
        logger.info(
            "Chunked %d document(s) into %d chunk(s) "
            "(size=%d, overlap=%d).",
            len(documents),
            len(chunks),
            self._splitter._chunk_size,
            self._splitter._chunk_overlap,
        )
        return chunks
