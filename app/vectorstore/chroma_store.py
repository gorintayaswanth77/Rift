"""
ChromaDB vector store wrapper for Rift.

Handles creation, persistence, loading, and deletion of ChromaDB
collections so the rest of the app never interacts with Chroma directly.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

_DEFAULT_COLLECTION = "rift_collection"


class VectorStore:
    """Manage a ChromaDB-backed vector store persisted to disk."""

    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIR,
        collection_name: str = _DEFAULT_COLLECTION,
    ) -> None:
        """
        Initialise the vector-store manager.

        Args:
            persist_directory: Filesystem path where ChromaDB stores its data.
            collection_name: Name of the Chroma collection to use.
        """
        self._persist_directory = persist_directory
        self._collection_name = collection_name

    def store(
        self,
        chunks: List[Document],
        embeddings: Embeddings,
    ) -> Chroma:
        """
        Embed *chunks* and persist them to the ChromaDB collection.

        If the collection already exists it will be extended with the new
        chunks (ChromaDB handles deduplication internally).

        Args:
            chunks: Document chunks to embed and store.
            embeddings: The embedding function to use.

        Returns:
            The :class:`Chroma` vector store instance.
        """
        if not chunks:
            raise ValueError("No chunks provided to store.")

        logger.info(
            "Storing %d chunk(s) in collection '%s' at '%s'.",
            len(chunks),
            self._collection_name,
            self._persist_directory,
        )

        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self._persist_directory,
                collection_name=self._collection_name,
            )
            # Close the client connection to release the file locks immediately on Windows
            if hasattr(vectorstore, "_client") and hasattr(vectorstore._client, "close"):
                try:
                    vectorstore._client.close()
                    logger.info("Closed temporary vectorstore client connection after storing.")
                except Exception as close_exc:
                    logger.warning("Failed to close vectorstore client connection after storing: %s", close_exc)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to store chunks in ChromaDB: {exc}"
            ) from exc

        logger.info("Chunks stored and persisted successfully.")
        return vectorstore

    def load(self, embeddings: Embeddings) -> Chroma:
        """
        Load an existing persisted ChromaDB collection.

        Args:
            embeddings: The embedding function (must match the one used
                during storage).

        Returns:
            The :class:`Chroma` vector store instance.

        Raises:
            FileNotFoundError: If the persist directory does not exist.
        """
        if not Path(self._persist_directory).exists():
            raise FileNotFoundError(
                f"Vector store directory not found: {self._persist_directory}"
            )

        logger.info(
            "Loading vector store from '%s' (collection: '%s').",
            self._persist_directory,
            self._collection_name,
        )

        try:
            vectorstore = Chroma(
                persist_directory=self._persist_directory,
                embedding_function=embeddings,
                collection_name=self._collection_name,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ChromaDB collection: {exc}"
            ) from exc

        logger.info("Vector store loaded successfully.")
        return vectorstore

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        """
        Delete a ChromaDB collection and its persisted data.

        Args:
            collection_name: Name of the collection to delete.  Defaults
                to the collection this instance was initialised with.
        """
        target = collection_name or self._collection_name
        logger.warning("Deleting collection '%s'.", target)

        try:
            from chromadb.api.shared_system_client import SharedSystemClient
            for key in list(SharedSystemClient._identifier_to_system.keys()):
                try:
                    system = SharedSystemClient._identifier_to_system.get(key)
                    if system is not None:
                        logger.info("Force-stopping ChromaDB system in delete_collection: %s", key)
                        system.stop()
                except Exception as sys_exc:
                    logger.warning("Failed to stop system %s: %s", key, sys_exc)
            SharedSystemClient.clear_system_cache()
        except Exception as cache_exc:
            logger.warning("Failed to clear system cache in delete_collection: %s", cache_exc)

        # Use Chroma's client reset API to clear all data.
        # On Windows, shutil.rmtree fails on the persist directory because
        # the OS may still hold brief locks on recently-closed DB files.
        # client.reset() wipes all collections and data without needing to
        # touch the filesystem directly, so we return immediately on success.
        reset_ok = False
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=self._persist_directory,
                settings=Settings(allow_reset=True),
            )
            client.reset()
            client.close()
            reset_ok = True
            logger.info("Chroma database reset successfully.")
        except Exception as exc:
            logger.warning(
                "Failed to reset database via client API: %s. "
                "Falling back to file deletion.",
                exc,
            )

        # Force stop again and clear cache to release any resources held by the reset client itself
        try:
            from chromadb.api.shared_system_client import SharedSystemClient
            for key in list(SharedSystemClient._identifier_to_system.keys()):
                try:
                    system = SharedSystemClient._identifier_to_system.get(key)
                    if system is not None:
                        system.stop()
                except Exception:
                    pass
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass

        if reset_ok:
            # Data is already cleared by reset(). Attempt to remove the
            # persist directory, but do NOT raise an error if it fails —
            # the directory is empty and will be recreated on next use.
            persist_path = Path(self._persist_directory)
            if persist_path.exists():
                try:
                    shutil.rmtree(persist_path)
                    logger.info(
                        "Deleted persist directory: %s",
                        self._persist_directory,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not remove persist directory after reset "
                        "(this is harmless): %s",
                        exc,
                    )
            return

        persist_path = Path(self._persist_directory)
        if persist_path.exists():
            import time
            for attempt in range(5):
                try:
                    shutil.rmtree(persist_path)
                    logger.info("Deleted persist directory: %s", self._persist_directory)
                    break
                except Exception as exc:
                    if attempt == 4:
                        raise RuntimeError(
                            f"Failed to delete collection '{target}': {exc}"
                        ) from exc
                    logger.warning("Retry %d/5: Failed to delete collection directory: %s", attempt + 1, exc)
                    time.sleep(0.5)
        else:
            logger.info(
                "Persist directory '%s' does not exist — nothing to delete.",
                self._persist_directory,
            )
