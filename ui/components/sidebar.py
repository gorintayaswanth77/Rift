"""
Sidebar component for Rift.

Handles file uploading, document listing, and collection management in
the Streamlit sidebar.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List

import streamlit as st

from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import DocumentChunker
from app.embeddings.embedder import EmbeddingModel
from app.vectorstore.chroma_store import VectorStore
from app.utils.helpers import sanitize_filename
from config.settings import UPLOAD_DIR

logger = logging.getLogger(__name__)

_ACCEPTED_TYPES = ["pdf", "docx", "txt"]


def _get_uploaded_files() -> List[str]:
    """Return a sorted list of filenames currently in the upload directory."""
    upload_path = Path(UPLOAD_DIR)
    if not upload_path.exists():
        return []
    return sorted(f.name for f in upload_path.iterdir() if f.is_file())


def _process_uploaded_file(file_path: str) -> bool:
    """
    Run the full ingestion pipeline on a single file.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        loader = DocumentLoader()
        documents = loader.load(file_path)

        chunker = DocumentChunker()
        chunks = chunker.chunk(documents)

        if not chunks:
            st.warning("No text content found in the uploaded file.")
            return False

        embedder = EmbeddingModel()
        embeddings = embedder.get_embeddings()

        store = VectorStore()
        store.store(chunks, embeddings)

        return True
    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", file_path, exc)
        st.error(f"Failed to process file: {exc}")
        return False


def render_sidebar() -> None:
    """Render the Streamlit sidebar with upload and document management."""
    st.sidebar.title("Document Manager")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Choose files to upload",
        type=_ACCEPTED_TYPES,
        accept_multiple_files=True,
        key="file_uploader",
        help="Supported formats: PDF, DOCX, TXT",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            safe_name = sanitize_filename(uploaded_file.name)
            save_path = os.path.join(UPLOAD_DIR, safe_name)

            # Skip if already processed in this session
            processed_key = f"processed_{safe_name}"
            if st.session_state.get(processed_key, False):
                continue

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.sidebar.status(f"Processing {safe_name}...", expanded=True) as status:
                st.write("Loading document...")
                st.write("Chunking text...")
                st.write("Generating embeddings...")
                st.write("Storing in vector database...")

                success = _process_uploaded_file(save_path)

                if success:
                    status.update(label=f"{safe_name} processed!", state="complete")
                    st.session_state[processed_key] = True
                    st.session_state["vectorstore_ready"] = True
                else:
                    status.update(label=f"{safe_name} failed", state="error")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Uploaded Documents")

    existing_files = _get_uploaded_files()
    if existing_files:
        for filename in existing_files:
            st.sidebar.text(f"• {filename}")
    else:
        st.sidebar.info("No documents uploaded yet.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear All Documents", type="secondary", use_container_width=True):
        try:
            # Clear RAG chain from session state and trigger garbage collection to release database file locks
            if "rag_chain" in st.session_state:
                chain = st.session_state["rag_chain"]
                if chain is not None:
                    try:
                        retriever = getattr(chain, "retriever", None)
                        if retriever is not None:
                            vectorstore = getattr(retriever, "vectorstore", None)
                            if vectorstore is not None and hasattr(vectorstore, "_client"):
                                if hasattr(vectorstore._client, "close"):
                                    vectorstore._client.close()
                                    logger.info("Closed active vectorstore client connection.")
                    except Exception as e:
                        logger.warning("Failed to close active vectorstore client: %s", e)
                del st.session_state["rag_chain"]

            # Explicitly stop all cached ChromaDB system components to release file locks on Windows
            try:
                from chromadb.api.shared_system_client import SharedSystemClient
                for key in list(SharedSystemClient._identifier_to_system.keys()):
                    try:
                        system = SharedSystemClient._identifier_to_system.get(key)
                        if system is not None:
                            logger.info("Force-stopping cached ChromaDB system for %s", key)
                            system.stop()
                    except Exception as sys_exc:
                        logger.warning("Failed to stop system for %s: %s", key, sys_exc)
                SharedSystemClient.clear_system_cache()
                logger.info("Cleared chromadb SharedSystemClient cache.")
            except Exception as cache_exc:
                logger.warning("Failed to stop systems or clear cache: %s", cache_exc)

            import gc
            gc.collect()


            upload_path = Path(UPLOAD_DIR)
            if upload_path.exists():
                for f in upload_path.iterdir():
                    if f.is_file():
                        f.unlink()


            store = VectorStore()
            store.delete_collection()


            keys_to_clear = [
                k for k in st.session_state.keys()
                if k.startswith("processed_")
            ]
            for k in keys_to_clear:
                del st.session_state[k]

            st.session_state["vectorstore_ready"] = False
            st.session_state["chat_history"] = []

            st.sidebar.success("All documents and data cleared.")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Failed to clear documents: {exc}")
