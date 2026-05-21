"""
Chat interface component for Rift.

Renders the conversational Q&A interface with message history, question
input, and source-citation display.
"""

from __future__ import annotations

import logging

import streamlit as st

from app.embeddings.embedder import EmbeddingModel
from app.vectorstore.chroma_store import VectorStore
from app.retrieval.retriever import DocumentRetriever
from app.llm.ollama_llm import LLMModel
from app.chain.rag_chain import RAGChain

logger = logging.getLogger(__name__)


def _initialise_chain() -> None:
    """Build the RAG chain and cache it in session state."""
    if "rag_chain" not in st.session_state:
        try:
            embedder = EmbeddingModel()
            embeddings = embedder.get_embeddings()

            store = VectorStore()
            vectorstore = store.load(embeddings)

            retriever_factory = DocumentRetriever()
            retriever = retriever_factory.get_retriever(vectorstore)

            llm_factory = LLMModel()
            llm = llm_factory.get_llm()

            chain = RAGChain.build(retriever, llm)
            st.session_state["rag_chain"] = chain
        except Exception as exc:
            logger.error("Failed to initialise RAG chain: %s", exc)
            st.session_state["rag_chain"] = None


def render_chat() -> None:
    """Render the chat interface with message history and input."""

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


    for message in st.session_state["chat_history"]:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.markdown(content)


            if role == "assistant" and message.get("sources"):
                with st.expander("Sources", expanded=False):
                    st.markdown(message["sources"])


    if not st.session_state.get("vectorstore_ready", False):
        st.chat_input(
            "Upload documents first...",
            disabled=True,
            key="chat_input_disabled",
        )
        return


    _initialise_chain()

    if st.session_state.get("rag_chain") is None:
        st.error(
            "Could not initialise the RAG chain. "
            "Please ensure Ollama is running (`ollama serve`) and the "
            "Mistral model is available (`ollama pull mistral`)."
        )
        return

    question = st.chat_input(
        "Ask a question about your documents...",
        key="chat_input",
    )

    if question:

        st.session_state["chat_history"].append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)


        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = st.session_state["rag_chain"]
                result = RAGChain.ask(chain, question)

            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)

            if sources:
                with st.expander("Sources", expanded=True):
                    st.markdown(sources)


        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
