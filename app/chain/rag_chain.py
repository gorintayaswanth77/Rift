"""
RAG chain orchestration for Rift.

Builds a LangChain RetrievalQA chain that retrieves relevant document
chunks, feeds them to the LLM, and returns an answer together with
source citations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel

from app.utils.helpers import format_sources

logger = logging.getLogger(__name__)

_RAG_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, say "I don't have enough information to answer this question."
Do not make up information that is not supported by the context.
Always be concise and factual.

Context:
{context}

Question: {question}

Helpful Answer:"""

_RAG_PROMPT = PromptTemplate(
    template=_RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


class RAGChain:
    """Build and query a Retrieval-Augmented Generation chain."""

    @staticmethod
    def build(retriever: BaseRetriever, llm: BaseLanguageModel) -> RetrievalQA:
        """
        Construct a :class:`RetrievalQA` chain.

        Args:
            retriever: A LangChain retriever (e.g. from ChromaDB).
            llm: A LangChain-compatible LLM instance.

        Returns:
            A configured :class:`RetrievalQA` chain ready for queries.
        """
        logger.info("Building RAG chain.")

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": _RAG_PROMPT},
        )

        logger.info("RAG chain built successfully.")
        return chain

    @staticmethod
    def ask(chain: RetrievalQA, question: str) -> Dict[str, Any]:
        """
        Send a question through the RAG chain and return the result.

        Args:
            chain: A :class:`RetrievalQA` chain built via :meth:`build`.
            question: The user's natural-language question.

        Returns:
            A dictionary with keys:

            - ``answer`` (str): The generated answer.
            - ``sources`` (str): Formatted source citations.
            - ``source_documents`` (list): Raw LangChain Document objects.
        """
        if not question or not question.strip():
            return {
                "answer": "Please provide a valid question.",
                "sources": "",
                "source_documents": [],
            }

        logger.info("Processing question: %s", question[:80])

        try:
            result = chain.invoke({"query": question})
        except Exception as exc:
            logger.error("RAG chain failed: %s", exc)
            return {
                "answer": (
                    "An error occurred while processing your question. "
                    "Please ensure Ollama is running and try again."
                ),
                "sources": "",
                "source_documents": [],
            }

        source_docs = result.get("source_documents", [])

        return {
            "answer": result.get("result", "No answer generated."),
            "sources": format_sources(source_docs),
            "source_documents": source_docs,
        }
