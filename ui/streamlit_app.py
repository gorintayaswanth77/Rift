"""
Rift — Streamlit application entry point.

Ties together the sidebar (document management) and chat (Q&A) components
into a single-page Streamlit application.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that config and app
# packages can be imported when Streamlit is launched from any directory.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st  # noqa: E402

from ui.components.sidebar import render_sidebar  # noqa: E402
from ui.components.chat import render_chat  # noqa: E402


st.set_page_config(
    page_title="Rift AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    /* Global */
    .stApp {
        background-color: #0a0a0a;
        color: #ffffff;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(90deg, #00d4d4 0%, #008b8b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #a0a0a0;
        font-size: 1.05rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111111;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #111111 !important;
        border: 1px solid #222222 !important;
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }

    /* Expander (sources) */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        color: #00d4d4 !important;
    }

    /* Divider */
    hr {
        border-color: #222222;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #00d4d4 !important;
        color: #0a0a0a !important;
        border: 1px solid #00d4d4 !important;
        font-weight: 600 !important;
    }
    div.stButton > button:hover {
        background-color: #008b8b !important;
        color: #ffffff !important;
        border: 1px solid #008b8b !important;
    }

    /* Inputs & Textareas */
    .stTextInput input, [data-testid="stChatInput"] textarea {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #222222 !important;
    }
    .stTextInput input:focus, [data-testid="stChatInput"] textarea:focus {
        border-color: #00d4d4 !important;
        box-shadow: 0 0 0 1px #00d4d4 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "vectorstore_ready" not in st.session_state:

    from config.settings import CHROMA_PERSIST_DIR  # noqa: E402

    chroma_path = Path(CHROMA_PERSIST_DIR)
    st.session_state["vectorstore_ready"] = (
        chroma_path.exists() and any(chroma_path.iterdir())
    )


render_sidebar()


st.markdown(
    '<div class="main-header">'
    "<h1>Rift AI</h1>"
    "<p>Upload documents and ask questions — answers are generated with source citations.</p>"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

render_chat()
