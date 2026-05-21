"""
Centralized configuration for Rift.

All application settings are defined here with sensible defaults.
Every setting can be overridden via environment variables or a .env file
placed in the project root.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(_PROJECT_ROOT / "data" / "vectorstore"),
)
UPLOAD_DIR: str = os.getenv(
    "UPLOAD_DIR",
    str(_PROJECT_ROOT / "data" / "uploads"),
)


CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))


EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "3"))


os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
