"""
Utility helpers for Rift.

General-purpose functions for file handling, filename sanitisation, and
source-document formatting.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import List

from langchain_core.documents import Document


def get_file_extension(filename: str) -> str:
    """
    Return the file extension (including the leading dot) in lowercase.

    Args:
        filename: A filename or path string.

    Returns:
        The extension, e.g. ``".pdf"``.  Returns ``""`` if the file has
        no extension.

    Examples:
        >>> get_file_extension("report.PDF")
        '.pdf'
        >>> get_file_extension("archive.tar.gz")
        '.gz'
    """
    _, ext = os.path.splitext(filename)
    return ext.lower()


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Sanitise a filename for safe filesystem storage.

    - Normalises Unicode to ASCII-safe form.
    - Strips characters that are invalid on Windows / macOS / Linux.
    - Replaces whitespace sequences with underscores.
    - Truncates to *max_length* characters while preserving the extension.

    Args:
        filename: The raw filename string.
        max_length: Maximum allowed length for the resulting filename.

    Returns:
        A safe, cleaned filename string.

    Examples:
        >>> sanitize_filename("My Report (Final).pdf")
        'My_Report_Final.pdf'
    """
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.encode("ascii", "ignore").decode("ascii")

    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", filename)

    filename = re.sub(r"[\s()]+", "_", filename)

    filename = filename.strip("_.")

    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[: max_length - len(ext)] + ext

    return filename or "unnamed_document"


def format_sources(source_documents: List[Document]) -> str:
    """
    Format a list of source Documents into a human-readable citation string.

    Each unique source is listed once with its page number(s).

    Args:
        source_documents: LangChain Document objects with ``source`` and
            ``page`` metadata keys.

    Returns:
        A formatted string, e.g.::

            report.pdf — Page(s): 1, 3
            notes.txt — Page(s): 1
    """
    if not source_documents:
        return "No sources available."

    seen: dict[str, list[int]] = {}
    for doc in source_documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0)
        seen.setdefault(source, [])
        if page not in seen[source]:
            seen[source].append(page)

    lines: list[str] = []
    for source, pages in seen.items():
        page_str = ", ".join(str(p) for p in sorted(pages))
        lines.append(f"{source} — Page(s): {page_str}")

    return "\n".join(lines)
