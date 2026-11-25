"""PDF validation, text extraction and cleaning (PyMuPDF)."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf as fitz  # PyMuPDF


@dataclass
class Page:
    doc: str
    page: int  # 1-based
    text: str


@dataclass
class DocInfo:
    name: str
    num_pages: int
    title: str = ""
    author: str = ""


def validate_pdf(data: bytes, filename: str, max_mb: int) -> None:
    """Raise ValueError if the upload is not an acceptable PDF."""
    if not filename.lower().endswith(".pdf"):
        raise ValueError(f"'{filename}' is not a .pdf file.")
    if not data:
        raise ValueError(f"'{filename}' is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise ValueError(f"'{filename}' exceeds the {max_mb} MB upload limit.")
    if b"%PDF-" not in data[:1024]:
        raise ValueError(f"'{filename}' does not look like a valid PDF.")


def clean_text(text: str) -> str:
    """Normalise whitespace, repair hyphenated line breaks, keep paragraph breaks."""
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join "exam-\nple"
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single newline -> space
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(data: bytes, filename: str) -> tuple[list[Page], DocInfo]:
    """Extract cleaned text per page, preserving document name and page number."""
    try:
        pdf = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # corrupt file
        raise ValueError(f"Could not open '{filename}': {exc}") from exc

    with pdf:
        if pdf.needs_pass:
            raise ValueError(f"'{filename}' is password-protected.")
        meta = pdf.metadata or {}
        pages: list[Page] = []
        for i, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append(Page(doc=filename, page=i, text=text))
        info = DocInfo(
            name=filename,
            num_pages=pdf.page_count,
            title=(meta.get("title") or "").strip(),
            author=(meta.get("author") or "").strip(),
        )
    return pages, info
