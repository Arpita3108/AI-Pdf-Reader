"""Sentence-aware text chunking with configurable size and overlap."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .pdf_processor import Page

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    doc: str
    page: int
    chunk_id: int  # index within the document


def _units(text: str, chunk_size: int) -> list[str]:
    """Break text into sentences; hard-split any sentence longer than chunk_size."""
    out: list[str] = []
    for para in text.split("\n\n"):
        for sent in _SENT_SPLIT.split(para.strip()):
            sent = sent.strip()
            if not sent:
                continue
            while len(sent) > chunk_size:
                out.append(sent[:chunk_size])
                sent = sent[chunk_size:]
            out.append(sent)
    return out


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Greedily pack sentences into chunks of <= chunk_size characters.

    Each new chunk starts with the trailing sentences of the previous one
    (up to `overlap` characters) so facts split across a boundary stay retrievable.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in _units(text, chunk_size):
        if cur and cur_len + len(s) + 1 > chunk_size:
            chunks.append(" ".join(cur))
            tail: list[str] = []
            tail_len = 0
            for t in reversed(cur):
                if tail_len + len(t) > overlap:
                    break
                tail.insert(0, t)
                tail_len += len(t) + 1
            cur, cur_len = tail, tail_len
            if cur_len + len(s) + 1 > chunk_size:  # overlap would overflow
                cur, cur_len = [], 0
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_pages(pages: list[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk page by page so every chunk keeps an exact document + page reference."""
    chunks: list[Chunk] = []
    for p in pages:
        for piece in split_text(p.text, chunk_size, overlap):
            chunks.append(Chunk(text=piece, doc=p.doc, page=p.page, chunk_id=len(chunks)))
    return chunks
