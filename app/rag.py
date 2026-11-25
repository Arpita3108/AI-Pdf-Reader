"""RAG pipeline: ingest PDFs, retrieve relevant chunks, generate a grounded, cited answer."""
from __future__ import annotations

import re
import threading
from dataclasses import asdict

from .chunker import chunk_pages
from .config import Settings, load_settings
from .embeddings import Embedder
from .llm import BaseLLM, build_llm
from .pdf_processor import extract_pages, validate_pdf
from .prompts import NOT_FOUND
from .vector_store import VectorStore


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        llm: BaseLLM | None = None,
    ):
        self.s = settings or load_settings()
        self.embedder = embedder or Embedder(self.s.embedding_model)
        self.store = store or VectorStore.load(self.embedder.dim, self.s.data_dir)
        self._llm = llm
        self._lock = threading.Lock()

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = build_llm(self.s)
        return self._llm

    # ---- ingestion ----------------------------------------------------
    def ingest(self, data: bytes, filename: str) -> dict:
        validate_pdf(data, filename, self.s.max_upload_mb)
        pages, info = extract_pages(data, filename)
        if not pages:
            raise ValueError(
                f"No extractable text in '{filename}'. It may be a scanned PDF (needs OCR)."
            )
        chunks = chunk_pages(pages, self.s.chunk_size, self.s.chunk_overlap)
        vectors = self.embedder.encode([c.text for c in chunks])
        with self._lock:
            self.store.add_document(
                {"name": filename, "pages": info.num_pages, "title": info.title, "author": info.author},
                vectors,
                [asdict(c) for c in chunks],
            )
            self.store.save()
        return {"document": filename, "pages": info.num_pages, "chunks": len(chunks)}

    def delete(self, filename: str) -> bool:
        with self._lock:
            removed = self.store.remove_document(filename)
            if removed:
                self.store.save()
        return removed

    def documents(self) -> list[dict]:
        return list(self.store.docs.values())

    # ---- question answering --------------------------------------------
    def ask(self, question: str, top_k: int | None = None, documents: list[str] | None = None) -> dict:
        question = question.strip()
        qvec = self.embedder.encode([question])[0]
        hits = self.store.search(qvec, top_k or self.s.top_k, self.s.min_similarity, documents)

        # Guardrail: nothing relevant retrieved -> never call the LLM, never guess.
        if not hits:
            return {"answer": NOT_FOUND, "sources": [], "grounded": False}

        answer = self.llm.answer(question, hits)
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        sources = [
            {
                "id": i,
                "document": h["doc"],
                "page": h["page"],
                "score": round(h["score"], 4),
                "snippet": h["text"][:400],
                "cited": i in cited,
            }
            for i, h in enumerate(hits, start=1)
        ]
        return {"answer": answer, "sources": sources, "grounded": NOT_FOUND not in answer}
