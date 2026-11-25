"""Offline tests: no model downloads. Uses a hashing embedder + the extractive LLM."""
import re
import zlib
from dataclasses import replace

import pymupdf as fitz
import numpy as np
import pytest

from app.chunker import split_text
from app.config import Settings
from app.llm import ExtractiveLLM
from app.pdf_processor import clean_text, validate_pdf
from app.prompts import NOT_FOUND
from app.rag import RAGPipeline
from app.vector_store import VectorStore


class FakeEmbedder:
    """Bag-of-words hashing embedder: similarity ~ shared (long) words."""

    dim = 256

    def encode(self, texts, batch_size=32):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]{4,}", t.lower()):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


def make_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def pipeline(tmp_path):
    s = replace(Settings(), data_dir=tmp_path, llm_provider="extractive", min_similarity=0.2)
    emb = FakeEmbedder()
    return RAGPipeline(s, embedder=emb, store=VectorStore(emb.dim, tmp_path), llm=ExtractiveLLM())


PDF = make_pdf([
    "Introduction. This report studies glacier melting in the Himalayan region.",
    "Results. Paris is the capital of France and hosts the annual climate summit.",
])


def test_chunker_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = split_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 3 and all(len(c) <= 200 for c in chunks)
    assert chunks[0].split()[-3:] == chunks[1].split()[: 3] or any(
        s in chunks[1] for s in chunks[0].split(". ")[-1:]
    )


def test_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_text("x", 100, 100)


def test_clean_text_joins_hyphenation():
    assert clean_text("exam-\nple line\nbreak") == "example line break"


def test_validation():
    with pytest.raises(ValueError):
        validate_pdf(b"%PDF-1.4", "notes.txt", 10)
    with pytest.raises(ValueError):
        validate_pdf(b"not a pdf", "a.pdf", 10)
    validate_pdf(PDF, "a.pdf", 10)


def test_ingest_and_cited_answer(pipeline):
    info = pipeline.ingest(PDF, "report.pdf")
    assert info["pages"] == 2 and info["chunks"] >= 2
    res = pipeline.ask("What is the capital of France?")
    assert res["grounded"]
    top = res["sources"][0]
    assert top["document"] == "report.pdf" and top["page"] == 2
    assert "Paris" in res["answer"]


def test_unrelated_question_is_not_answered(pipeline):
    pipeline.ingest(PDF, "report.pdf")
    res = pipeline.ask("Explain quantum chromodynamics gluons")
    assert res["answer"] == NOT_FOUND and res["sources"] == []


def test_multi_doc_filter_and_delete(pipeline):
    pipeline.ingest(PDF, "a.pdf")
    pipeline.ingest(make_pdf(["Bananas are rich in potassium and grow in tropical climates."]), "b.pdf")
    res = pipeline.ask("Where do bananas grow?", documents=["b.pdf"])
    assert {s["document"] for s in res["sources"]} == {"b.pdf"}
    assert pipeline.delete("b.pdf") and not pipeline.delete("b.pdf")
    assert pipeline.ask("Where do bananas grow?")["sources"] == []
    assert pipeline.ask("capital of France")["sources"][0]["document"] == "a.pdf"


def test_persistence_roundtrip(pipeline, tmp_path):
    pipeline.ingest(PDF, "report.pdf")
    reloaded = VectorStore.load(FakeEmbedder.dim, tmp_path)
    assert "report.pdf" in reloaded.docs and reloaded.index.ntotal == pipeline.store.index.ntotal
