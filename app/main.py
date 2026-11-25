"""FastAPI backend.  Run:  uvicorn app.main:app --reload"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .rag import RAGPipeline

app = FastAPI(title="AI PDF Research Assistant", version="1.0.0")


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    documents: list[str] | None = None  # restrict search to these file names


@app.get("/health")
def health() -> dict:
    p = get_pipeline()
    return {"status": "ok", "llm_provider": p.s.llm_provider, "documents": len(p.store.docs)}


@app.post("/upload")
def upload(files: list[UploadFile] = File(...)) -> dict:
    """Upload one or more PDFs. Each file succeeds or fails independently."""
    pipeline = get_pipeline()
    results = []
    for f in files:
        try:
            results.append({"status": "ok", **pipeline.ingest(f.file.read(), f.filename or "unnamed.pdf")})
        except ValueError as exc:
            results.append({"status": "error", "document": f.filename, "error": str(exc)})
    return {"results": results}


@app.get("/documents")
def list_documents() -> dict:
    return {"documents": get_pipeline().documents()}


@app.delete("/documents/{name}")
def delete_document(name: str) -> dict:
    if not get_pipeline().delete(name):
        raise HTTPException(404, f"Document '{name}' not found.")
    return {"deleted": name}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    pipeline = get_pipeline()
    if not pipeline.store.docs:
        raise HTTPException(400, "Upload at least one PDF before asking questions.")
    try:
        return pipeline.ask(req.question, req.top_k, req.documents)
    except Exception as exc:  # LLM backend down, missing API key, etc.
        raise HTTPException(502, f"Answer generation failed: {exc}") from exc
