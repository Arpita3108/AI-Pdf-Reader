"""FAISS vector store with chunk metadata and on-disk persistence."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """Exact cosine-similarity search (IndexFlatIP on normalised vectors).

    A flat index is the right choice up to ~100k chunks: it is exact, needs no
    training, and searches in milliseconds. Swap in IndexHNSWFlat for larger corpora.
    """

    def __init__(self, dim: int, data_dir: Path | None = None):
        self.dim = dim
        self.data_dir = Path(data_dir) if data_dir else None
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[dict] = []  # metadata, aligned with index rows
        self.docs: dict[str, dict] = {}  # doc name -> {pages, chunks, title, author}

    # ---- writes -------------------------------------------------------
    def add_document(self, info: dict, vectors: np.ndarray, chunks: list[dict]) -> None:
        name = info["name"]
        if name in self.docs:  # re-upload replaces the old version
            self.remove_document(name)
        if len(chunks):
            self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
            self.chunks.extend(chunks)
        self.docs[name] = {**info, "chunks": len(chunks)}

    def remove_document(self, name: str) -> bool:
        if name not in self.docs:
            return False
        keep = [i for i, c in enumerate(self.chunks) if c["doc"] != name]
        vectors = self.index.reconstruct_n(0, self.index.ntotal) if self.index.ntotal else None
        self.index = faiss.IndexFlatIP(self.dim)
        if keep and vectors is not None:
            self.index.add(np.ascontiguousarray(vectors[keep]))
        self.chunks = [self.chunks[i] for i in keep]
        del self.docs[name]
        return True

    # ---- reads --------------------------------------------------------
    def search(
        self,
        query_vec: np.ndarray,
        k: int,
        min_score: float = 0.0,
        documents: list[str] | None = None,
    ) -> list[dict]:
        """Return up to k chunks with cosine score >= min_score, best first."""
        if self.index.ntotal == 0:
            return []
        # With a document filter, scan everything then filter (cheap for a flat index).
        n = self.index.ntotal if documents else min(k, self.index.ntotal)
        scores, ids = self.index.search(query_vec.reshape(1, -1).astype("float32"), n)
        allowed = set(documents) if documents else None
        results: list[dict] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or score < min_score:
                continue
            chunk = self.chunks[idx]
            if allowed is not None and chunk["doc"] not in allowed:
                continue
            results.append({**chunk, "score": float(score)})
            if len(results) == k:
                break
        return results

    # ---- persistence --------------------------------------------------
    def save(self) -> None:
        if not self.data_dir:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.data_dir / "index.faiss"))
        payload = {"dim": self.dim, "chunks": self.chunks, "docs": self.docs}
        (self.data_dir / "meta.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, dim: int, data_dir: Path) -> "VectorStore":
        store = cls(dim, data_dir)
        idx_path, meta_path = Path(data_dir) / "index.faiss", Path(data_dir) / "meta.json"
        if idx_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("dim") == dim:  # ignore stale index built with another model
                store.index = faiss.read_index(str(idx_path))
                store.chunks = meta["chunks"]
                store.docs = meta["docs"]
        return store
