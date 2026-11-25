"""Sentence Transformer embeddings (lazy-loaded, L2-normalised)."""
from __future__ import annotations

import numpy as np


class Embedder:
    """Wraps a Sentence Transformer model.

    Vectors are L2-normalised, so the inner product used by FAISS equals
    cosine similarity (range -1..1, higher = more semantically similar).
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vecs = self._load().encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vecs, dtype="float32")
