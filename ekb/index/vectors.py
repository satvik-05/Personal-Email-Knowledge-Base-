"""Embedding + FAISS index for semantic search."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ekb.models import Chunk

_MODEL_NAME = "all-MiniLM-L6-v2"
_DIM = 384


class VectorIndex:
    def __init__(self, faiss_path: str | Path = "ekb.faiss"):
        self.faiss_path = Path(faiss_path)
        self._meta_path = self.faiss_path.with_suffix(".meta.npy")
        self._model: SentenceTransformer | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._chunk_ids: list[str] = []
        self._load()

    def _load(self) -> None:
        if self.faiss_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self.faiss_path))
            self._chunk_ids = np.load(str(self._meta_path), allow_pickle=True).tolist()
        else:
            self._index = faiss.IndexFlatIP(_DIM)
            self._chunk_ids = []

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(_MODEL_NAME)
        return self._model

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        model = self._get_model()
        texts = [c.text for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._index.add(np.array(embeddings, dtype=np.float32))
        self._chunk_ids.extend(c.chunk_id for c in chunks)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Returns [(chunk_id, score), ...] sorted by descending similarity."""
        if self._index.ntotal == 0:
            return []
        model = self._get_model()
        qvec = model.encode([query], normalize_embeddings=True)
        scores, indices = self._index.search(np.array(qvec, dtype=np.float32), min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self._chunk_ids[idx], float(score)))
        return results

    def save(self) -> None:
        faiss.write_index(self._index, str(self.faiss_path))
        np.save(str(self._meta_path), np.array(self._chunk_ids, dtype=object))

    @property
    def count(self) -> int:
        return self._index.ntotal
