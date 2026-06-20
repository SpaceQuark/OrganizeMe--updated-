"""Persistent FAISS vector store.

A thin wrapper over ``faiss.IndexFlatIP`` (exact inner-product search; with
L2-normalized vectors this is cosine similarity). Each row carries a metadata dict
(path, modality, hash, ...) persisted alongside the index so search/dedup can map
results back to files.
"""

import os
import json

import numpy as np


class VectorIndex:
    """An exact inner-product FAISS index plus a parallel metadata list."""

    INDEX_FILE = "vectors.faiss"
    META_FILE = "metadata.json"

    def __init__(self, dimension):
        import faiss

        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self.metadata = []

    # --- mutation ----------------------------------------------------------

    def add(self, vectors, metadata):
        """Add ``vectors`` (n×d) with their per-row ``metadata`` dicts."""
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.shape[0] != len(metadata):
            raise ValueError("vectors and metadata length mismatch")
        if vectors.shape[0] == 0:
            return
        self._index.add(vectors)
        self.metadata.extend(metadata)

    # --- query -------------------------------------------------------------

    def search(self, query_vector, k=5):
        """Return up to ``k`` ``(score, metadata)`` pairs for ``query_vector``."""
        if self._index.ntotal == 0:
            return []
        query = np.asarray(query_vector, dtype="float32")
        if query.ndim == 1:
            query = query[None, :]
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((float(score), self.metadata[idx]))
        return results

    def __len__(self):
        return self._index.ntotal

    # --- persistence -------------------------------------------------------

    def save(self, directory):
        import faiss

        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self._index, os.path.join(directory, self.INDEX_FILE))
        with open(os.path.join(directory, self.META_FILE), "w") as handle:
            json.dump({"dimension": self.dimension, "metadata": self.metadata}, handle, indent=2)

    @classmethod
    def load(cls, directory):
        import faiss

        meta_path = os.path.join(directory, cls.META_FILE)
        index_path = os.path.join(directory, cls.INDEX_FILE)
        if not (os.path.exists(meta_path) and os.path.exists(index_path)):
            raise FileNotFoundError(f"No index found in {directory!r}")
        with open(meta_path) as handle:
            payload = json.load(handle)
        instance = cls(payload["dimension"])
        instance._index = faiss.read_index(index_path)
        instance.metadata = payload["metadata"]
        return instance
