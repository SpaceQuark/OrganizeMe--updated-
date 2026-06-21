"""End-to-end index/retrieve tests against real FAISS, with deterministic
embeddings so no Ollama/torch backend is required."""

import numpy as np

from organizeme import search, embeddings

_VOCAB = ["cat", "dog", "fish"]


def _vec(text):
    return [float(text.lower().split().count(word)) for word in _VOCAB]


def _fake_embed_texts(texts):
    """Bag-of-vocab embedding: normalized counts of a tiny fixed vocabulary."""
    vectors, kept = [], []
    for i, text in enumerate(texts):
        counts = _vec(text)
        if sum(counts) == 0:
            continue
        vectors.append(counts)
        kept.append(i)
    if not vectors:
        return np.zeros((0, 0), dtype="float32"), []
    arr = np.asarray(vectors, dtype="float32")
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    return arr, kept


def test_build_index_chunks_and_retrieves_nearest(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "embed_texts", _fake_embed_texts)
    (tmp_path / "a.txt").write_text("cat cat cat")
    (tmp_path / "b.txt").write_text("dog dog dog")

    indexed = search.build_index(str(tmp_path))
    assert indexed == 2  # one chunk per short file

    hits = search.retrieve_text("cat", search.default_index_dir(str(tmp_path)), k=5)
    assert hits[0]["path"].endswith("a.txt")
    assert hits[0]["text"] == "cat cat cat"
    assert hits[0]["start"] == 0


def test_semantic_search_collapses_chunks_per_file(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "embed_texts", _fake_embed_texts)
    # Long enough to split into many chunks, all matching "cat".
    (tmp_path / "a.txt").write_text("cat " * 400)
    (tmp_path / "b.txt").write_text("dog dog")

    search.build_index(str(tmp_path))
    results = search.semantic_search("cat", search.default_index_dir(str(tmp_path)), k=5)

    paths = [p for _, p in results]
    assert any(p.endswith("a.txt") for p in paths)
    # Despite many matching chunks, a.txt collapses to a single result.
    assert sum(1 for p in paths if p.endswith("a.txt")) == 1
