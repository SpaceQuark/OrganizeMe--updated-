"""Build a FAISS semantic index over a directory and search it.

Two indexes live side by side under ``<index_dir>/``:
  * ``text/``  — nomic-embed-text vectors of extracted document text.
  * ``image/`` — OpenCLIP image vectors.

A text query searches both: the nomic space for documents, and (cross-modally) the
CLIP space for images, then results are merged by score.
"""

import os

from organizeme import scanning, embeddings
from organizeme.extract import read_file_data
from organizeme.index import VectorIndex


def default_index_dir(base_path):
    return os.path.join(base_path, ".organizeme", "index")


def build_index(input_path, index_dir=None):
    """Embed every file under ``input_path`` and persist text + image indexes.

    Returns the total number of files indexed.
    """
    index_dir = index_dir or default_index_dir(input_path)
    file_paths = scanning.collect_file_paths(input_path)
    image_files, text_files = scanning.separate_files_by_type(file_paths)

    total = 0

    # Text documents -> nomic space.
    texts, sources = [], []
    for path in text_files:
        text = read_file_data(path)
        if text:
            texts.append(text)
            sources.append(path)
    if texts:
        try:
            vectors, kept = embeddings.embed_texts(texts)
            if kept:
                meta = [{"path": sources[i], "modality": "text"} for i in kept]
                idx = VectorIndex(vectors.shape[1])
                idx.add(vectors, meta)
                idx.save(os.path.join(index_dir, "text"))
                total += len(kept)
        except Exception as exc:
            print(f"Text indexing skipped: {exc}")

    # Images -> CLIP space.
    if image_files:
        try:
            vectors, kept = embeddings.embed_images(image_files)
            if kept:
                meta = [{"path": image_files[i], "modality": "image"} for i in kept]
                idx = VectorIndex(vectors.shape[1])
                idx.add(vectors, meta)
                idx.save(os.path.join(index_dir, "image"))
                total += len(kept)
        except Exception as exc:
            print(f"Image indexing skipped: {exc}")

    return total


def _search_subindex(index_dir, subdir, query_vector, k):
    path = os.path.join(index_dir, subdir)
    try:
        idx = VectorIndex.load(path)
    except FileNotFoundError:
        return []
    return [(score, meta["path"]) for score, meta in idx.search(query_vector, k)]


def semantic_search(query, index_dir, k=5):
    """Return up to ``k`` ``(score, path)`` results across text and image indexes."""
    results = []

    text_vec = embeddings.embed_query(query)
    if text_vec is not None:
        results.extend(_search_subindex(index_dir, "text", text_vec, k))

    # Cross-modal: text query against CLIP image vectors (best-effort; needs torch).
    if os.path.isdir(os.path.join(index_dir, "image")):
        try:
            clip_vec = embeddings.embed_clip_text(query)
            results.extend(_search_subindex(index_dir, "image", clip_vec, k))
        except Exception as exc:
            print(f"Image search skipped: {exc}")

    results.sort(key=lambda item: item[0], reverse=True)
    return results[:k]
