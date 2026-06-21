"""Build a FAISS semantic index over a directory and search it.

Two indexes live side by side under ``<index_dir>/``:
  * ``text/``  — nomic-embed-text vectors of *chunked* document text. Each row is
                 one chunk and carries ``{path, chunk_index, start, end, text}`` so
                 retrieval can return the exact passage and its location.
  * ``image/`` — OpenCLIP image vectors (one per image).

A text query searches both: the nomic space for document chunks, and (cross-modally)
the CLIP space for images. ``retrieve_text`` exposes raw chunk hits for RAG, while
``semantic_search`` collapses chunks to one best hit per file for plain search.
"""

import os

from organizeme import scanning, embeddings, chunking
from organizeme.extract import read_file_data
from organizeme.index import VectorIndex


def default_index_dir(base_path):
    return os.path.join(base_path, ".organizeme", "index")


def build_index(input_path, index_dir=None):
    """Embed every file under ``input_path`` and persist text + image indexes.

    Text files are chunked; each chunk becomes a row in the text index. Returns
    the total number of rows indexed (text chunks + images).
    """
    index_dir = index_dir or default_index_dir(input_path)
    file_paths = scanning.collect_file_paths(input_path)
    image_files, text_files = scanning.separate_files_by_type(file_paths)

    total = 0

    # Text documents -> chunked nomic space. Index per-file so one unreadable or
    # un-embeddable file (e.g. Ollama hiccup) doesn't sink the whole run.
    text_index = None
    for path in text_files:
        text = read_file_data(path)
        if not text:
            continue
        chunks = chunking.chunk_text(text)
        if not chunks:
            continue
        try:
            vectors, kept = embeddings.embed_texts([c["text"] for c in chunks])
        except Exception as exc:
            print(f"Text indexing skipped for {path}: {exc}")
            continue
        if not kept:
            continue
        meta = [
            {
                "path": path,
                "modality": "text",
                "chunk_index": idx,
                "start": chunks[idx]["start"],
                "end": chunks[idx]["end"],
                "text": chunks[idx]["text"],
            }
            for idx in kept
        ]
        if text_index is None:
            text_index = VectorIndex(vectors.shape[1])
        text_index.add(vectors, meta)
    if text_index is not None and len(text_index):
        text_index.save(os.path.join(index_dir, "text"))
        total += len(text_index)

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


def _load(index_dir, subdir):
    try:
        return VectorIndex.load(os.path.join(index_dir, subdir))
    except FileNotFoundError:
        return None


def retrieve_text(query, index_dir, k=5):
    """Return up to ``k`` text-chunk hits for ``query``, best score first.

    Each hit: ``{score, path, chunk_index, start, end, text}``. This is the
    retrieval primitive RAG (``rag.answer``) builds on.
    """
    query_vec = embeddings.embed_query(query)
    if query_vec is None:
        return []
    index = _load(index_dir, "text")
    if index is None:
        return []
    hits = []
    for score, meta in index.search(query_vec, k):
        hits.append(
            {
                "score": float(score),
                "path": meta["path"],
                "chunk_index": meta.get("chunk_index"),
                "start": meta.get("start"),
                "end": meta.get("end"),
                "text": meta.get("text", ""),
            }
        )
    return hits


def semantic_search(query, index_dir, k=5):
    """Return up to ``k`` ``(score, path)`` results across text and image indexes.

    Text chunks are collapsed to the single best-scoring chunk per file so a
    document with many matching chunks doesn't crowd out everything else.
    """
    best_text = {}
    # Over-fetch chunks so collapsing to unique files still yields ~k documents.
    for hit in retrieve_text(query, index_dir, k * 4):
        path = hit["path"]
        if path not in best_text or hit["score"] > best_text[path]:
            best_text[path] = hit["score"]
    results = [(score, path) for path, score in best_text.items()]

    # Cross-modal: text query against CLIP image vectors (best-effort; needs torch).
    if os.path.isdir(os.path.join(index_dir, "image")):
        try:
            clip_vec = embeddings.embed_clip_text(query)
            image_index = _load(index_dir, "image")
            if image_index is not None:
                for score, meta in image_index.search(clip_vec, k):
                    results.append((float(score), meta["path"]))
        except Exception as exc:
            print(f"Image search skipped: {exc}")

    results.sort(key=lambda item: item[0], reverse=True)
    return results[:k]
