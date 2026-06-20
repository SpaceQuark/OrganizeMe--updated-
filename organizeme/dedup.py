"""Two-stage duplicate detection.

  Stage 1 (exact):   SHA-256 of file bytes -> identical files.
  Stage 2 (near):    per-modality embedding cosine similarity above a threshold
                     (text via nomic, images via OpenCLIP), linked into clusters.

Both stages feed a union-find so an exact pair and a near pair can join the same
cluster. Each cluster keeps one canonical file (oldest, then shortest path); the rest
become quarantine operations (move into ``_duplicates/``).
"""

import os
import hashlib

from organizeme import scanning, embeddings, organize, native
from organizeme.extract import read_file_data
from organizeme.index import VectorIndex

DEFAULT_TEXT_THRESHOLD = 0.95
DEFAULT_IMAGE_THRESHOLD = 0.92
_NEIGHBORS = 10


class _UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self):
        clusters = {}
        for item in self.parent:
            clusters.setdefault(self.find(item), []).append(item)
        return list(clusters.values())


def _sha256(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _near_edges(paths, vectors, threshold):
    """Return list of (i_path, j_path, similarity) for pairs above ``threshold``."""
    if len(paths) < 2 or vectors.shape[0] < 2:
        return []
    idx = VectorIndex(vectors.shape[1])
    idx.add(vectors, [{"path": p} for p in paths])
    edges = []
    for i, path in enumerate(paths):
        for score, meta in idx.search(vectors[i], k=min(_NEIGHBORS, len(paths))):
            other = meta["path"]
            if other != path and score >= threshold:
                edges.append((path, other, score))
    return edges


def _pick_canonical(paths):
    """Keep the oldest file; tie-break on shortest path string."""
    def key(p):
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            mtime = float("inf")
        return (mtime, len(p), p)

    return min(paths, key=key)


def find_clusters(input_path, text_threshold=None, image_threshold=None):
    """Detect duplicate clusters under ``input_path``. Returns a list of cluster dicts."""
    text_threshold = DEFAULT_TEXT_THRESHOLD if text_threshold is None else text_threshold
    image_threshold = DEFAULT_IMAGE_THRESHOLD if image_threshold is None else image_threshold

    file_paths = scanning.collect_file_paths(input_path)
    if not file_paths:
        return []

    uf = _UnionFind(file_paths)
    edge_sims = {}  # frozenset({a,b}) -> similarity

    # Stage 1: exact byte-for-byte duplicates. The digest algorithm is internal
    # to exact dedup, so the native (BLAKE3) and fallback (SHA-256) paths only
    # need to be internally consistent within a single run.
    if native.HAVE_NATIVE:
        hashes = native.hash_files(file_paths)
    else:
        hashes = {path: _sha256(path) for path in file_paths}
    by_hash = {}
    for path, digest in hashes.items():
        if digest is not None:
            by_hash.setdefault(digest, []).append(path)
    for group in by_hash.values():
        for other in group[1:]:
            uf.union(group[0], other)
            edge_sims[frozenset({group[0], other})] = 1.0

    # Stage 2: near-duplicates per modality.
    image_files, text_files = scanning.separate_files_by_type(file_paths)

    # Near-dup stages are best-effort: if an embedding backend is unavailable
    # (Ollama down, torch not installed), we still return exact-hash duplicates.
    if text_files:
        texts, sources = [], []
        for path in text_files:
            text = read_file_data(path)
            if text:
                texts.append(text)
                sources.append(path)
        if sources:
            try:
                vectors, kept = embeddings.embed_texts(texts)
                kept_paths = [sources[i] for i in kept]
                for a, b, sim in _near_edges(kept_paths, vectors, text_threshold):
                    uf.union(a, b)
                    edge_sims[frozenset({a, b})] = max(edge_sims.get(frozenset({a, b}), 0.0), sim)
            except Exception as exc:
                print(f"Text near-duplicate detection skipped: {exc}")

    if image_files:
        try:
            vectors, kept = embeddings.embed_images(image_files)
            kept_paths = [image_files[i] for i in kept]
            for a, b, sim in _near_edges(kept_paths, vectors, image_threshold):
                uf.union(a, b)
                edge_sims[frozenset({a, b})] = max(edge_sims.get(frozenset({a, b}), 0.0), sim)
        except Exception as exc:
            print(f"Image near-duplicate detection skipped: {exc}")

    # Assemble clusters of size >= 2.
    clusters = []
    for members in uf.groups():
        if len(members) < 2:
            continue
        keep = _pick_canonical(members)
        duplicates = [m for m in members if m != keep]
        member_hashes = {hashes[m] for m in members if hashes[m] is not None}
        method = "exact" if len(member_hashes) == 1 else "near"
        sims = [edge_sims[k] for k in edge_sims if k <= set(members)]
        clusters.append({
            "method": method,
            "similarity": min(sims) if sims else 1.0,
            "keep": keep,
            "duplicates": duplicates,
        })
    return clusters


def detect_duplicates(input_path, threshold=None, apply=False, quarantine_dir=None):
    """Detect duplicates and, when ``apply``, quarantine the non-canonical files.

    ``threshold`` (if given) overrides both modality thresholds. Returns a report:
    ``{clusters, duplicate_count, quarantine_dir, applied}``.
    """
    clusters = find_clusters(input_path, text_threshold=threshold, image_threshold=threshold)
    quarantine_dir = quarantine_dir or os.path.join(input_path, "_duplicates")
    operations = organize.quarantine_operations(clusters, quarantine_dir)

    if apply and operations:
        os.makedirs(quarantine_dir, exist_ok=True)
        organize.execute_operations(operations, dry_run=False)

    return {
        "clusters": clusters,
        "duplicate_count": sum(len(c["duplicates"]) for c in clusters),
        "quarantine_dir": quarantine_dir,
        "operations": operations,
        "applied": bool(apply and operations),
    }
