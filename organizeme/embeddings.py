"""Hybrid, fully-local embeddings.

  * text  -> Ollama ``nomic-embed-text`` (via ``OllamaEmbeddingInference``)
  * image -> OpenCLIP ``ViT-B-32`` (lazily loaded; torch only imported on first use)

All vectors are L2-normalized so a FAISS inner-product index yields cosine similarity.
Text and image embeddings live in different spaces, so dedup/search treat the two
modalities separately.
"""

import numpy as np

from organizeme.inference import OllamaEmbeddingInference

# OpenCLIP ViT-B-32 produces 512-d image embeddings.
_CLIP_MODEL_NAME = "ViT-B-32"
_CLIP_PRETRAINED = "laion2b_s34b_b79k"

_clip = {"model": None, "preprocess": None, "tokenizer": None, "device": None}
_text_embedder = None


def _normalize(vectors):
    """L2-normalize rows of a 2-D float32 array (safe against zero vectors)."""
    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# --- text ------------------------------------------------------------------

def _get_text_embedder():
    global _text_embedder
    if _text_embedder is None:
        _text_embedder = OllamaEmbeddingInference()
    return _text_embedder


def embed_texts(texts):
    """Embed a list of strings. Returns ``(vectors, kept_indices)``.

    ``kept_indices`` reports which inputs embedded successfully (Ollama can fail
    on a given string), so callers can map rows back to their source files.
    """
    embedder = _get_text_embedder()
    vectors, kept = [], []
    for i, text in enumerate(texts):
        vec = embedder.embed(text)
        if vec is not None:
            vectors.append(vec)
            kept.append(i)
    if not vectors:
        return np.zeros((0, 0), dtype="float32"), []
    return _normalize(vectors), kept


# --- image -----------------------------------------------------------------

def _get_clip():
    if _clip["model"] is None:
        import torch
        import open_clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            _CLIP_MODEL_NAME, pretrained=_CLIP_PRETRAINED
        )
        model.eval().to(device)
        _clip.update(
            model=model,
            preprocess=preprocess,
            tokenizer=open_clip.get_tokenizer(_CLIP_MODEL_NAME),
            device=device,
        )
    return _clip


def embed_images(image_paths):
    """Embed a list of image paths. Returns ``(vectors, kept_indices)``."""
    import torch
    from PIL import Image

    clip = _get_clip()
    model, preprocess, device = clip["model"], clip["preprocess"], clip["device"]

    tensors, kept = [], []
    for i, path in enumerate(image_paths):
        try:
            image = Image.open(path).convert("RGB")
            tensors.append(preprocess(image))
            kept.append(i)
        except Exception as exc:
            print(f"Error reading image {path}: {exc}")
    if not tensors:
        return np.zeros((0, 0), dtype="float32"), []

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        features = model.encode_image(batch).cpu().numpy()
    return _normalize(features), kept


def embed_clip_text(text):
    """Embed ``text`` with CLIP's *text* encoder (same space as image embeddings).

    Enables cross-modal search: a text query can be matched against image vectors.
    Returns a 1-D normalized vector.
    """
    import torch

    clip = _get_clip()
    model, tokenizer, device = clip["model"], clip["tokenizer"], clip["device"]
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens).cpu().numpy()
    return _normalize(features)[0]


def embed_query(text):
    """Embed a single query string into the nomic text space (1-D normalized vector)."""
    vectors, kept = embed_texts([text])
    if not kept:
        return None
    return vectors[0]
