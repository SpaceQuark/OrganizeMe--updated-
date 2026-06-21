"""Retrieval-augmented generation: answer questions grounded in indexed files.

``answer`` retrieves the most relevant text chunks (``search.retrieve_text``),
packs them into a source-cited prompt, and asks the local Ollama text model to
synthesize an answer that cites its sources by number. The returned ``sources``
list maps those numbers back to files and character offsets for provenance.

Generation is local and best-effort: if no chunks are retrieved we return early
without calling the model.
"""

import os

from organizeme import search

DEFAULT_K = 5
# Cap total context so a few large chunks can't blow past the model's window.
DEFAULT_MAX_CONTEXT_CHARS = 6000
_SNIPPET_CHARS = 240


def _snippet(text):
    text = " ".join((text or "").split())
    return text if len(text) <= _SNIPPET_CHARS else text[:_SNIPPET_CHARS].rstrip() + "…"


def _trim_to_budget(hits, max_context_chars):
    """Keep hits (in order) until the character budget is exhausted."""
    trimmed, used = [], 0
    for hit in hits:
        if used >= max_context_chars:
            break
        text = hit["text"]
        remaining = max_context_chars - used
        if len(text) > remaining:
            text = text[:remaining]
        if not text:
            continue
        trimmed.append({**hit, "text": text})
        used += len(text)
    return trimmed


def build_prompt(query, hits):
    """Build a grounded, cite-your-sources prompt from retrieved ``hits``."""
    blocks = []
    for n, hit in enumerate(hits, 1):
        label = os.path.basename(hit["path"]) or hit["path"]
        blocks.append(f"[{n}] {label}\n{hit['text']}")
    sources = "\n\n".join(blocks)
    return (
        "You are a precise assistant. Answer the question using ONLY the sources "
        "below. Cite the sources you use inline with bracketed numbers like [1] or "
        "[2]. If the sources do not contain the answer, say you don't know.\n\n"
        f"Sources:\n{sources}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def answer(query, index_dir, k=DEFAULT_K, text_inference=None, max_context_chars=DEFAULT_MAX_CONTEXT_CHARS):
    """Answer ``query`` from the index at ``index_dir``.

    Returns ``{answer, sources, grounded}`` where ``grounded`` is ``False`` when
    nothing was retrieved (and no model call was made). ``sources`` is a list of
    ``{n, path, score, start, end, snippet}`` for citation mapping.
    """
    hits = _trim_to_budget(search.retrieve_text(query, index_dir, k), max_context_chars)
    if not hits:
        return {"answer": "", "sources": [], "grounded": False}

    if text_inference is None:
        from organizeme import pipeline

        _, text_inference = pipeline.get_models()

    prompt = build_prompt(query, hits)
    response = (text_inference.generate(prompt) or "").strip()

    sources = [
        {
            "n": n,
            "path": hit["path"],
            "score": hit["score"],
            "start": hit["start"],
            "end": hit["end"],
            "snippet": _snippet(hit["text"]),
        }
        for n, hit in enumerate(hits, 1)
    ]
    return {"answer": response, "sources": sources, "grounded": True}
