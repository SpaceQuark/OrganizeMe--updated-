"""Split extracted document text into overlapping chunks for retrieval.

Embedding a whole document as one vector truncates long files and blurs distinct
sections. Chunking embeds smaller, overlapping windows so retrieval can pinpoint
the relevant passage and report its location.

Each chunk carries character offsets (``start``/``end``) into the *original* text
so search results keep provenance: ``text[start:end] == chunk["text"]``.

Sizes are in characters (cheap, dependency-free). The default ``max_chars`` stays
well under ``nomic-embed-text``'s context so chunks embed without truncation.
"""

import re

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 150

_WORD = re.compile(r"\S+")


def chunk_text(text, max_chars=DEFAULT_MAX_CHARS, overlap=DEFAULT_OVERLAP):
    """Split ``text`` into overlapping chunks on whitespace boundaries.

    Returns a list of ``{"text", "start", "end"}`` dicts. Windows never split a
    word, are at most ``max_chars`` long (except a single word longer than
    ``max_chars``, which is emitted whole), and consecutive chunks overlap by up
    to ``overlap`` characters. Empty/whitespace-only input yields ``[]``.
    """
    if not text or not text.strip():
        return []
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    spans = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    if not spans:
        return []

    chunks = []
    i = 0
    n = len(spans)
    while i < n:
        start = spans[i][0]
        j = i
        # Grow the window while it fits; always keep at least one word.
        while j + 1 < n and (spans[j + 1][1] - start) <= max_chars:
            j += 1
        end = spans[j][1]
        chunks.append({"text": text[start:end], "start": start, "end": end})

        if j + 1 >= n:
            break

        # Next window starts back far enough to overlap by ~``overlap`` chars,
        # but always advances at least one word so we can't loop forever.
        next_i = j + 1
        while next_i - 1 > i and (end - spans[next_i - 1][0]) <= overlap:
            next_i -= 1
        i = next_i

    return chunks
