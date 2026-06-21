import pytest

from organizeme import chunking


def test_empty_and_whitespace_yield_no_chunks():
    assert chunking.chunk_text("") == []
    assert chunking.chunk_text("   \n\t  ") == []


def test_short_text_is_a_single_chunk():
    text = "a short document about cats and dogs"
    chunks = chunking.chunk_text(text, max_chars=1200, overlap=150)
    assert len(chunks) == 1
    assert chunks[0]["start"] == 0
    assert chunks[0]["text"] == text


def test_offsets_map_back_to_original_text():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunking.chunk_text(text, max_chars=120, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert text[chunk["start"]:chunk["end"]] == chunk["text"]


def test_chunks_respect_max_chars_and_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    max_chars, overlap = 120, 20
    chunks = chunking.chunk_text(text, max_chars=max_chars, overlap=overlap)
    for chunk in chunks:
        assert len(chunk["text"]) <= max_chars
    # consecutive chunks overlap (next starts before the previous one ends).
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt["start"] < prev["end"]
        assert nxt["start"] > prev["start"]  # always makes forward progress


def test_chunks_cover_all_words_in_order():
    text = " ".join(f"w{i}" for i in range(200))
    chunks = chunking.chunk_text(text, max_chars=60, overlap=10)
    # The union of chunk spans should cover the first and last word.
    assert chunks[0]["start"] == 0
    assert chunks[-1]["end"] == len(text)


def test_word_longer_than_max_chars_is_emitted_whole():
    long_word = "x" * 50
    chunks = chunking.chunk_text(long_word, max_chars=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0]["text"] == long_word


def test_overlap_must_be_smaller_than_max_chars():
    with pytest.raises(ValueError):
        chunking.chunk_text("some text", max_chars=100, overlap=100)
