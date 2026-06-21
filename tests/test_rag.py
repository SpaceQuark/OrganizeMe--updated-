from organizeme import rag


class _FakeInference:
    """Records the prompt it was given and returns a canned response."""

    def __init__(self, response="The answer is 42 [1]."):
        self.response = response
        self.prompt = None

    def generate(self, prompt):
        self.prompt = prompt
        return self.response


def _hits():
    return [
        {"score": 0.91, "path": "/docs/budget.txt", "chunk_index": 0,
         "start": 0, "end": 20, "text": "Q3 revenue was 42M."},
        {"score": 0.80, "path": "/docs/notes.md", "chunk_index": 2,
         "start": 100, "end": 130, "text": "Costs rose in the third quarter."},
    ]


def test_answer_grounds_and_cites(monkeypatch):
    monkeypatch.setattr(rag.search, "retrieve_text", lambda q, d, k=5: _hits())
    fake = _FakeInference()

    result = rag.answer("What was Q3 revenue?", "/idx", k=5, text_inference=fake)

    assert result["grounded"] is True
    assert result["answer"] == "The answer is 42 [1]."
    # Sources are numbered 1..n and map back to retrieved files.
    assert [s["n"] for s in result["sources"]] == [1, 2]
    assert result["sources"][0]["path"] == "/docs/budget.txt"
    assert result["sources"][0]["start"] == 0
    # The prompt fed to the model contains the question and the source text.
    assert "What was Q3 revenue?" in fake.prompt
    assert "Q3 revenue was 42M." in fake.prompt
    assert "[1]" in fake.prompt and "[2]" in fake.prompt


def test_answer_returns_early_when_no_hits(monkeypatch):
    monkeypatch.setattr(rag.search, "retrieve_text", lambda q, d, k=5: [])
    fake = _FakeInference()

    result = rag.answer("anything", "/idx", text_inference=fake)

    assert result == {"answer": "", "sources": [], "grounded": False}
    assert fake.prompt is None  # model is never called when nothing is retrieved


def test_context_is_trimmed_to_budget(monkeypatch):
    # Path deliberately free of the letter 'x' so the count below is unambiguous.
    big = [{"score": 0.9, "path": "/a.md", "chunk_index": 0,
            "start": 0, "end": 5000, "text": "x" * 5000}]
    monkeypatch.setattr(rag.search, "retrieve_text", lambda q, d, k=5: big)
    fake = _FakeInference()

    rag.answer("q", "/idx", text_inference=fake, max_context_chars=100)

    # Only the budgeted number of source characters reaches the prompt.
    assert fake.prompt.count("x") == 100
