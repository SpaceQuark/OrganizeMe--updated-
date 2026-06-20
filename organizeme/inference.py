"""Local inference clients (Ollama) and an output-suppression helper.

Re-exports the working ``OllamaTextInference`` / ``OllamaVLMInference`` clients and
adds:
  * ``OllamaEmbeddingInference`` — text embeddings via ``nomic-embed-text``.
  * ``filter_specific_output()`` — the context manager main.py expects (replaces the
    missing ``output_filter`` module); silences noisy stdout during model init.
"""

import os
import sys
import contextlib
from typing import List, Optional

import requests

from ollama_integration import OllamaTextInference, OllamaVLMInference

DEFAULT_BASE_URL = "http://localhost:11434/api"

__all__ = [
    "OllamaTextInference",
    "OllamaVLMInference",
    "OllamaEmbeddingInference",
    "filter_specific_output",
]


@contextlib.contextmanager
def filter_specific_output():
    """Suppress stdout/stderr for the duration of the block.

    Used while initializing models so library banners don't clutter the CLI.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = devnull, devnull
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


class OllamaEmbeddingInference:
    """Text embeddings from a local Ollama model (default ``nomic-embed-text``)."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout

    def embed(self, text: str) -> Optional[List[float]]:
        """Return the embedding vector for ``text`` (or ``None`` on failure)."""
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as exc:
            print(f"Error in text embedding: {exc}")
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed a list of texts sequentially (Ollama embeds one prompt per call)."""
        return [self.embed(t) for t in texts]
