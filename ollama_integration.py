import requests
import json
from typing import List, Optional
import base64
from pathlib import Path

class OllamaTextInference:
    def __init__(
        self,
        model_name: str = "llama2",
        temperature: float = 0.5,
        max_tokens: int = 3000,
        top_k: int = 3,
        top_p: float = 0.3,
        stop_words: Optional[List[str]] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.stop_words = stop_words or []
        self.base_url = "http://localhost:11434/api"

    def generate(self, prompt: str) -> str:
        """Generate text based on the given prompt."""
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                    "stop": self.stop_words
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error in text generation: {e}")
            return ""

class OllamaVLMInference:
    def __init__(
        self,
        model_name: str = "llava",
        temperature: float = 0.3,
        max_tokens: int = 3000,
        top_k: int = 3,
        top_p: float = 0.2,
        stop_words: Optional[List[str]] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.stop_words = stop_words or []
        self.base_url = "http://localhost:11434/api"

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate(self, image_path: str, prompt: str) -> str:
        """Generate text based on the given image and prompt."""
        try:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image_base64 = self._encode_image(image_path)
            
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [image_base64],
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                    "stop": self.stop_words
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error in image generation: {e}")
            return "" 