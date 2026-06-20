"""OrganizeMe — local, privacy-preserving multimodal AI file organizer.

A clean package that powers both the CLI (``main.py``) and the MCP server
(``mcp_server/server.py``). Everything runs offline: LLM/VLM inference via a
local Ollama server, embeddings via Ollama + OpenCLIP, similarity search via FAISS.
"""

__version__ = "0.1.0"
