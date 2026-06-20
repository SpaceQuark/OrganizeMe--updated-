# OrganizeMe

Local, privacy-preserving, **multimodal AI file organizer**. It sorts files by content
(using local LLMs/VLMs via [Ollama](https://ollama.com)), by date, or by type — and adds
embedding-based **deduplication** and **semantic search** over a FAISS vector store. Everything
runs **offline**: no files or content ever leave your machine.

## Features

- **Content sorting** — a local text model (`llama2`) and vision model (`llava`) read each
  file and propose a semantic folder/filename.
- **Date / type sorting** — fast, deterministic grouping.
- **Deduplication** — exact (SHA-256) + near-duplicate (embedding cosine similarity) detection.
  Duplicates are **quarantined** into `_duplicates/` (nothing is deleted).
- **Semantic search** — FAISS index over text (`nomic-embed-text`) and images (OpenCLIP).
  Text queries match documents *and*, cross-modally, images.
- **MCP server** — expose the pipeline to desktop agents (dry-run organize, content sort,
  duplicate detection, semantic search).
- **Dry-run by default** — every destructive action requires explicit `--apply`.

## Architecture

```
organizeme/
  scanning.py   # discover files, separate by type, render trees
  extract.py    # text/pdf/doc/spreadsheet/image-OCR/audio -> text
  inference.py  # Ollama text/VLM/embedding clients
  metadata.py   # LLM/VLM -> folder name, filename, description
  embeddings.py # nomic-embed-text (text) + OpenCLIP (image), L2-normalized
  index.py      # FAISS IndexFlatIP wrapper (persist/load)
  dedup.py      # hash + embedding clustering -> quarantine ops
  organize.py   # compute/execute operations, by-date, by-type
  search.py     # build index, semantic search
  pipeline.py   # high-level API shared by CLI + MCP
main.py         # CLI (interactive default + subcommands)
mcp_server/     # MCP stdio server
```

## Prerequisites

1. **Python 3.10+** and dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Ollama** running locally with the models pulled:
   ```bash
   ollama serve
   ollama pull llava llama2 nomic-embed-text
   ```
3. **Tesseract OCR** (for image text extraction): `brew install tesseract` (macOS) or
   `apt install tesseract-ocr` (Debian/Ubuntu).

## CLI usage

```bash
# Interactive (default)
python main.py

# Organize (dry-run, then apply)
python main.py organize ~/Downloads --mode content
python main.py organize ~/Downloads --mode content --apply

# Build a semantic index, then search it
python main.py index ~/Documents
python main.py search "quarterly revenue report" --index ~/Documents/.organizeme/index

# Detect duplicates (report), then quarantine
python main.py dedup ~/Pictures
python main.py dedup ~/Pictures --apply
```

## MCP server

Run over stdio:

```bash
python -m mcp_server.server
```

Register with a desktop agent (e.g. Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "organizeme": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/OrganizeMe_updated"
    }
  }
}
```

Exposed tools: `dry_run_organize`, `content_sort`, `detect_duplicates`, `build_index`,
`semantic_search`. All writes default to `apply=false`.

## Tests

```bash
pytest
```

Tests mock Ollama/CLIP, so they run without a model server or GPU.
