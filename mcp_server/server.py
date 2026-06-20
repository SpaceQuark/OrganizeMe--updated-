"""MCP server exposing the OrganizeMe pipeline to desktop agents.

Runs over stdio using the official ``mcp`` SDK (FastMCP). Every tool returns
structured JSON, and any tool that writes to disk defaults to ``apply=False`` so an
agent dry-runs before committing changes.

Register with a desktop agent, e.g. Claude Desktop ``claude_desktop_config.json``:

    {
      "mcpServers": {
        "organizeme": {
          "command": "python",
          "args": ["-m", "mcp_server.server"],
          "cwd": "/path/to/OrganizeMe_updated"
        }
      }
    }
"""

import os
import sys

# Allow ``python mcp_server/server.py`` as well as ``python -m mcp_server.server``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

from organizeme import pipeline

mcp = FastMCP("organizeme")


@mcp.tool()
def dry_run_organize(path: str, mode: str = "type") -> dict:
    """Preview how files under ``path`` would be organized. Writes nothing.

    Args:
        path: Directory (or file) to analyze.
        mode: 'content' (semantic, LLM/VLM), 'date', or 'type'.
    Returns: mode, output_path, the proposed tree, and the operation list.
    """
    plan = pipeline.dry_run_organize(path, mode=mode)
    return {
        "mode": plan["mode"],
        "output_path": plan["output_path"],
        "tree": plan["tree"],
        "operations": plan["operations"],
        "operation_count": len(plan["operations"]),
    }


@mcp.tool()
def content_sort(path: str, output: str = None, apply: bool = False) -> dict:
    """Semantically sort files by content (LLM/VLM). Dry-run unless ``apply`` is true."""
    plan = pipeline.organize_files(path, mode="content", output_path=output, apply=apply)
    return {
        "output_path": plan["output_path"],
        "applied": plan["applied"],
        "operations": plan["operations"],
        "operation_count": len(plan["operations"]),
    }


@mcp.tool()
def detect_duplicates(path: str, threshold: float = None, apply: bool = False) -> dict:
    """Find exact + near-duplicate files. Quarantines them only when ``apply`` is true."""
    report = pipeline.detect_duplicates(path, threshold=threshold, apply=apply)
    return {
        "duplicate_count": report["duplicate_count"],
        "quarantine_dir": report["quarantine_dir"],
        "applied": report["applied"],
        "clusters": report["clusters"],
    }


@mcp.tool()
def build_index(path: str) -> dict:
    """Build a FAISS semantic index for ``path`` so ``semantic_search`` can run."""
    count = pipeline.build_index(path)
    return {"indexed": count, "index_dir": os.path.join(path, ".organizeme", "index")}


@mcp.tool()
def semantic_search(query: str, path: str, k: int = 5) -> dict:
    """Search a directory's index for files matching ``query`` (build the index first)."""
    results = pipeline.semantic_search(query, input_path=path, k=k)
    return {"results": [{"score": score, "path": p} for score, p in results]}


def main():
    mcp.run()


if __name__ == "__main__":
    main()
