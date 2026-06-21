"""High-level orchestration shared by the CLI and the MCP server.

Functions here return structured results and never write to disk unless ``apply=True``
is passed, so the same code path powers dry-runs (CLI preview, MCP agent calls) and
real organization.
"""

import os

from organizeme import scanning
from organizeme import organize
from organizeme.extract import read_file_data
from organizeme.metadata import generate_text_metadata, generate_image_metadata

# Lazily-initialized model singletons (Ollama clients are cheap handles, but we still
# avoid importing/constructing them until a content operation actually needs them).
_models = {"text": None, "vlm": None}


def get_models():
    """Return ``(image_inference, text_inference)``, constructing them once."""
    if _models["text"] is None or _models["vlm"] is None:
        from organizeme.inference import (
            OllamaTextInference,
            OllamaVLMInference,
            filter_specific_output,
        )
        with filter_specific_output():
            _models["vlm"] = OllamaVLMInference(model_name="llava", temperature=0.3)
            _models["text"] = OllamaTextInference(model_name="llama2", temperature=0.5)
    return _models["vlm"], _models["text"]


def analyze_files(file_paths, silent=False, log_file=None):
    """Run content analysis over files, returning metadata records.

    Each record: ``{file_path, foldername, filename, description}``.
    """
    image_inference, text_inference = get_models()
    image_files, text_files = scanning.separate_files_by_type(file_paths)

    records = []
    for path in image_files:
        foldername, filename, description = generate_image_metadata(path, image_inference, text_inference)
        records.append({"file_path": path, "foldername": foldername, "filename": filename, "description": description})

    for path in text_files:
        text = read_file_data(path)
        if not text:
            continue
        foldername, filename, description = generate_text_metadata(text, path, text_inference)
        records.append({"file_path": path, "foldername": foldername, "filename": filename, "description": description})

    return records


def _default_output(input_path):
    return os.path.join(os.path.dirname(os.path.abspath(input_path)), "organized_folder")


def plan_operations(input_path, mode, output_path=None):
    """Compute operations for ``mode`` ('content'|'date'|'type') without applying them."""
    output_path = output_path or _default_output(input_path)
    file_paths = scanning.collect_file_paths(input_path)

    if mode == "content":
        records = analyze_files(file_paths)
        operations = organize.compute_operations(records, output_path)
    elif mode == "date":
        operations = organize.process_files_by_date(file_paths, output_path)
    elif mode == "type":
        operations = organize.process_files_by_type(file_paths, output_path)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return {"mode": mode, "output_path": output_path, "operations": operations}


def proposed_tree(operations, output_path):
    """Render the proposed destination layout as a list of lines."""
    destinations = [op["destination"] for op in operations]
    tree = scanning.build_tree(destinations, output_path)
    return scanning.render_tree(tree)


def dry_run_organize(input_path, mode="type", output_path=None):
    """Plan operations and include a rendered tree. No files are touched."""
    plan = plan_operations(input_path, mode, output_path)
    plan["tree"] = proposed_tree(plan["operations"], plan["output_path"])
    return plan


def organize_files(input_path, mode="type", output_path=None, apply=False, silent=False, log_file=None):
    """Plan and (when ``apply``) execute organization for ``mode``."""
    plan = plan_operations(input_path, mode, output_path)
    if apply:
        os.makedirs(plan["output_path"], exist_ok=True)
    organize.execute_operations(plan["operations"], dry_run=not apply, silent=silent, log_file=log_file)
    plan["applied"] = apply
    return plan


# --- dedup + search (Phase 3) ---------------------------------------------- #

def detect_duplicates(input_path, threshold=None, apply=False):
    """Detect (and optionally quarantine) duplicate files. See ``dedup.detect_duplicates``."""
    from organizeme import dedup
    return dedup.detect_duplicates(input_path, threshold=threshold, apply=apply)


def build_index(input_path, index_dir=None):
    """Build the FAISS semantic index for ``input_path``. Returns files indexed."""
    from organizeme import search
    return search.build_index(input_path, index_dir)


def semantic_search(query, input_path=None, index_dir=None, k=5):
    """Search the index. Pass either an ``index_dir`` or an ``input_path`` to derive it."""
    from organizeme import search
    if index_dir is None:
        if input_path is None:
            raise ValueError("provide index_dir or input_path")
        index_dir = search.default_index_dir(input_path)
    return search.semantic_search(query, index_dir, k=k)


def answer(query, input_path=None, index_dir=None, k=5):
    """Answer a question grounded in the index (RAG). See ``rag.answer``.

    Pass either an ``index_dir`` or an ``input_path`` to derive it.
    """
    from organizeme import search, rag
    if index_dir is None:
        if input_path is None:
            raise ValueError("provide index_dir or input_path")
        index_dir = search.default_index_dir(input_path)
    return rag.answer(query, index_dir, k=k)
