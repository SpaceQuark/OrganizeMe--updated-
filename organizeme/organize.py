"""Turn analysis results into concrete file operations and execute them.

This is the module the original ``main.py`` expected as ``data_processing_common``:
``compute_operations`` / ``execute_operations`` plus the by-date and by-type sorters.
Also provides ``quarantine_operations`` used by the dedup feature.

An *operation* is a dict: ``{"source": str, "destination": str, "action": "copy"|"move"}``.
"""

import os
import shutil
from datetime import datetime

from organizeme.scanning import category_for
from organizeme.metadata import sanitize_filename


def _unique_destination(destination, used):
    """Return ``destination`` adjusted with a numeric suffix if already taken."""
    if destination not in used and not os.path.exists(destination):
        used.add(destination)
        return destination
    root, ext = os.path.splitext(destination)
    counter = 1
    candidate = f"{root}_{counter}{ext}"
    while candidate in used or os.path.exists(candidate):
        counter += 1
        candidate = f"{root}_{counter}{ext}"
    used.add(candidate)
    return candidate


def compute_operations(all_data, output_path, renamed_files=None, processed_files=None, action="copy"):
    """Build operations placing each record at ``output/foldername/filename.ext``.

    ``all_data`` items are dicts with ``file_path``, ``foldername`` and ``filename``
    (as produced by ``metadata.generate_*_metadata``). Name collisions are resolved
    with numeric suffixes via ``renamed_files``.
    """
    renamed_files = renamed_files if renamed_files is not None else set()
    processed_files = processed_files if processed_files is not None else set()

    operations = []
    for record in all_data:
        source = record["file_path"]
        if source in processed_files:
            continue
        processed_files.add(source)

        ext = os.path.splitext(source)[1]
        foldername = sanitize_filename(record.get("foldername") or "misc", 2)
        filename = sanitize_filename(record.get("filename") or os.path.splitext(os.path.basename(source))[0], 3)

        destination = os.path.join(output_path, foldername, filename + ext)
        destination = _unique_destination(destination, renamed_files)
        operations.append({"source": source, "destination": destination, "action": action})
    return operations


def process_files_by_date(file_paths, output_path, action="copy", **_ignored):
    """Operations grouping files into ``YYYY/MM`` folders by modification time."""
    used = set()
    operations = []
    for source in file_paths:
        try:
            mtime = os.path.getmtime(source)
        except OSError:
            continue
        dt = datetime.fromtimestamp(mtime)
        folder = os.path.join(output_path, f"{dt.year:04d}", f"{dt.month:02d}")
        destination = _unique_destination(os.path.join(folder, os.path.basename(source)), used)
        operations.append({"source": source, "destination": destination, "action": action})
    return operations


def process_files_by_type(file_paths, output_path, action="copy", **_ignored):
    """Operations grouping files into folders by coarse category."""
    used = set()
    operations = []
    for source in file_paths:
        folder = os.path.join(output_path, category_for(source))
        destination = _unique_destination(os.path.join(folder, os.path.basename(source)), used)
        operations.append({"source": source, "destination": destination, "action": action})
    return operations


def quarantine_operations(clusters, quarantine_dir):
    """Move every non-canonical file in each duplicate cluster into ``quarantine_dir``.

    ``clusters`` is a list of dicts: ``{"keep": path, "duplicates": [paths]}``.
    """
    used = set()
    operations = []
    for cluster in clusters:
        for source in cluster.get("duplicates", []):
            destination = _unique_destination(
                os.path.join(quarantine_dir, os.path.basename(source)), used
            )
            operations.append({"source": source, "destination": destination, "action": "move"})
    return operations


def execute_operations(operations, dry_run=True, silent=False, log_file=None):
    """Perform (or simulate, when ``dry_run``) the given operations."""

    def emit(message):
        if silent:
            if log_file:
                with open(log_file, "a") as handle:
                    handle.write(message + "\n")
        else:
            print(message)

    for op in operations:
        source, destination, act = op["source"], op["destination"], op.get("action", "copy")
        if dry_run:
            emit(f"[dry-run] {act}: {source} -> {destination}")
            continue
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        try:
            if act == "move":
                shutil.move(source, destination)
            else:
                shutil.copy2(source, destination)
            emit(f"{act}: {source} -> {destination}")
        except Exception as exc:
            emit(f"Error during {act} of {source}: {exc}")
    return operations
