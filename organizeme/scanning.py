"""File discovery, type separation and directory-tree display.

Thin wrappers around the working helpers in ``utils.organize_utils`` so the rest
of the package (and the MCP layer) has a single, stable surface to call.
"""

import os

from organizeme import native
from utils.organize_utils import find_file_paths

# Extension -> coarse category. Kept in sync with utils.organize_utils.separate_by_type.
FILE_TYPES = {
    "text": [".txt", ".md", ".doc", ".docx", ".pdf", ".rtf"],
    "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"],
    "audio": [".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"],
    "video": [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv"],
    "archive": [".zip", ".tar", ".gz", ".rar", ".7z"],
    "spreadsheet": [".xls", ".xlsx", ".csv"],
}

EXT_TO_TYPE = {ext: cat for cat, exts in FILE_TYPES.items() for ext in exts}

# Directories we never want to walk into or organize.
DEFAULT_EXCLUDE_DIRS = (".git", "__pycache__", ".organizeme", "_duplicates", "organized_folder")


def category_for(path):
    """Return the coarse category ('text', 'image', ...) for a file path."""
    return EXT_TO_TYPE.get(os.path.splitext(path)[1].lower(), "other")


def collect_file_paths(base_path, exclude_dirs=DEFAULT_EXCLUDE_DIRS):
    """Return a list of every file under ``base_path`` (recursively).

    A single file path is returned as a one-element list so callers can treat
    files and directories uniformly.
    """
    if os.path.isfile(base_path):
        return [base_path]
    if native.HAVE_NATIVE:
        return native.scan_paths(base_path, exclude_dirs)
    return list(find_file_paths(base_path, recursive=True, exclude_dirs=exclude_dirs))


def separate_files_by_type(file_paths):
    """Split a list of paths into ``(image_files, text_files)``.

    Mirrors the contract main.py's content mode expects. Image files go to the
    VLM; text-bearing files (documents and spreadsheets) go to the text model.
    """
    image_files, text_files = [], []
    for path in file_paths:
        category = category_for(path)
        if category == "image":
            image_files.append(path)
        elif category in ("text", "spreadsheet"):
            text_files.append(path)
    return image_files, text_files


def build_tree(file_paths, base_path):
    """Build a nested-dict tree of ``file_paths`` relative to ``base_path``."""
    tree = {}
    for path in file_paths:
        rel = os.path.relpath(path, base_path)
        node = tree
        for part in rel.split(os.sep):
            node = node.setdefault(part, {})
    return tree


def render_tree(tree, prefix=""):
    """Render a nested-dict tree as box-drawing lines (returns a list of str)."""
    lines = []
    keys = list(tree)
    pointers = ["├── "] * (len(keys) - 1) + ["└── "] if keys else []
    for pointer, key in zip(pointers, keys):
        lines.append(prefix + pointer + key)
        if tree[key]:
            extension = "│   " if pointer.startswith("├") else "    "
            lines.extend(render_tree(tree[key], prefix + extension))
    return lines


def display_directory_tree(base_path):
    """Print the directory tree rooted at ``base_path``."""
    file_paths = collect_file_paths(base_path)
    tree = build_tree(file_paths, base_path)
    print(os.path.abspath(base_path))
    for line in render_tree(tree):
        print(line)
