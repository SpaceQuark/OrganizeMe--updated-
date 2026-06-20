"""Content extraction: turn a file on disk into text the LLM can reason about.

Dispatches by extension to the working readers in ``utils.process_utils``.
Returns ``None`` for unsupported or unreadable files so callers can skip them.
"""

import os

from utils.process_utils import (
    process_text_file,
    process_pdf_file,
    process_doc_file,
    process_spreadsheet_file,
    process_image_file,
    process_audio_file,
)

# Extension -> reader. Image OCR and audio STT are included so any modality can
# contribute text for embeddings/dedup even when not sent to the VLM directly.
_READERS = {
    ".txt": process_text_file,
    ".md": process_text_file,
    ".rtf": process_text_file,
    ".pdf": process_pdf_file,
    ".doc": process_doc_file,
    ".docx": process_doc_file,
    ".csv": process_spreadsheet_file,
    ".xls": process_spreadsheet_file,
    ".xlsx": process_spreadsheet_file,
    ".png": process_image_file,
    ".jpg": process_image_file,
    ".jpeg": process_image_file,
    ".gif": process_image_file,
    ".bmp": process_image_file,
    ".tiff": process_image_file,
    ".webp": process_image_file,
    ".wav": process_audio_file,
    ".aac": process_audio_file,
    ".flac": process_audio_file,
}


def read_file_data(file_path):
    """Extract text content from ``file_path``, or ``None`` if unsupported."""
    reader = _READERS.get(os.path.splitext(file_path)[1].lower())
    if reader is None:
        return None
    try:
        text = reader(file_path)
    except Exception as exc:  # readers are defensive, but never let one file kill a run
        print(f"Error extracting {file_path}: {exc}")
        return None
    if text is None:
        return None
    text = text.strip()
    return text or None
