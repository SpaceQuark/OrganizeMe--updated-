"""Generate semantic folder names, filenames and descriptions for files.

Ported from the original ``utils/prompt_utils.py`` with its bugs fixed:
  * uses ``inference.generate(prompt) -> str`` (not the nonexistent ``create_completion``);
  * ``clean_ai_output`` takes its dependencies as parameters and tokenizes *before*
    filtering (the original referenced ``words``/``unwanted_words``/``lemmatizer``
    before they existed).
Also adds image metadata (VLM caption) and ``sanitize_filename`` (the helper the old
``data_processing_common`` module was supposed to provide).
"""

import os
import re
import functools

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- word lists ------------------------------------------------------------

_FILE_WORDS = {
    "file", "files", "filename", "filenames", "document", "documents", "text", "texts",
    "content", "contents", "data", "information", "output", "input", "folder", "folders",
    "directory", "directories", "path", "paths", "name", "names", "title", "titles",
}
_GENERIC_VERBS = {
    "show", "shows", "display", "displays", "present", "presents", "depict", "depicts",
    "illustrate", "illustrates", "demonstrate", "demonstrates", "describe", "describes",
    "explain", "explains", "discuss", "discusses", "cover", "covers", "include", "includes",
    "contain", "contains", "provide", "provides", "give", "gives", "offer", "offers",
}
_GENERIC_MODIFIERS = {
    "general", "generic", "common", "basic", "simple", "standard", "typical", "usual",
    "normal", "regular", "fundamental", "essential", "important", "significant",
    "major", "main", "primary", "principal", "key", "central", "core", "vital",
}


@functools.lru_cache(maxsize=1)
def get_unwanted_words():
    """Stopwords + generic file/verb/modifier words, computed once."""
    try:
        nltk_stopwords = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        nltk_stopwords = set(stopwords.words("english"))
    return nltk_stopwords | _FILE_WORDS | _GENERIC_VERBS | _GENERIC_MODIFIERS


@functools.lru_cache(maxsize=1)
def _lemmatizer():
    return WordNetLemmatizer()


# --- cleaning / sanitizing -------------------------------------------------

def clean_ai_output(text, max_words, unwanted_words=None, lemmatizer=None):
    """Turn raw model output into an underscore-joined slug of useful words."""
    unwanted_words = unwanted_words if unwanted_words is not None else get_unwanted_words()
    lemmatizer = lemmatizer or _lemmatizer()

    text = re.sub(r"[^\w\s]", " ", text)          # drop punctuation
    text = re.sub(r"\d+", "", text)               # drop digits
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # split camelCase
    words = [w.lower() for w in text.split() if w.isalpha()]
    words = [lemmatizer.lemmatize(w) for w in words]

    filtered, seen = [], set()
    for word in words:
        if word not in unwanted_words and word not in seen:
            filtered.append(word)
            seen.add(word)
    return "_".join(filtered[:max_words])


def sanitize_filename(name, max_words=5):
    """Make ``name`` safe for the filesystem and limited to ``max_words`` words."""
    name = re.sub(r'[<>:"/\\|?*]', "", name).strip()
    name = re.sub(r"[\s]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        return "untitled"
    return "_".join(name.split("_")[:max_words])


# --- prompt helpers --------------------------------------------------------

def _generate(text_inference, prompt):
    return (text_inference.generate(prompt) or "").strip()


def summarize_text_content(text, text_inference):
    """Concise (≤150 word) summary of ``text``."""
    prompt = (
        "Please provide a concise and accurate summary of the following text, focusing "
        "on the main ideas and key details. Limit your summary to a maximum of 150 words.\n\n"
        f"Text: {text}\n\nSummary:"
    )
    return _generate(text_inference, prompt)


def _filename_from_summary(summary, text_inference):
    prompt = (
        "Based on the summary below, generate a specific, descriptive filename that "
        "captures the essence of the document. Limit it to 3 words, use nouns, avoid "
        "verbs and data-type words (text, document, pdf). Use only letters joined by "
        f"underscores.\n\nSummary: {summary}\n\nOutput only the filename.\n\nFilename:"
    )
    raw = _generate(text_inference, prompt)
    return re.sub(r"^Filename:\s*", "", raw, flags=re.IGNORECASE).strip()


def _foldername_from_summary(summary, text_inference):
    prompt = (
        "Based on the summary below, generate a general category or theme (the folder "
        "name) for this document. Limit it to 2 words, use nouns, avoid verbs and generic "
        f"terms like 'untitled'.\n\nSummary: {summary}\n\nOutput only the category.\n\nCategory:"
    )
    raw = _generate(text_inference, prompt)
    return re.sub(r"^Category:\s*", "", raw, flags=re.IGNORECASE).strip()


def generate_text_metadata(input_text, file_path, text_inference):
    """Return ``(foldername, filename, description)`` for a text-bearing file."""
    unwanted = get_unwanted_words()
    lemmatizer = _lemmatizer()

    description = summarize_text_content(input_text, text_inference)

    filename = clean_ai_output(_filename_from_summary(description, text_inference), 3, unwanted, lemmatizer)
    if not filename or filename.lower() == "untitled":
        filename = clean_ai_output(description, 3, unwanted, lemmatizer)
    if not filename:
        filename = "document_" + os.path.splitext(os.path.basename(file_path))[0]

    foldername = clean_ai_output(_foldername_from_summary(description, text_inference), 2, unwanted, lemmatizer)
    if not foldername or foldername.lower() == "untitled":
        foldername = clean_ai_output(description, 2, unwanted, lemmatizer) or "documents"

    return sanitize_filename(foldername, 2), sanitize_filename(filename, 3), description


def generate_image_metadata(file_path, image_inference, text_inference):
    """Return ``(foldername, filename, description)`` for an image via the VLM."""
    caption_prompt = (
        "Describe this image in one or two sentences, focusing on the main subject and "
        "context. Be specific and concrete."
    )
    description = (image_inference.generate(file_path, caption_prompt) or "").strip()
    if not description:
        # Fall back to the filename stem so we still produce a usable operation.
        stem = os.path.splitext(os.path.basename(file_path))[0]
        return "images", sanitize_filename(stem, 3), ""

    unwanted = get_unwanted_words()
    lemmatizer = _lemmatizer()

    filename = clean_ai_output(_filename_from_summary(description, text_inference), 3, unwanted, lemmatizer)
    if not filename:
        filename = clean_ai_output(description, 3, unwanted, lemmatizer) or "image"

    foldername = clean_ai_output(_foldername_from_summary(description, text_inference), 2, unwanted, lemmatizer)
    if not foldername:
        foldername = clean_ai_output(description, 2, unwanted, lemmatizer) or "images"

    return sanitize_filename(foldername, 2), sanitize_filename(filename, 3), description
