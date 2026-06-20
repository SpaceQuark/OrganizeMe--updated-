"""Optional Rust acceleration for the I/O- and CPU-bound systems core.

Wraps the ``organizeme_native`` extension (parallel directory walking and
BLAKE3 content hashing). When the extension is not installed, ``HAVE_NATIVE``
is ``False`` and callers fall back to the pure-Python implementations.
"""

try:
    import organizeme_native as _native
except ImportError:
    _native = None

HAVE_NATIVE = _native is not None


def scan_paths(base_path, exclude_dirs):
    """Return every file path under ``base_path`` via the native parallel walk."""
    return _native.scan_paths(base_path, list(exclude_dirs))


def hash_files(paths):
    """Return ``{path: blake3_hex_or_None}`` via parallel native hashing."""
    return dict(_native.hash_files(list(paths)))
