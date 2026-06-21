use std::collections::HashSet;
use std::sync::Arc;

use jwalk::WalkDir;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Parallel directory walk. Returns every file path under `base`, pruning any
/// directory whose name is in `exclude_dirs` (the whole subtree is skipped, not
/// just the directory entry itself).
#[pyfunction]
fn scan_paths(base: &str, exclude_dirs: Vec<String>) -> PyResult<Vec<String>> {
    let excluded: Arc<HashSet<String>> = Arc::new(exclude_dirs.into_iter().collect());
    let excluded_for_prune = Arc::clone(&excluded);

    let walker = WalkDir::new(base).skip_hidden(false).process_read_dir(
        move |_depth, _path, _state, children| {
            children.retain(|child| match child {
                Ok(entry) => {
                    // Prune excluded directories so their contents are never walked.
                    if entry.file_type().is_dir() {
                        match entry.file_name().to_str() {
                            Some(name) => !excluded_for_prune.contains(name),
                            None => true,
                        }
                    } else {
                        true
                    }
                }
                Err(_) => true,
            });
        },
    );

    let mut out = Vec::new();
    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if entry.file_type().is_file() {
            if let Some(p) = entry.path().to_str() {
                out.push(p.to_string());
            }
        }
    }
    Ok(out)
}

/// Parallel BLAKE3 hashing. Returns `(path, hex_digest_or_None)` pairs; files
/// that cannot be read map to `None` (mirrors the Python fallback contract).
#[pyfunction]
fn hash_files(py: Python<'_>, paths: Vec<String>) -> PyResult<Vec<(String, Option<String>)>> {
    py.allow_threads(|| {
        let results = paths
            .par_iter()
            .map(|p| {
                let digest = std::fs::read(p)
                    .ok()
                    .map(|bytes| blake3::hash(&bytes).to_hex().to_string());
                (p.clone(), digest)
            })
            .collect();
        Ok(results)
    })
}

#[pymodule]
fn organizeme_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_paths, m)?)?;
    m.add_function(wrap_pyfunction!(hash_files, m)?)?;
    Ok(())
}
