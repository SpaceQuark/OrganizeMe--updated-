from collections import defaultdict
import os

def find_file_paths(
    base_file_path,
    recursive=True,
    follow_symlinks=False,
    extensions=None,
    exclude_dirs=None,
    max_depth=None
):
    """
    Generator function to find file paths within a directory.

    Args:
        base_file_path (str): The root directory to start searching from.
        recursive (bool): Whether to traverse directories recursively.
        follow_symlinks (bool): Whether to follow symbolic links.
        extensions (list or tuple, optional): File extensions to include (e.g., ['.txt', '.jpg']).
        exclude_dirs (list or tuple, optional): Directory names to exclude from traversal.
        max_depth (int, optional): Maximum depth to traverse. None means no limit.

    Yields:
        str: Full path to each file that matches the criteria.
    """
    base_depth = base_file_path.rstrip(os.path.sep).count(os.path.sep)
    for root, dirs, files in os.walk(base_file_path, followlinks=follow_symlinks):
        current_depth = root.count(os.path.sep) - base_depth
        
        if max_depth is not None and current_depth > max_depth:
            break
        
        if exclude_dirs:
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

        
        for file in files:
            if extensions and not file.lower().endswith(tuple(extensions)):
                continue
            yield os.path.join(root, file)
        
        if not recursive:
            break

def separate_by_type(base_file_path):
    """
    Separate files in a directory by type (e.g. text, image, audio).

    Args:
        base_path (str): The root directory to organize files from.

    Returns:
        dict: A dictionary where keys are file types (e.g., 'text', 'image', 'audio'),
              and values are lists of file paths in those categories.
    """

    file_types = {
        "text": [".txt", ".doc", ".docx", ".pdf"],
        "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".svg"],
        "audio": [".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"],
        "video": [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv"],
        "archive": [".zip", ".tar", ".gz", ".rar", ".7z"],
        "spreadsheet": [".xls", ".xlsx", ".csv"],
    }
    
    # inverse mapping for lookup
    ext_to_type = {
        ext: file_type for file_type, extensions in file_types.items() for ext in extensions
    }
    
    cat_files = defaultdict(list)
    
    for file_path in find_file_paths(base_file_path, recursive=True):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # if unknown default to "other"
        file_type = ext_to_type.get(ext, "other")  
        cat_files[file_type].append(file_path)
    
    return dict(cat_files)
