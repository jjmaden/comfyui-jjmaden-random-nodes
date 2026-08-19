"""Small helpers shared by nodes that resolve pasted file paths and cache by mtime."""

import os


def strip_wrapping(s):
    """Strip whitespace and a single layer of surrounding quotes from a pasted path."""
    return (s or "").strip().strip('"').strip("'")


def mtime_or_nan(path):
    """os.path.getmtime(path), or float('nan') if the file can't be stat'd."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return float("nan")
