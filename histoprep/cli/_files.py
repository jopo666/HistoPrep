import os
from pathlib import Path
from typing import Union


def convert_to_path(ctx, param, value) -> Path:  # noqa.
    if value is not None:
        return Path(value)
    return None


def find_files(parent_dir: Union[str, Path], suffix: str, depth: int = 1) -> list[Path]:
    """Find files with a given suffix.

    Args:
        parent_dir: Path to search directory.
        suffix: File suffix to match.
        depth: Depth for traversal. Defaults to 1.


    Returns:
        List of paths with the given extension.
    """
    if isinstance(parent_dir, str):
        parent_dir = Path(parent_dir)
    return _list_files(parent_dir, suffix, 1, depth)


def _list_files(
    path: Path, suffix: str, current_depth: int, max_depth: int
) -> list[str]:
    """Recursive helper function for `find_files`."""
    output = []
    for f in os.scandir(path):
        if f.is_dir() and current_depth < max_depth:
            output += _list_files(f.path, suffix, current_depth + 1, max_depth)
        elif f.name.endswith(suffix):
            output.append(Path(f.path))
    return output
