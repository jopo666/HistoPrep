import os
import shutil
from typing import Tuple, Union

# def find_files(
#     parent_dir: str, extension: Union[Tuple[str], str], depth: int = 0
# ) -> List[str]:
#     """Find all files in the ``parent_dir`` with extension.

#     Args:
#         parent_dir: Parent directory to loop.
#         extension: File extension(s).
#         depth: Depth for recursive search. Set -1 for fully recursive search.
#             Defaults to 0.

#     Returns:
#        Paths with the desired extension(s).

#     Example:
#         ```python
#         from histoprep.helpers import find_files

#         # Load spot metadata and images.
#         metadata = find_files("path/to/output_dir/", extension="csv", depth=1)
#         spots = find_files("path/to/output_dir/", extension="jpeg", depth=2)
#         ```
#     """
#     if not os.path.isdir(parent_dir):
#         raise ValueError("{} is not a directory".format(parent_dir))
#     if depth < 0 or not isinstance(depth, int):
#         raise TypeError("Depth should be a positive integer.")
#     if isinstance(extension, str):
#         if not extension.startswith("."):
#             extension = "." + extension
#     elif isinstance(extension, (tuple, list)):
#         if len(extension) > 0 and not isinstance(extension[0], str):
#             raise TypeError(
#                 "Extension should be a string not {}.".format(type(extension[0]))
#             )
#         extension = tuple(x if x.startswith(".") else "." + x for x in extension)
#     else:
#         raise ValueError(
#             "Expected extension to be a tuple or string not {}.".format(type(extension))
#         )
#     return _list_files(parent_dir, extension, 0, depth)


# def _list_files(dir_path: str, extensions: tuple, current_depth: int, max_depth: int):
#     """Helper function to list files recursively."""
#     file_paths = []
#     for f in os.scandir(dir_path):
#         if f.is_file() and f.name.endswith(extensions):
#             file_paths.append(f.path)
#         if f.is_dir() and (max_depth == -1 or current_depth + 1 <= max_depth):
#             file_paths += _list_files(f.path, extensions, current_depth + 1, max_depth)
#     return file_paths


def remove_extension(path: str) -> str:
    """Return filename with the extension removed."""
    filename = os.path.basename(path)
    if "." in filename:
        filename = ".".join(filename.split(".")[:-1])
    return os.path.join(os.path.dirname(path), filename)


def get_extension(path: str) -> str:
    """Return file extension."""
    filename = os.path.basename(path)
    if "." in filename:
        return filename.split(".")[-1]
    else:
        return None


def remove_directory(dir_path: str) -> None:
    """Remove directory and all files.

    Args:
        dir_path: Directory to be removed.

    Example:
        ```python
        from histoprep.helpers import remove_dir

        remove_dir("/shitty/slide/output_dir")
        ```
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError("Path %s does not exist." % dir_path)
    elif os.path.isfile(dir_path):
        raise ValueError("Path {} is not a directory.".format(dir_path))
    shutil.rmtree(dir_path)


def remove(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
