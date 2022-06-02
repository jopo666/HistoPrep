import os
import warnings
from os.path import basename, dirname, exists, join
from typing import List, Tuple

import pandas

from ._multiprocess import multiprocess_loop
from ._verbose import progress_bar

METRIC_PREFIXES = {
    "red",
    "green",
    "blue",
    "gray",
    "saturation",
    "hue",
    "brightness",
    "sharpness",
    "black",
    "white",
}

__all__ = ["combine_metadata", "strip_metric_colums"]


def strip_metric_colums(metadata: pandas.DataFrame) -> pandas.DataFrame:
    """Remove columns with preprocessing metrics.

    Args:
        metadata: Tile metadata.

    Returns:
        Tile metadata without preprocessing metrics.

    Example:
        ```python
        from histoprep.helpers import combine_metadata, strip_metric_colums

        metadata = combine_metadata("/output/dir")
        tile_info = strip_metric_columns(metadata)
        ```
    """
    keep = []
    for col in metadata.columns:
        prefix = col.split("_")[0]
        if prefix not in METRIC_PREFIXES:
            keep.append(col)
    return metadata[keep]


def __yield_paths(parent_dir: str, filenames: List[str]):
    """Helper function for yielding metadata fileapaths."""
    for f in os.scandir(parent_dir):
        if f.is_dir():
            for name in filenames:
                path = join(f.path, name)
                if exists(path):
                    yield path


def combine_metadata(
    parent_dir: str, filename: str = "tile_metadata.csv"
) -> pandas.DataFrame:
    """Combine metadata under `parent_dir` into a single dataframe.

    Args:
        parent_dir: Output directory with the processed slides.
        filename: Filename to match. Defaults to "tile_metadata.csv".

    Raises:
        IOError: Parent directory does not exist.
        NotADirectoryError: Parent directory is not a directory.

    Returns:
        pandas.DataFrame: Combined metadata.

     Example:
        ```python
        from histoprep.helpers import combine_metadata

        metadata = combine_metadata("/output/dir")
        ```
    """
    if not exists(parent_dir):
        raise IOError("Directory {} does not exist.".format(parent_dir))
    elif os.path.isfile(parent_dir):
        raise NotADirectoryError("{} is not a directory.".format(parent_dir))
    # Collect metadata.
    combined = []
    for metadata in progress_bar(
        multiprocess_loop(__load_metadata, __yield_paths(parent_dir, [filename])),
        desc="Combining metadata",
    ):
        if metadata is not None:
            combined.append(metadata)
    if len(combined) == 0:
        return pandas.DataFrame()
    else:
        return pandas.concat(combined).reset_index(drop=True)


def __load_metadata(path: str):
    """Parallelizable metadata loading."""
    try:
        return pandas.read_csv(path)
    except Exception as e:
        warnings.warn("Could not load {} due to exception: {}.".format(path, e))
        return None


def rename_paths(parent_dir: str) -> List[Tuple[str, Exception]]:
    """Finds all metadata files inside `parent_dir` and updates the `path`
    column to match the current directory. Useful if you rename/move around
    the directories with the processed data.

    Args:
        parent_dir: Output directory with the processed slides.

    Returns:
        List csv-paths and Exceptions, for the dataframes which could not be
        updated.

     Example:
        ```python
        from histoprep.helpers import rename_paths

        failures = rename_paths("/new/output/dir")
        ```
    """
    if not exists(parent_dir):
        raise IOError("Directory {} does not exist.".format(parent_dir))
    elif os.path.isfile(parent_dir):
        raise NotADirectoryError("{} is not a directory.".format(parent_dir))
    # Rename paths in metadata
    failures = []
    for path, exception in progress_bar(
        multiprocess_loop(
            __rename_paths,
            __yield_paths(parent_dir, ["tile_metadata.csv", "spot_metadata.csv"]),
        ),
        desc="Renaming paths",
    ):
        if isinstance(exception, Exception):
            failures.append((path, exception))
    return failures


def __rename_paths(path: tuple):
    """Parallelizable path renaming."""
    try:
        # Define new directory.
        if basename(path) == "spot_metadata.csv":
            new_dir = join(dirname(path), "spots")
        elif basename(path) == "tile_metadata.csv":
            new_dir = join(dirname(path), "tiles")
        # Load metadata.
        metadata = pandas.read_csv(path)
        # Rename paths.
        metadata["path"] = [join(new_dir, basename(x)) for x in metadata["path"]]
        # Save new metadata.
        metadata.to_csv(path, index=False)
        return None, None
    except Exception as e:
        return path, e
