import math
import random
from typing import List, Union

import numpy
import pandas
from PIL import Image

from ._io import read_and_resize

__all__ = ["random_tile_collage"]


def random_tile_collage(
    paths: Union[List[str], pandas.Series, numpy.ndarray],
    nrows: int = 16,
    ncols: int = 32,
    px: int = 32,
) -> Image.Image:
    """Plot random selection of the given paths.

    Args:
        paths: Image paths.
        nrows: Number of rows in collage. Defaults to 16.
        ncols:  Number of columns in collage. Defaults to 32.
        px: Size of each tile in collage. Defaults to 32.

    Returns:
        Collage image of random tiles.

    Example:
        ```python
        import matplotlib.pyplot as plt
        from histoprep.helpers import random_tile_collage, combine_metadata

        # Load metadata.
        metadata = combine_metadata("/output_dir/")
        data_loss_paths = metadata["black_pixels" > 0.05]["path"]
        # Plot some tiles with data loss.
        plt.imshow(random_tile_collage(data_loss_paths))
        ```
    """
    if isinstance(paths, (pandas.Series, numpy.ndarray)):
        paths = paths.tolist()
    if not isinstance(paths, list):
        raise TypeError("Expected a list of paths, not {}.".format(type(paths)))
    elif len(paths) == 0:
        return
    elif not isinstance(paths[0], str):
        raise TypeError(
            "Expected list items to be 'str', not {}.".format(type(paths[0]))
        )
    if not isinstance(ncols, int) or not isinstance(nrows, int):
        raise TypeError("Number of columns and rows should be integers.")
    if not isinstance(px, int):
        raise TypeError("Tile size should be an integer.")
    # Select random paths.
    paths = random.sample(paths, min(nrows * ncols, len(paths)))
    # Remove rows if necessary.
    nrows = math.ceil(len(paths) / ncols)
    while len(paths) < nrows * ncols:
        paths.append(None)
    # Build collage.
    row = []
    collage = []
    for tile in read_and_resize(paths, px):
        row.append(tile)
        if len(row) == ncols:
            collage.append(numpy.hstack(row))
            row = []
    return Image.fromarray(numpy.vstack(collage))
