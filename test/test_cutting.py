import os
import shutil

import histoprep
import pandas
import pytest
from PIL import Image

from paths import DATA_PATH

SLIDE_1 = os.path.join(DATA_PATH, "slide_1.svs")
SLIDE_2 = os.path.join(DATA_PATH, "slide_2.mrxs")
SLIDE_3 = os.path.join(DATA_PATH, "slide_3.czi")
SLIDE_4 = os.path.join(DATA_PATH, "tile.jpeg")
OUTPUT_DIR = os.path.join(DATA_PATH, "output")


def test_cutting_tiles():
    for path, width in [
        (SLIDE_1, 1024),
        (SLIDE_2, 1024),
        (SLIDE_3, 1024),
        (SLIDE_4, 32),
    ]:
        reader = histoprep.SlideReader(path)
        # Get coords.
        coordinates = reader.get_tile_coordinates(
            width, overlap=0.1, max_background=0.1
        )[:2]
        # Save tiles.
        reader.save_tiles(OUTPUT_DIR, coordinates=coordinates, num_workers=1)
        assert isinstance(reader.tile_metadata, pandas.DataFrame)
        assert isinstance(reader.annotated_thumbnail_tiles, Image.Image)
    # Overwrite is False.
    with pytest.warns(UserWarning):
        assert (
            reader.save_tiles(
                OUTPUT_DIR, coordinates=coordinates, num_workers=1
            )
            is None
        )
    shutil.rmtree(OUTPUT_DIR)
