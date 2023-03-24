from PIL import Image

import histoprep.functional as F
from histoprep import SlideReader

from ._utils import SLIDE_PATH_JPEG, TMP_DIRECTORY, clean_temporary_directory


def test_image_collage() -> None:
    clean_temporary_directory()
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(None, width=128)
    metadata = reader.save_regions(TMP_DIRECTORY, tile_coords)
    collage = F.get_random_image_collage(
        metadata["path"], num_rows=4, num_cols=8, shape=(32, 32)
    )
    assert isinstance(collage, Image.Image)
    assert collage.size == (8 * 32, 4 * 32)
    # Not enough images for all rows.
    collage = F.get_random_image_collage(
        metadata["path"][:6], num_rows=4, num_cols=8, shape=(32, 32)
    )
    assert isinstance(collage, Image.Image)
    assert collage.size == (8 * 32, 1 * 32)
    clean_temporary_directory()
