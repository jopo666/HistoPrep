import histoprep.functional as F
from histoprep import SlideReader

from ._utils import SLIDE_PATH_JPEG, TMP_DIRECTORY, clean_temporary_directory


def test_mean_and_std_from_images() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(None, 512)
    images = (tile for tile, __ in reader.yield_regions(tile_coords))
    mean, std = F.get_mean_and_std_from_images(images)
    assert [round(x, 2) for x in mean] == [0.84, 0.70, 0.78]
    assert [round(x, 2) for x in std] == [0.14, 0.19, 0.14]


def test_mean_and_std_from_paths() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(None, 512)
    clean_temporary_directory()
    reader.save_regions(TMP_DIRECTORY, tile_coords)
    paths = list((TMP_DIRECTORY / "slide" / "tiles").iterdir())
    mean, std = F.get_mean_and_std_from_paths(paths)
    clean_temporary_directory()
    assert [round(x, 2) for x in mean] == [0.84, 0.70, 0.78]
    assert [round(x, 2) for x in std] == [0.14, 0.19, 0.14]
