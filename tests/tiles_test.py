import numpy as np
import pytest
from PIL import Image

from histoprep import SlideReader
from histoprep import functional as F

from ._utils import DATA_DIRECTORY, SLIDE_PATH_JPEG

KWARGS = {"dimensions": (100, 80), "width": 40, "height": 30}


def test_tiles() -> None:
    assert F.get_tile_coordinates(**KWARGS) == [
        (0, 0, 40, 30),
        (40, 0, 40, 30),
        (0, 30, 40, 30),
        (40, 30, 40, 30),
        (0, 60, 40, 30),
        (40, 60, 40, 30),
    ]


def test_tiles_out_of_bounds() -> None:
    # Out-of-bounds.
    assert F.get_tile_coordinates(**KWARGS, out_of_bounds=True) == [
        (0, 0, 40, 30),
        (40, 0, 40, 30),
        (0, 30, 40, 30),
        (40, 30, 40, 30),
        (0, 60, 40, 30),
        (40, 60, 40, 30),
        (0, 90, 40, 30),
        (40, 90, 40, 30),
    ]


def test_tiles_overlap() -> None:
    assert F.get_tile_coordinates(**KWARGS, overlap=0.25) == [
        (0, 0, 40, 30),
        (30, 0, 40, 30),
        (0, 22, 40, 30),
        (30, 22, 40, 30),
        (0, 44, 40, 30),
        (30, 44, 40, 30),
        (0, 66, 40, 30),
        (30, 66, 40, 30),
    ]


def test_tiles_overlap_always_at_least_one() -> None:
    assert F.get_tile_coordinates((4, 4), width=2, overlap=0.9999999999999999) == [
        (0, 0, 2, 2),
        (1, 0, 2, 2),
        (2, 0, 2, 2),
        (0, 1, 2, 2),
        (1, 1, 2, 2),
        (2, 1, 2, 2),
        (0, 2, 2, 2),
        (1, 2, 2, 2),
        (2, 2, 2, 2),
    ]


def test_tiles_bad_inputs() -> None:
    # Overlap should be between [0, 1).
    with pytest.raises(ValueError, match="Overlap should be in range"):
        F.get_tile_coordinates(**KWARGS, overlap=1.0)
    with pytest.raises(ValueError, match="Overlap should be in range"):
        F.get_tile_coordinates(**KWARGS, overlap=-1)
    # Width should be smaller than dim and positive integer.
    with pytest.raises(ValueError, match="should non-zero positive integers"):
        F.get_tile_coordinates((10, 10), width=0)
    with pytest.raises(TypeError, match="should be integers"):
        F.get_tile_coordinates((10, 10), width=1.5)
    with pytest.raises(ValueError, match="should be smaller than image dimensions"):
        F.get_tile_coordinates((10, 10), width=11)


def test_background_percentages() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    __, mask = reader.get_tissue_mask(level=-1)
    tile_coords = F.get_tile_coordinates(reader.dimensions, 1000)
    assert F.get_background_percentages(
        tile_coords, mask, F.get_downsample(mask, reader.dimensions)
    ) == [0.3456, 0.297744, 0.435136, 0.29896]


def test_overlap_area() -> None:
    coordinates = [[0, 0, 100, 100], [0, 0, 4, 4], [4, 4, 2, 2], [11, 11, 2, 2]]
    assert F.get_overlap_area((5, 5, 5, 5), coordinates).tolist() == [25, 0, 1, 0]


def test_overlap_index() -> None:
    coordinates = [[0, 0, 5, 5], [0, 0, 5, 6], [0, 0, 6, 6], [10, 10, 1, 1]]
    assert F.get_overlap_index((5, 5, 5, 5), coordinates).tolist() == [2]


def test_read_region() -> None:
    image = np.arange(9).reshape(3, 3)
    assert (
        F.get_region_from_array(image, (0, 0, 2, 2)) == np.array([[0, 1], [3, 4]])
    ).all()


def test_draw_tiles() -> None:
    image = np.zeros((200, 200), dtype=np.uint8)
    dimensions = image.shape[:2]
    coords = F.get_tile_coordinates(dimensions, width=40)
    img = F.draw_tiles(
        image=image,
        coordinates=coords,
        downsample=1.0,
        rectangle_fill=None,
        rectangle_outline="red",
        rectangle_width=2,
        highlight_first=True,
        text_items=range(len(coords)),
        text_color="white",
        text_proportion=0.8,
        text_font="monospace",
    )
    arr1 = np.array(img)
    arr2 = np.array(Image.open(DATA_DIRECTORY / "correctly_drawn_tiles.png"))
    assert (arr1 == arr2).all()
