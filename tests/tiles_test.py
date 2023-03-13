import numpy as np
import pytest
from PIL import Image

from histoprep import functional as F

from .utils import DATA_DIRECTORY


def test_tile_coordinates() -> None:
    kwargs = {"dimensions": (100, 80), "width": 40, "height": 30}
    # No out-of-bounds.
    assert F.get_tile_coordinates(**kwargs) == [
        (0, 0, 40, 30),
        (40, 0, 40, 30),
        (0, 30, 40, 30),
        (40, 30, 40, 30),
        (0, 60, 40, 30),
        (40, 60, 40, 30),
    ]
    # Out-of-bounds.
    assert F.get_tile_coordinates(**kwargs, out_of_bounds=True) == [
        (0, 0, 40, 30),
        (40, 0, 40, 30),
        (0, 30, 40, 30),
        (40, 30, 40, 30),
        (0, 60, 40, 30),
        (40, 60, 40, 30),
        (0, 90, 40, 30),
        (40, 90, 40, 30),
    ]
    # Overlap.
    assert F.get_tile_coordinates(**kwargs, overlap=0.25) == [
        (0, 0, 40, 30),
        (30, 0, 40, 30),
        (0, 22, 40, 30),
        (30, 22, 40, 30),
        (0, 44, 40, 30),
        (30, 44, 40, 30),
        (0, 66, 40, 30),
        (30, 66, 40, 30),
    ]
    # Overlap step should be always at least 1.
    assert F.get_tile_coordinates((4, 4), width=2, overlap=0.99999999999) == [
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
    # Overlap should be between [0, 1).
    with pytest.raises(ValueError):
        F.get_tile_coordinates(**kwargs, overlap=1.0)
    with pytest.raises(ValueError):
        F.get_tile_coordinates(**kwargs, overlap=-1)
    # Width should be smaller than dim and positive integer.
    with pytest.raises(ValueError):
        F.get_tile_coordinates((10, 10), width=0)
    with pytest.raises(TypeError):
        F.get_tile_coordinates((10, 10), width=1.5)
    with pytest.raises(ValueError):
        F.get_tile_coordinates((10, 10), width=11)


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
