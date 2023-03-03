import os

import numpy as np
import pytest
from PIL import Image

from histoprep import functional as F

from .utils import DATA_DIRECTORY


def test_tile_coordinates():
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


def test_draw_tiles():
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
    arr2 = np.array(
        Image.open(os.path.join(DATA_DIRECTORY, "correctly_drawn_tiles.png"))
    )
    assert (arr1 == arr2).all()


def test_multiply():
    assert F.multiply_xywh((100, 100, 200, 200), 1 / 8) == (800, 800, 1600, 1600)
    assert F.multiply_xywh((100, 100, 200, 200), 1 / 4) == (400, 400, 800, 800)
    assert F.multiply_xywh((100, 100, 200, 200), 1 / 2) == (200, 200, 400, 400)
    assert F.multiply_xywh((100, 100, 200, 200), 1) == (100, 100, 200, 200)
    assert F.multiply_xywh((100, 100, 200, 200), 2) == (50, 50, 100, 100)
    assert F.multiply_xywh((100, 100, 200, 200), 4) == (25, 25, 50, 50)
    assert F.multiply_xywh((100, 100, 200, 200), 8) == (12, 12, 25, 25)
    assert F.multiply_xywh((100, 100, 200, 200), 16) == (6, 6, 12, 12)
    assert F.multiply_xywh((100, 100, 200, 200), 32) == (3, 3, 6, 6)


def test_allowed():
    DIMENSIONS = (20, 5)
    assert F.allowed_xywh((0, 0, 10, 15), DIMENSIONS) == (0, 0, 5, 15)
    assert F.allowed_xywh((0, 0, 100000, 1000000), DIMENSIONS) == (0, 0, 5, 20)
    assert F.allowed_xywh((0, 0, 5, 20), DIMENSIONS) == (0, 0, 5, 20)
    assert F.allowed_xywh((10000, 0, 5, 20), DIMENSIONS) == (10000, 0, 0, 0)
    assert F.allowed_xywh((5, 20, 10, 10), DIMENSIONS) == (5, 20, 0, 0)
    assert F.allowed_xywh((5, 19, 10, 10), DIMENSIONS) == (4, 19, 0, 1)
    assert F.allowed_xywh((4, 20, 10, 10), DIMENSIONS) == (4, 20, 1, 0)
