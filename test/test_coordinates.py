import os

import histoprep.functional as F
import pytest
from histoprep.helpers import read_image

from paths import DATA_PATH

IMAGE = read_image(os.path.join(DATA_PATH, "tile.jpeg"))
__, MASK = F.detect_tissue(IMAGE)


def test_tile_coordinates():
    coordinates = F.tile_coordinates(
        dimensions=(256, 256), width=32, overlap=0.25
    )
    assert len(coordinates[0]) == 4
    assert coordinates[0] == (0, 0, 32, 32)
    assert len(coordinates) == 121
    coordinates = F.tile_coordinates(
        dimensions=(256, 256), width=32, height=64, overlap=0.25
    )
    assert len(coordinates[0]) == 4
    assert coordinates[0] == (0, 0, 32, 64)
    assert len(coordinates) == 66


def test_tile_coordinates_failure():
    with pytest.raises(TypeError):
        F.tile_coordinates(dimensions=(256, 256), width=1.0, overlap=0.25)
    with pytest.raises(ValueError):
        F.tile_coordinates(dimensions=(256, 256), width=300, overlap=0.25)
    with pytest.raises(ValueError):
        F.tile_coordinates(dimensions=(256, 256), width=0, overlap=0.25)
    with pytest.raises(ValueError):
        F.tile_coordinates(
            dimensions=(256, 256), width=1, height=0, overlap=0.25
        )
    with pytest.raises(ValueError):
        F.tile_coordinates(dimensions=(256, 256), width=32, overlap=-1)
    with pytest.raises(ValueError):
        F.tile_coordinates(dimensions=(256, 256), width=32, overlap=1)
    with pytest.raises(ValueError):
        F.tile_coordinates(dimensions=(256, 256), width=32, overlap=1.2423)


def test_filter_tile_coordinates():
    coords = F.tile_coordinates(dimensions=MASK.shape, width=32, overlap=0.25)
    filtered_coordinates = F.filter_coordinates(
        coords, MASK, max_background=0.5
    )
    assert len(filtered_coordinates[0]) == 4  # (X, Y, W, H)
    assert len(F.filter_coordinates([], MASK, max_background=0.5)) == 0
    assert len(F.filter_coordinates(coords, MASK, max_background=1.0)) == len(
        coords
    )


def test_filter_tile_coordinates_failure():
    coords = F.tile_coordinates(dimensions=MASK.shape, width=32, overlap=0.25)
    with pytest.raises(TypeError):
        F.filter_coordinates(-1, MASK, max_background=0.5)
    with pytest.raises(ValueError):
        F.filter_coordinates([0], MASK, max_background=0.5)
    with pytest.raises(ValueError):
        F.filter_coordinates([(0, 4235)], MASK, max_background=0.5)
    with pytest.raises(ValueError):
        F.filter_coordinates(coords, MASK, max_background=-1)
    with pytest.raises(ValueError):
        F.filter_coordinates(coords, MASK, max_background=1.2)
