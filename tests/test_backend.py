import pytest
from aicspylibczi import CziFile
from openslide import OpenSlide
from PIL import Image

from histoprep import LevelNotFoundError
from histoprep.backend import CziReader, OpenSlideReader, PillowReader
from tests.utils import SLIDE_PATH_CZI, SLIDE_PATH_JPEG, SLIDE_PATH_MRXS, SLIDE_PATH_SVS


def read_regions_with_reader(reader, tile_width=256) -> None:
    with pytest.raises(LevelNotFoundError):
        reader.read_region((0, 0, 10, 10), level=193273645)
    assert reader.read_region((0, 0, 0, 0), 0).shape == (0, 0, 3)
    assert reader.read_region((0, 0, 1, 0), 0).shape == (0, 1, 3)
    if len(reader.level_dimensions) > 1:
        assert reader.read_region((0, 0, 1, 1), 1).shape == (0, 0, 3)
    assert reader.read_region((0, 0, 100, 100), 0).shape == (100, 100, 3)
    for level in reader.level_dimensions:
        tile_dims = reader.read_region(
            (0, 0, tile_width, tile_width), level=level
        ).shape
        expected_dims = (tile_width // 2**level, tile_width // 2**level, 3)
        assert tile_dims == expected_dims


def test_pillow_backend() -> None:
    reader = PillowReader(SLIDE_PATH_JPEG)
    assert isinstance(reader.backend, Image.Image)
    assert reader.dimensions == (1000, 2000)
    assert reader.level_dimensions == {0: (1000, 2000), 1: (500, 1000)}
    assert reader.level_downsamples == {0: (1.0, 1.0), 1: (2.0, 2.0)}
    assert reader.level_count == 2
    assert reader.data_bounds == (0, 0, 2000, 1000)
    for level, dimensions in reader.level_dimensions.items():
        assert reader.read_level(level).shape[:2] == dimensions
    read_regions_with_reader(reader)


def test_openslide_backend() -> None:
    # Single level.
    reader = OpenSlideReader(SLIDE_PATH_SVS)
    assert isinstance(reader.backend, OpenSlide)
    assert reader.dimensions == (2967, 2220)
    assert reader.level_dimensions == {0: (2967, 2220)}
    assert reader.level_downsamples == {0: (1.0, 1.0)}
    assert reader.level_count == 1
    assert reader.data_bounds == (0, 0, 2220, 2967)
    read_regions_with_reader(reader)
    # Multi level.
    reader = OpenSlideReader(SLIDE_PATH_MRXS)
    assert isinstance(reader.backend, OpenSlide)
    assert reader.dimensions == (410429, 170489)
    assert reader.level_dimensions == {
        0: (410429, 170489),
        1: (205214, 85244),
        2: (102607, 42622),
        3: (51303, 21311),
        4: (25651, 10655),
        5: (12825, 5327),
        6: (6412, 2663),
        7: (3206, 1331),
        8: (1603, 665),
    }
    assert reader.level_downsamples == {
        0: (1.0, 1.0),
        1: (2.000004872961884, 2.000011731030923),
        2: (4.000009745923768, 4.000023462061846),
        3: (8.000097460187513, 8.000046924123692),
        4: (16.00050680285369, 16.000844673862037),
        5: (32.002261208576996, 32.004693073024214),
        6: (64.00951341235184, 64.02140443109275),
        7: (128.01902682470367, 128.0909090909091),
        8: (256.03805364940735, 256.3744360902256),
    }
    assert reader.level_count == 9
    assert reader.data_bounds == (21065, 179718, 94074, 410429)
    read_regions_with_reader(reader)


def test_czi_backend() -> None:
    reader = CziReader(SLIDE_PATH_CZI)
    assert isinstance(reader.backend, CziFile)
    assert reader.dimensions == (107903, 188868)
    assert reader.level_dimensions == {
        0: (107903, 188868),
        1: (53951, 94434),
        2: (26975, 47217),
        3: (13487, 23608),
        4: (6743, 11804),
        5: (3371, 5902),
        6: (1685, 2951),
        7: (842, 1475),
    }
    assert reader.level_downsamples == {
        0: (1.0, 1.0),
        1: (2.000018535337621, 2.0),
        2: (4.000111214087117, 4.0),
        3: (8.000519018313932, 8.00016943409014),
        4: (16.002224529141333, 16.00033886818028),
        5: (32.009196084247996, 32.00067773636056),
        6: (64.03738872403561, 64.00135547272112),
        7: (128.15083135391924, 128.04610169491525),
    }
    assert reader.level_count == 8
    assert reader.data_bounds == (0, 0, 188868, 107903)
    with pytest.warns():
        reader.read_region((0, 0, 100, 100), level=2)
    read_regions_with_reader(reader)
