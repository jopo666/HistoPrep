import pytest
from aicspylibczi import CziFile
from openslide import OpenSlide
from PIL import Image

from histoprep.backend import CziBackend, OpenSlideBackend, PillowBackend
from tests.utils import SLIDE_PATH_CZI, SLIDE_PATH_JPEG, SLIDE_PATH_MRXS, SLIDE_PATH_SVS


def read_regions_with_reader(backend, tile_width=256) -> None:
    with pytest.raises(ValueError):
        backend.read_region((0, 0, 10, 10), level=193273645)
    assert backend.read_region((0, 0, 0, 0), 0).shape == (0, 0, 3)
    assert backend.read_region((0, 0, 1, 0), 0).shape == (0, 1, 3)
    if len(backend.level_dimensions) > 1:
        assert backend.read_region((0, 0, 1, 1), 1).shape == (0, 0, 3)
    assert backend.read_region((0, 0, 100, 100), 0).shape == (100, 100, 3)
    for level in backend.level_dimensions:
        tile_dims = backend.read_region(
            (0, 0, tile_width, tile_width), level=level
        ).shape
        expected_dims = (tile_width // 2**level, tile_width // 2**level, 3)
        assert tile_dims == expected_dims


def test_pillow_backend() -> None:
    backend = PillowBackend(SLIDE_PATH_JPEG)
    assert isinstance(backend.reader, Image.Image)
    assert backend.dimensions == (1000, 2000)
    assert backend.read_level(1).shape == (*backend.level_dimensions[1], 3)
    assert backend.level_dimensions == {0: (1000, 2000), 1: (500, 1000)}
    assert backend.level_downsamples == {0: (1.0, 1.0), 1: (2.0, 2.0)}
    assert backend.level_count == 2
    assert backend.data_bounds == (0, 0, 2000, 1000)
    for level, dimensions in backend.level_dimensions.items():
        assert backend.read_level(level).shape[:2] == dimensions
    read_regions_with_reader(backend)


def test_openslide_backend() -> None:
    # Single level.
    backend = OpenSlideBackend(SLIDE_PATH_SVS)
    assert isinstance(backend.reader, OpenSlide)
    assert backend.dimensions == (2967, 2220)
    assert backend.read_level(0).shape == (*backend.level_dimensions[0], 3)
    assert backend.level_dimensions == {0: (2967, 2220)}
    assert backend.level_downsamples == {0: (1.0, 1.0)}
    assert backend.level_count == 1
    assert backend.data_bounds == (0, 0, 2220, 2967)
    read_regions_with_reader(backend)
    # Multi level.
    backend = OpenSlideBackend(SLIDE_PATH_MRXS)
    assert isinstance(backend.reader, OpenSlide)
    assert backend.dimensions == (410429, 170489)
    assert backend.read_level(7).shape == (*backend.level_dimensions[7], 3)
    assert backend.level_dimensions == {
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
    assert backend.level_downsamples == {
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
    assert backend.level_count == 9
    assert backend.data_bounds == (21065, 179718, 94074, 410429)
    read_regions_with_reader(backend)


def test_czi_backend() -> None:
    backend = CziBackend(SLIDE_PATH_CZI)
    assert isinstance(backend.reader, CziFile)
    assert backend.dimensions == (107903, 188868)
    assert backend.level_dimensions == {
        0: (107903, 188868),
        1: (53952, 94434),
        2: (26976, 47217),
        3: (13488, 23608),
        4: (6744, 11804),
        5: (3372, 5902),
        6: (1686, 2951),
        7: (843, 1476),
    }
    assert backend.level_downsamples == {
        0: (1.0, 1.0),
        1: (2.000018535337621, 2.0),
        2: (4.000111214087117, 4.0),
        3: (8.000519018313932, 8.00016943409014),
        4: (16.002224529141333, 16.00033886818028),
        5: (32.009196084247996, 32.00067773636056),
        6: (64.03738872403561, 64.00135547272112),
        7: (128.15083135391924, 128.04610169491525),
    }
    assert backend.level_count == 8
    assert backend.data_bounds == (0, 0, 188868, 107903)
    with pytest.warns():
        backend.read_region((0, 0, 100, 100), level=2)
    assert backend.read_level(-1).shape == (*backend.level_dimensions[7], 3)
    read_regions_with_reader(backend)
