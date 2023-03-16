from __future__ import annotations

import pytest
from aicspylibczi import CziFile
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from PIL import Image, UnidentifiedImageError

from histoprep.backend import CziBackend, OpenSlideBackend, PillowBackend
from tests.utils import SLIDE_PATH_CZI, SLIDE_PATH_JPEG, SLIDE_PATH_MRXS, SLIDE_PATH_SVS


def read_zero_sized_region(
    backend: CziBackend | OpenSlideBackend | PillowBackend,
) -> None:
    assert backend.read_region((0, 0, 0, 0), 0).shape == (0, 0, 3)
    assert backend.read_region((0, 0, 1, 0), 0).shape == (0, 1, 3)
    if len(backend.level_dimensions) > 1:
        assert backend.read_region((0, 0, 1, 1), level=1).shape == (0, 0, 3)


def read_region_from_all_levels(
    backend: CziBackend | OpenSlideBackend | PillowBackend, tile_width: int = 256
) -> None:
    for level in backend.level_dimensions:
        tile_dims = backend.read_region(
            (0, 0, tile_width, tile_width), level=level
        ).shape
        expected_dims = (tile_width // 2**level, tile_width // 2**level, 3)
        assert tile_dims == expected_dims


def read_invalid_level(
    backend: CziBackend | OpenSlideBackend | PillowBackend,
) -> None:
    with pytest.raises(ValueError, match="Level 100 could not be found"):
        backend.read_region((0, 0, 10, 10), level=100)


def test_pillow_init() -> None:
    __ = PillowBackend(SLIDE_PATH_JPEG)
    with pytest.raises(UnidentifiedImageError):
        __ = PillowBackend(SLIDE_PATH_CZI)


def test_czi_init() -> None:
    __ = CziBackend(SLIDE_PATH_CZI)
    with pytest.raises(RuntimeError):
        __ = CziBackend(SLIDE_PATH_JPEG)


def test_openslide_init() -> None:
    __ = OpenSlideBackend(SLIDE_PATH_SVS)
    with pytest.raises(OpenSlideUnsupportedFormatError):
        __ = OpenSlideBackend(SLIDE_PATH_JPEG)


def test_zero_region_pillow() -> None:
    backend = PillowBackend(SLIDE_PATH_JPEG)
    read_zero_sized_region(backend)


def test_invalid_level_pillow() -> None:
    backend = PillowBackend(SLIDE_PATH_JPEG)
    read_invalid_level(backend)


def test_read_region_pillow() -> None:
    backend = PillowBackend(SLIDE_PATH_JPEG)
    read_region_from_all_levels(backend)


def test_read_level_pillow() -> None:
    backend = PillowBackend(SLIDE_PATH_JPEG)
    assert backend.read_level(-1).shape == (500, 1000, 3)


def test_zero_region_czi() -> None:
    backend = CziBackend(SLIDE_PATH_CZI)
    read_zero_sized_region(backend)


def test_invalid_level_czi() -> None:
    backend = CziBackend(SLIDE_PATH_CZI)
    read_invalid_level(backend)


def test_read_region_czi() -> None:
    backend = CziBackend(SLIDE_PATH_CZI)
    read_region_from_all_levels(backend)


def test_read_level_czi() -> None:
    backend = CziBackend(SLIDE_PATH_CZI)
    assert backend.read_level(-1).shape == (843, 1476, 3)


def test_zero_region_openslide() -> None:
    backend = OpenSlideBackend(SLIDE_PATH_MRXS)
    read_zero_sized_region(backend)


def test_invalid_level_openslide() -> None:
    backend = OpenSlideBackend(SLIDE_PATH_MRXS)
    read_invalid_level(backend)


def test_read_region_openslide() -> None:
    backend = OpenSlideBackend(SLIDE_PATH_MRXS)
    read_region_from_all_levels(backend)


def test_read_level_openslide() -> None:
    backend = OpenSlideBackend(SLIDE_PATH_MRXS)
    assert backend.read_level(-1).shape == (1603, 665, 3)


def test_properties_pillow() -> None:
    backend = PillowBackend(SLIDE_PATH_JPEG)
    assert backend.path == SLIDE_PATH_JPEG
    assert backend.name == "slide"
    assert backend.BACKEND_NAME == "PILLOW"
    assert backend.level_count == 2
    assert backend.dimensions == (1000, 2000)
    assert backend.level_dimensions == {0: (1000, 2000), 1: (500, 1000)}
    assert backend.level_downsamples == {0: (1.0, 1.0), 1: (2.0, 2.0)}
    assert backend.data_bounds == (0, 0, 2000, 1000)
    assert isinstance(backend.reader, Image.Image)


def test_properties_czi() -> None:
    backend = CziBackend(SLIDE_PATH_CZI)
    assert backend.path == SLIDE_PATH_CZI
    assert backend.name == "slide"
    assert backend.BACKEND_NAME == "CZI"
    assert backend.dimensions == (107903, 188868)
    assert backend.level_count == 8
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
    assert backend.data_bounds == (0, 0, 188868, 107903)
    assert isinstance(backend.reader, CziFile)


def test_openslide_properties() -> None:
    backend = OpenSlideBackend(SLIDE_PATH_MRXS)
    assert backend.dimensions == (410429, 170489)
    assert backend.level_count == 9
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
    assert backend.data_bounds == (21065, 179718, 94074, 410429)
    assert isinstance(backend.reader, OpenSlide)
