from __future__ import annotations

import pytest
from aicspylibczi import CziFile
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from PIL import Image, UnidentifiedImageError

from histoprep._backend import CziBackend, OpenSlideBackend, PillowBackend
from tests._utils import (
    SLIDE_PATH_CZI,
    SLIDE_PATH_JPEG,
    SLIDE_PATH_SVS,
)


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
        h_d, w_d = backend.level_downsamples[level]
        expected_dims = (round(tile_width / h_d), round(tile_width / w_d), 3)
        assert tile_dims == expected_dims


def read_invalid_level(
    backend: CziBackend | OpenSlideBackend | PillowBackend,
) -> None:
    with pytest.raises(ValueError, match="Level 100 could not be found"):
        backend.read_region((0, 0, 10, 10), level=100)
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
    assert backend.read_level(-1).shape == (625, 625, 3)


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
    assert backend.read_level(-1).shape == (1047, 1160, 3)


def test_zero_region_openslide() -> None:
    backend = OpenSlideBackend(SLIDE_PATH_SVS)
    read_zero_sized_region(backend)


def test_invalid_level_openslide() -> None:
    backend = OpenSlideBackend(SLIDE_PATH_SVS)
    read_invalid_level(backend)


def test_read_region_openslide() -> None:
    backend = OpenSlideBackend(SLIDE_PATH_SVS)
    read_region_from_all_levels(backend)


def test_read_level_openslide() -> None:
    backend = OpenSlideBackend(SLIDE_PATH_SVS)
    assert backend.read_level(-1).shape == (1867, 1904, 3)


def test_properties_pillow() -> None:
    backend = PillowBackend(SLIDE_PATH_JPEG)
    assert backend.path == str(SLIDE_PATH_JPEG)
    assert backend.name == "slide"
    assert backend.suffix == ".jpeg"
    assert backend.BACKEND_NAME == "PILLOW"
    assert backend.level_count == 3
    assert backend.dimensions == (2500, 2500)
    assert backend.level_dimensions == {0: (2500, 2500), 1: (1250, 1250), 2: (625, 625)}
    assert backend.level_downsamples == {0: (1.0, 1.0), 1: (2.0, 2.0), 2: (4.0, 4.0)}
    assert backend.data_bounds == (0, 0, 2500, 2500)
    assert isinstance(backend.reader, Image.Image)


def test_properties_czi() -> None:
    backend = CziBackend(SLIDE_PATH_CZI)
    assert backend.path == str(SLIDE_PATH_CZI)
    assert backend.name == "slide"
    assert backend.suffix == ".czi"
    assert backend.BACKEND_NAME == "CZI"
    assert backend.dimensions == (134009, 148428)
    assert backend.level_count == 8
    assert backend.level_dimensions == {
        0: (134009, 148428),
        1: (67004, 74214),
        2: (33502, 37107),
        3: (16751, 18554),
        4: (8376, 9277),
        5: (4188, 4638),
        6: (2094, 2319),
        7: (1047, 1160),
    }
    assert backend.level_downsamples == {
        0: (1.0, 1.0),
        1: (2.0000149244821204, 2.0),
        2: (4.000029848964241, 4.0),
        3: (8.000059697928481, 8.00021559855549),
        4: (16.001074626865673, 16.001293661060803),
        5: (32.0059708621925, 32.002587322121606),
        6: (64.02723363592929, 64.00517464424321),
        7: (128.11567877629062, 128.0655737704918),
    }
    assert backend.data_bounds == (0, 0, 148428, 134009)
    assert isinstance(backend.reader, CziFile)


def test_openslide_properties() -> None:
    backend = OpenSlideBackend(SLIDE_PATH_SVS)
    assert backend.path == str(SLIDE_PATH_SVS)
    assert backend.name == "slide"
    assert backend.suffix == ".svs"
    assert backend.BACKEND_NAME == "OPENSLIDE"
    assert backend.level_count == 3
    assert backend.dimensions == (29875, 30464)
    assert backend.level_dimensions == {
        0: (29875, 30464),
        1: (7468, 7616),
        2: (1867, 1904),
    }
    assert backend.level_downsamples == {
        0: (1.0, 1.0),
        1: (4.000401713979646, 4.0),
        2: (16.001606855918585, 16.0),
    }
    assert backend.data_bounds == (0, 0, 30464, 29875)
    assert isinstance(backend.reader, OpenSlide)
