import os

import histoprep
import pytest
from PIL import Image

from paths import DATA_PATH

SLIDE_SVS = os.path.join(DATA_PATH, "slide_1.svs")
SLIDE_CZI = os.path.join(DATA_PATH, "slide_zeiss.czi")
SLIDE_JPEG = os.path.join(DATA_PATH, "tile.jpeg")
OUTPUT_DIR = os.path.join(DATA_PATH, "output")
SCRATCH_DIR = os.path.join(DATA_PATH, "scratch")


def test_bad_paths():
    with pytest.raises(FileNotFoundError):
        histoprep.SlideReader("i/dont/exist")
    with pytest.raises(IOError):
        histoprep.SlideReader(DATA_PATH)


def test_bad_region():
    reader = histoprep.SlideReader(SLIDE_SVS)
    for xywh in [
        (-1, 0, 10, 10),
        (0, -1, 10, 10),
        (0, 0, 0, 10),
        (0, 0, 10, -1),
    ]:
        with pytest.raises(ValueError):
            reader.read_region(xywh)


def test_out_of_bounds():
    reader = histoprep.SlideReader(SLIDE_SVS)
    with pytest.raises(ValueError):
        reader.read_region((reader.dimensions[1] + 10, 0, 10, 10), fill=None)


def test_get_thumbnail():
    reader = histoprep.SlideReader(SLIDE_SVS)
    for kwargs in [
        dict(preferred_dimension=None, level=None),
        dict(preferred_dimension=None, level=1000),
    ]:
        with pytest.raises(ValueError):
            reader.get_thumbnail(**kwargs)
    # Successes.
    assert reader.get_thumbnail(level=2).size == reader.level_dimensions[2][::-1]
    assert reader.get_thumbnail(2187).size == reader.level_dimensions[2][::-1]
    assert reader.get_thumbnail(1).size == reader.level_dimensions[2][::-1]


def test_pillow_padding():
    reader = histoprep.SlideReader(SLIDE_JPEG)
    assert reader.read_region((0, 128, 224, 224), return_arr=True).shape == (
        224,
        224,
        3,
    )
    assert reader.read_region((0, 128, 400, 224), return_arr=True).shape == (
        224,
        400,
        3,
    )
    assert reader.read_region((0, 128, 224, 400), return_arr=True).shape == (
        400,
        224,
        3,
    )


def test_read_region_above_zero_level():
    reader = histoprep.SlideReader(SLIDE_SVS)
    coords = reader.get_tile_coordinates(512, level=1)
    assert reader.read_region(coords[0], level=1, return_arr=True)[
        0, 0, :
    ].tolist() == [247, 246, 249]
    reader = histoprep.SlideReader(SLIDE_CZI)
    coords = reader.get_tile_coordinates(512, level=1, max_background=0.01)
    reader.read_region(coords[0], level=1)
    assert reader.read_region(coords[0], level=1, return_arr=True)[
        -1, -1, :
    ].tolist() == [188, 136, 196]
    reader = histoprep.SlideReader(SLIDE_JPEG)
    coords = reader.get_tile_coordinates(64, level=1, max_background=0.5)
    reader.read_region(coords[0], level=1)
    assert reader.read_region(coords[0], level=1, return_arr=True)[
        -1, -1, :
    ].tolist() == [203, 88, 181]


def test_properties():
    reader = histoprep.SlideReader(SLIDE_SVS)
    assert str(reader.backend) == "OPENSLIDE"
    assert reader.channel_order == "XYWH"
    assert reader.dimension_order == "HW"
    assert isinstance(reader.dimensions, tuple) and isinstance(
        reader.dimensions[0], int
    )
    assert isinstance(reader.level_dimensions, dict)
    assert reader.level_dimensions[0] == reader.dimensions
    assert isinstance(reader.level_downsamples, dict)
    assert reader.level_downsamples[0] == (1, 1)
    assert reader.path == SLIDE_SVS
    assert reader.slide_name == "slide_1"
    assert isinstance(reader.thumbnail, Image.Image)
    assert [int(x) for x in reader.thumbnail_downsample] == [8, 8]
    assert reader.tile_metadata is None
    assert isinstance(reader.tissue_mask, Image.Image)
    assert reader.tissue_mask.size == reader.thumbnail.size
    # Initialized properties with no data.
    assert reader.annotated_thumbnail_spots is None
    assert reader.annotated_thumbnail_tiles is None
    assert reader.spot_mask is None
    assert reader.spot_metadata is None


def test_tissue_detection():
    reader = histoprep.SlideReader(SLIDE_SVS)
    thresh, mask = reader.detect_tissue(threshold=0, multiplier=1.0)
    assert mask.sum() == 0
    assert thresh == reader.tissue_threshold
    thresh, mask = reader.detect_tissue(threshold=255, multiplier=1.0)
    assert mask.sum() == mask.size
    assert thresh == reader.tissue_threshold


def test_max_dimension():
    with pytest.raises(ValueError):
        histoprep.SlideReader(SLIDE_SVS, max_dimension=200)
    reader = histoprep.SlideReader(SLIDE_SVS)
    assert reader.MAX_DIMENSION == 2**14
    with pytest.raises(TypeError):
        reader.MAX_DIMENSION = "penis"
    with pytest.raises(ValueError):
        reader.MAX_DIMENSION = -1
    reader.MAX_DIMENSION = 4000
