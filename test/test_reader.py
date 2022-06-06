import os

import histoprep
import pytest
from PIL import Image

from paths import DATA_PATH

SLIDE_1 = os.path.join(DATA_PATH, "slide_1.svs")
SLIDE_2 = os.path.join(DATA_PATH, "slide_2.mrxs")
SLIDE_3 = os.path.join(DATA_PATH, "slide_3.czi")
SLIDE_4 = os.path.join(DATA_PATH, "tile.jpeg")
OUTPUT_DIR = os.path.join(DATA_PATH, "output")
SCRATCH_DIR = os.path.join(DATA_PATH, "scratch")


def test_bad_paths():
    with pytest.raises(FileNotFoundError):
        histoprep.SlideReader("i/dont/exist")
    with pytest.raises(IOError):
        histoprep.SlideReader(DATA_PATH)


def test_bad_region():
    reader = histoprep.SlideReader(os.path.join(DATA_PATH, "slide_1.svs"))
    for xywh in [
        (-1, 0, 10, 10),
        (0, -1, 10, 10),
        (0, 0, 0, 10),
        (0, 0, 10, -1),
    ]:
        with pytest.raises(ValueError):
            reader.read_region(xywh)


def test_out_of_bounds():
    reader = histoprep.SlideReader(os.path.join(DATA_PATH, "slide_1.svs"))
    with pytest.raises(ValueError):
        reader.read_region((reader.dimensions[1] + 10, 0, 10, 10), fill=None)


def test_get_thumbnail():
    reader = histoprep.SlideReader(os.path.join(DATA_PATH, "slide_1.svs"))
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


# def test_scratch():
#     reader = histoprep.SlideReader(SLIDE_1, scratch_dir=SCRATCH_DIR)
#     assert os.path.exists(os.path.join(SCRATCH_DIR, "slide_1.svs"))
#     shutil.rmtree(os.path.join(DATA_PATH, "scratch"))


def test_pillow_padding():
    reader = histoprep.SlideReader(SLIDE_4)
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


def test_properties():
    reader = histoprep.SlideReader(SLIDE_1)
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
    assert reader.path == SLIDE_1
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
    reader = histoprep.SlideReader(SLIDE_1)
    thresh, mask = reader.detect_tissue(threshold=0, multiplier=1.0)
    assert mask.sum() == 0
    assert thresh == reader.tissue_threshold
    thresh, mask = reader.detect_tissue(threshold=255, multiplier=1.0)
    assert mask.sum() == mask.size
    assert thresh == reader.tissue_threshold


def test_max_dimension():
    with pytest.raises(ValueError):
        histoprep.SlideReader(SLIDE_1, max_dimension=200)
    reader = histoprep.SlideReader(SLIDE_1)
    assert reader.MAX_DIMENSION == 2**14
    with pytest.raises(TypeError):
        reader.MAX_DIMENSION = "penis"
    with pytest.raises(ValueError):
        reader.MAX_DIMENSION = -1
    reader.MAX_DIMENSION = 4000
