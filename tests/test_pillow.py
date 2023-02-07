from histoprep import ReaderBackend, SlideReader

from .utils import SLIDE_PATH_JPEG


def test_pillow_jpeg() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG, backend=ReaderBackend.PILLOW)
    # Check dimensions and downsamples.
    assert reader.dimensions == (1000, 2000)
    assert reader.level_dimensions == {
        0: (1000, 2000),
        1: (500, 1000),
        2: (250, 500),
    }
    assert reader.level_downsamples == {
        0: (1.0, 1.0),
        1: (2.0, 2.0),
        2: (4.0, 4.0),
    }
    # Check get level.
    for level, dimensions in reader.level_dimensions.items():
        assert reader.get_level(level).size[::-1] == dimensions
    # Check read region.
    for kwargs in reader.tile_coordinates(500):
        tile = reader.read_region(kwargs["xywh"], level=0)
        assert tile.shape == (500, 500, 3)
