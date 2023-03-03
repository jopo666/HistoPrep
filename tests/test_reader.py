from histoprep import SlideReader

from .utils import SLIDE_PATH_JPEG, TMP_DIRECTORY, clean_temporary_directory


def test_slidereader_methods():
    reader = SlideReader(SLIDE_PATH_JPEG)
    tissue_mask = reader.get_tissue_mask(sigma=2.0)
    tile_coords = reader.get_tile_coordinates(tissue_mask, 512)
    clean_temporary_directory()
    reader.save_tiles(parent_dir=TMP_DIRECTORY, coordinates=tile_coords)
    clean_temporary_directory()
