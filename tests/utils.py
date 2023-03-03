import os
import shutil
from pathlib import Path

from PIL import Image

from histoprep import functional as F

DATA_DIRECTORY = Path(__file__).parent / "data"
TMP_DIRECTORY = DATA_DIRECTORY.parent / "tmp"
TILE_PATH = DATA_DIRECTORY / "tile.jpeg"
SLIDE_PATH_JPEG = DATA_DIRECTORY / "slide.jpeg"
SLIDE_PATH_MRXS = DATA_DIRECTORY / "slide_tma.mrxs"
SLIDE_PATH_SVS = DATA_DIRECTORY / "slide.svs"
SLIDE_PATH_CZI = DATA_DIRECTORY / "slide.czi"
SLIDE_PATH_TMA = SLIDE_PATH_MRXS

TILE_IMAGE = Image.open(TILE_PATH)
TILE_THRESHOLD, TILE_MASK = F.get_tissue_mask(TILE_IMAGE)
SLIDE_IMAGE = Image.open(SLIDE_PATH_JPEG)
SLIDE_THRESHOLD, SLIDE_MASK = F.get_tissue_mask(SLIDE_IMAGE)


def clean_temporary_directory():
    if TMP_DIRECTORY.exists():
        shutil.rmtree(TMP_DIRECTORY)
