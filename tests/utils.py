import os
from pathlib import Path

from PIL import Image

from histoprep import functional as F

DATA_DIR = Path(__file__).parent / "data"
TILE_PATH = DATA_DIR / "tile.jpeg"
SLIDE_PATH_JPEG = DATA_DIR / "slide.jpeg"
SLIDE_PATH_MRXS = DATA_DIR / "tma_slide.mrxs"
SLIDE_PATH_SVS = DATA_DIR / "slide.svs"
SLIDE_PATH_CZI = DATA_DIR / "slide.czi"
SLIDE_PATH_TMA = SLIDE_PATH_MRXS

TILE_IMAGE = Image.open(TILE_PATH)
TILE_THRESHOLD, TILE_MASK = F.detect_tissue(TILE_IMAGE)
SLIDE_IMAGE = Image.open(SLIDE_PATH_JPEG)
SLIDE_THRESHOLD, SLIDE_MASK = F.detect_tissue(SLIDE_IMAGE)
