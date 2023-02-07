import os

from PIL import Image

from histoprep import functional as F

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TILE_PATH = os.path.join(DATA_DIR, "tile.jpeg")
SLIDE_PATH_JPEG = os.path.join(DATA_DIR, "slide.jpg")

TILE_IMAGE = Image.open(TILE_PATH)
TILE_THRESHOLD, TILE_MASK = F.detect_tissue(TILE_IMAGE)
SLIDE_IMAGE = Image.open(SLIDE_PATH_JPEG)
SLIDE_THRESHOLD, SLIDE_MASK = F.detect_tissue(SLIDE_IMAGE)
