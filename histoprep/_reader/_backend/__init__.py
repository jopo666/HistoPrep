# from ._bioformats import BioformatsBackend, BIOFORMATS_READABLE
from ._openslide import OPENSLIDE_READABLE, OpenSlideBackend
from ._pillow import PILLOW_READABLE, PillowBackend
from ._zeiss import ZEISS_READABLE, ZeissBackend

# Create readable formats
READABLE_FORMATS = tuple(set(PILLOW_READABLE + OPENSLIDE_READABLE + ZEISS_READABLE))
