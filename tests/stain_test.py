import numpy as np
from PIL import Image

import histoprep.functional as F

from .utils import SLIDE_PATH_JPEG, TILE_IMAGE

IMAGE_1 = np.array(TILE_IMAGE)
IMAGE_2 = np.array(Image.open(SLIDE_PATH_JPEG))


def test_macenko_stain_matrix() -> None:
    for img in [IMAGE_1, IMAGE_2]:
        __, mask = F.get_tissue_mask(img)
        empty_mask = np.zeros_like(img)[..., 0]
        assert F.get_macenko_stain_matrix(img).shape == (2, 3)
        assert F.get_macenko_stain_matrix(img, mask).shape == (2, 3)
        assert F.get_macenko_stain_matrix(img, empty_mask).shape == (2, 3)


def test_vahadane_stain_matrix() -> None:
    for img in [IMAGE_1, IMAGE_2]:
        __, mask = F.get_tissue_mask(img)
        empty_mask = np.zeros_like(img)[..., 0]
        assert F.get_vahadane_stain_matrix(img).shape == (2, 3)
        assert F.get_vahadane_stain_matrix(img, mask).shape == (2, 3)
        assert F.get_vahadane_stain_matrix(img, empty_mask).shape == (2, 3)
