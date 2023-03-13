import numpy as np
from PIL import Image

import histoprep.functional as F
from histoprep.utils import MachenkoStainNormalizer, VahadaneStainNormalizer

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
    norm = MachenkoStainNormalizer()
    norm.normalize(IMAGE_2)
    norm.fit(IMAGE_1, F.get_tissue_mask(IMAGE_1)[1])
    assert norm.normalize(IMAGE_2).shape == IMAGE_2.shape
    assert norm.normalize(IMAGE_2, F.get_tissue_mask(IMAGE_2)[1]).shape == IMAGE_2.shape


def test_vahadane_stain_matrix() -> None:
    for img in [IMAGE_1, IMAGE_2]:
        __, mask = F.get_tissue_mask(img)
        empty_mask = np.zeros_like(img)[..., 0]
        assert F.get_vahadane_stain_matrix(img).shape == (2, 3)
        assert F.get_vahadane_stain_matrix(img, mask).shape == (2, 3)
        assert F.get_vahadane_stain_matrix(img, empty_mask).shape == (2, 3)

    norm = VahadaneStainNormalizer()
    norm.normalize(IMAGE_2)
    norm.fit(IMAGE_1, F.get_tissue_mask(IMAGE_1)[1])
    assert norm.normalize(IMAGE_2).shape == IMAGE_2.shape
    assert norm.normalize(IMAGE_2, F.get_tissue_mask(IMAGE_2)[1]).shape == IMAGE_2.shape
