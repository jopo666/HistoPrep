import numpy as np

import histoprep.functional as F
from histoprep import SlideReader
from histoprep.utils import MachenkoStainNormalizer, VahadaneStainNormalizer

from ._utils import IMAGE, SLIDE_PATH_SVS

IMAGE_1 = IMAGE
IMAGE_2 = SlideReader(SLIDE_PATH_SVS).read_level(-1)[500:1000, 500:1000, :]


def test_macenko_stain_matrix() -> None:
    __, mask = F.get_tissue_mask(IMAGE_1)
    empty_mask = np.zeros_like(IMAGE_1)[..., 0]
    assert F.get_macenko_stain_matrix(IMAGE_1).shape == (2, 3)
    assert F.get_macenko_stain_matrix(IMAGE_1, mask).shape == (2, 3)
    assert F.get_macenko_stain_matrix(IMAGE_1, empty_mask).shape == (2, 3)


def test_vahadane_stain_matrix() -> None:
    __, mask = F.get_tissue_mask(IMAGE_1)
    empty_mask = np.zeros_like(IMAGE_1)[..., 0]
    assert F.get_vahadane_stain_matrix(IMAGE_1).shape == (2, 3)
    assert F.get_vahadane_stain_matrix(IMAGE_1, mask).shape == (2, 3)
    assert F.get_vahadane_stain_matrix(IMAGE_1, empty_mask).shape == (2, 3)


def test_macenko_normalizer_fit() -> None:
    norm = MachenkoStainNormalizer()
    norm.fit(IMAGE_2)
    assert norm.normalize(IMAGE_1).shape == IMAGE_1.shape


def test_macenko_normalizer_fit_with_mask() -> None:
    norm = MachenkoStainNormalizer()
    norm.fit(IMAGE_2, tissue_mask=F.get_tissue_mask(IMAGE_2)[1])
    assert norm.normalize(IMAGE_1).shape == IMAGE_1.shape


def test_macenko_normalizer_no_fit() -> None:
    norm = MachenkoStainNormalizer()
    assert norm.normalize(IMAGE_1).shape == IMAGE_1.shape


def test_vahadane_normalizer_fit() -> None:
    norm = VahadaneStainNormalizer()
    norm.fit(IMAGE_2)
    assert norm.normalize(IMAGE_1).shape == IMAGE_1.shape


def test_vahadane_normalizer_fit_with_mask() -> None:
    norm = VahadaneStainNormalizer()
    norm.fit(IMAGE_2, tissue_mask=F.get_tissue_mask(IMAGE_2)[1])
    assert norm.normalize(IMAGE_1).shape == IMAGE_1.shape


def test_vahadane_normalizer_no_fit() -> None:
    norm = VahadaneStainNormalizer()
    assert norm.normalize(IMAGE_1).shape == IMAGE_1.shape


def test_split_stains() -> None:
    haema, eosin = F.separate_stains(IMAGE_1, F.get_macenko_stain_matrix(IMAGE_1))
    assert haema.shape == IMAGE_1.shape
    assert eosin.shape == IMAGE_1.shape


def test_adjust_stains() -> None:
    adjusted = F.adjust_stains(
        IMAGE_1,
        F.get_macenko_stain_matrix(IMAGE_1),
        haematoxylin_magnitude=0.5,
        eosin_magnitude=1.2,
    )
    assert adjusted.shape == IMAGE_1.shape
