import numpy as np
import pytest

import histoprep.functional as F
from histoprep import SlideReader
from tests._utils import IMAGE, SLIDE_PATH_TMA


def test_tissue_mask_otsu() -> None:
    thresh, mask = F.get_tissue_mask(IMAGE)
    assert mask.shape == IMAGE.shape[:2]
    assert thresh == 200
    assert mask.sum() == 184158


def test_tissue_mask_otsu_multiplier() -> None:
    thresh, mask = F.get_tissue_mask(IMAGE, multiplier=1.05)
    assert thresh == 210
    assert mask.sum() == 192803


def test_tissue_mask_threshold() -> None:
    thresh, mask = F.get_tissue_mask(IMAGE, threshold=210)
    assert thresh == 210
    assert mask.sum() == 192803


def test_tissue_mask_bad_threshold() -> None:
    with pytest.raises(ValueError, match="Threshold should be in range"):
        F.get_tissue_mask(IMAGE, threshold=500)


def test_clean_tissue_mask() -> None:
    image = SlideReader(SLIDE_PATH_TMA).read_level(-1)
    __, tissue_mask = F.get_tissue_mask(image, sigma=0.0)
    # We fill the areas.
    assert F.clean_tissue_mask(tissue_mask).sum() > tissue_mask.sum()


def test_clean_empty_mask() -> None:
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    assert F.clean_tissue_mask(empty_mask).sum() == 0
