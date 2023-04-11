import warnings

import pytest

import histoprep as hp
import histoprep.functional as F
from tests._utils import SLIDE_PATH_TMA

IMAGE = hp.SlideReader(SLIDE_PATH_TMA).read_level(-1)


def test_dearray_good_mask() -> None:
    __, tissue_mask = F.get_tissue_mask(IMAGE, sigma=2)
    tissue_mask_clean = F.clean_tissue_mask(tissue_mask)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spots = F.get_spot_coordinates(tissue_mask_clean)
    assert len(spots) == 94


def test_dearray_bad_mask() -> None:
    __, tissue_mask = F.get_tissue_mask(IMAGE, sigma=0.0)
    with pytest.warns():
        F.get_spot_coordinates(tissue_mask)


def test_dearray_empty_mask() -> None:
    __, tissue_mask = F.get_tissue_mask(IMAGE)
    tissue_mask[...] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spots = F.get_spot_coordinates(tissue_mask)
    assert len(spots) == 0
