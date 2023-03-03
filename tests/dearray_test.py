import warnings

import pytest

import histoprep as hp
import histoprep.functional as F
from tests.utils import SLIDE_PATH_TMA


def test_dearray() -> None:
    reader = hp.SlideReader(SLIDE_PATH_TMA)
    # Good sigma value.
    tissue_mask = reader.get_tissue_mask(sigma=2).mask
    tissue_mask_clean = F.clean_tissue_mask(tissue_mask)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spots = F.dearray_tma(tissue_mask_clean)
    assert len(spots) == 38
    # Bad sigma value.
    tissue_mask = reader.get_tissue_mask(sigma=0).mask
    tissue_mask_clean = F.clean_tissue_mask(tissue_mask)
    with pytest.warns():
        spots = F.dearray_tma(tissue_mask_clean)
    # Bad sigma, good cleaning
    tissue_mask_clean = F.clean_tissue_mask(tissue_mask, min_area_pixel=200)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spots = F.dearray_tma(tissue_mask_clean)
    assert len(spots) == 38
