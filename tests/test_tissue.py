import pytest

import histoprep.functional as F
from tests.utils import TILE_IMAGE


def test_tissue_detection():
    # No arguments.
    thresh, mask = F.detect_tissue(TILE_IMAGE)
    assert thresh == 153
    assert mask.shape == TILE_IMAGE.size
    assert mask.sum() == 47643
    # Explicit threshold.
    thresh, mask = F.detect_tissue(TILE_IMAGE, threshold=100)
    assert thresh == 100
    assert mask.sum() == 23655
    assert F.detect_tissue(TILE_IMAGE, threshold=100, multiplier=20000)[0] == 100
    # Multiplier.
    thresh, mask = F.detect_tissue(TILE_IMAGE, multiplier=20000)
    assert thresh == 255
    assert mask.sum() == 256 * 256
    thresh, mask = F.detect_tissue(TILE_IMAGE, multiplier=0)
    assert thresh == 0
    assert mask.sum() == 0
    thresh, mask = F.detect_tissue(TILE_IMAGE, multiplier=1.1)
    assert thresh == 168
    assert mask.sum() == 50365
    # Errors.
    with pytest.raises(ValueError):
        F.detect_tissue(TILE_IMAGE, threshold=-1)
    with pytest.raises(ValueError):
        F.detect_tissue(TILE_IMAGE, multiplier=-1)
    with pytest.raises(ValueError):
        F.detect_tissue(TILE_IMAGE, sigma=-1)
