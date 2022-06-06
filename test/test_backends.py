import os

from histoprep._reader._backend import OpenSlideBackend, PillowBackend, ZeissBackend

from paths import DATA_PATH


def test_pillow_backend():
    backend = PillowBackend(os.path.join(DATA_PATH, "tile.jpeg"))
    assert backend.dimensions == (256, 256)
    assert backend.level_dimensions == {0: (256, 256), 1: (128, 128)}
    assert backend.level_downsamples == {0: (1.0, 1.0), 1: (2.0, 2.0)}
    assert backend.read_region((200, 200, 1, 1), 0).tolist() == [[[194, 45, 129]]]
    assert backend.read_region((50, 50, 1, 1), 1).tolist() == [[[250, 228, 240]]]
    assert backend.get_thumbnail(0).shape == (256, 256, 3)
    assert backend.get_thumbnail(1).shape == (128, 128, 3)


# def test_bioformats_backend():
#     backend = BioformatsBackend(os.path.join(DATA_PATH, "slide_1.svs"))
#     assert backend.dimensions == (17497, 15374)
#     assert backend.level_dimensions == {
#         0: (17497, 15374),
#         1: (4374, 3843),
#         2: (2187, 1921),
#     }
#     assert backend.level_downsamples == {
#         0: (1.0, 1.0),
#         1: (4.000228623685413, 4.000520426749935),
#         2: (8.000457247370827, 8.003123373243103),
#     }
#     assert backend.read_region((200, 200, 1, 1)).tolist() == [[[245, 247, 244]]]
#     assert backend.get_thumbnail(2).shape == (2187, 1921, 3)


def test_openslide_backend():
    backend = OpenSlideBackend(os.path.join(DATA_PATH, "slide_1.svs"))
    assert backend.dimensions == (17497, 15374)
    assert backend.level_dimensions == {
        0: (17497, 15374),
        1: (4374, 3843),
        2: (2187, 1921),
    }
    assert backend.level_downsamples == {
        0: (1.0, 1.0),
        1: (4.000228623685413, 4.000520426749935),
        2: (8.000457247370827, 8.003123373243103),
    }
    assert backend.read_region((200, 200, 1, 1), 0).tolist() == [[[246, 248, 245]]]
    assert backend.read_region((200, 200, 1, 1), 1).tolist() == [[[243, 243, 241]]]
    assert backend.get_thumbnail(2).shape == (2187, 1921, 3)


def test_zeiss_backend():
    backend = ZeissBackend(os.path.join(DATA_PATH, "slide_zeiss.czi"))
    assert backend.dimensions == (134009, 148428)
    assert backend.level_dimensions == {
        0: (134009, 148428),
        1: (67004, 74214),
        2: (33502, 37107),
        3: (16751, 18553),
        4: (8375, 9276),
        5: (4187, 4638),
        6: (2093, 2319),
        7: (1046, 1159),
        8: (523, 579),
    }
    assert backend.level_downsamples == {
        0: (1.0, 1.0),
        1: (2.0000149244821204, 2.0),
        2: (4.000029848964241, 4.0),
        3: (8.000059697928481, 8.00021559855549),
        4: (16.001074626865673, 16.001293661060803),
        5: (32.0059708621925, 32.002587322121606),
        6: (64.02723363592929, 64.00517464424321),
        7: (128.11567877629062, 128.0655737704918),
        8: (256.23135755258124, 256.3523316062176),
    }
    assert backend.read_region((5_000, 50_000, 1, 1), 0).tolist() == [[[225, 210, 217]]]
    assert backend.read_region((25, 155, 1, 1), 8).tolist() == [[[255, 255, 255]]]
    assert backend.get_thumbnail(6).shape == (2096, 2320, 3)
