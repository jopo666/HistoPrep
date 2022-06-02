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
    assert backend.read_region((200, 200, 1, 1), 1).tolist() == [[[244, 244, 242]]]
    assert backend.get_thumbnail(2).shape == (2187, 1921, 3)


def test_zeiss_backend():
    backend = ZeissBackend(os.path.join(DATA_PATH, "slide_3.czi"))
    assert backend.dimensions == (75715, 167220)
    assert backend.level_dimensions == {
        0: (75715, 167220),
        1: (37857, 83610),
        2: (18928, 41805),
        3: (9464, 20902),
        4: (4732, 10451),
        5: (2366, 5225),
        6: (1183, 2612),
        7: (591, 1306),
        8: (295, 653),
    }
    assert backend.level_downsamples == {
        0: (1.0, 1.0),
        1: (2.0000264151940197, 2.0),
        2: (4.000158495350803, 4.0),
        3: (8.000316990701606, 8.000191369246963),
        4: (16.00063398140321, 16.000382738493926),
        5: (32.00126796280642, 32.00382775119617),
        6: (64.00253592561285, 64.0199081163859),
        7: (128.11336717428088, 128.0398162327718),
        8: (256.66101694915255, 256.0796324655436),
    }
    assert backend.read_region((5_000, 50_000, 1, 1), 0).tolist() == [[[179, 168, 198]]]
    assert backend.read_region((25, 155, 1, 1), 8).tolist() == [[[206, 196, 205]]]
    assert backend.get_thumbnail(6).shape == (1183, 2613, 3)
