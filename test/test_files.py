import os

from histoprep.helpers._files import get_extension, remove_extension

DIR_PATH = os.path.dirname(__file__)


def test_remove_extension():
    for src, dst in {
        "/data.p": "/data",
        "./data.p": "./data",
        "/d.a.t.a.p": "/d.a.t.a",
        "./pickle": "./pickle",
    }.items():
        assert dst == remove_extension(src)


def test_get_extension():
    for src, dst in {
        "/data.p": "p",
        "./data.p": "p",
        "/d.a.t.a.p": "p",
        "./pickle": None,
    }.items():
        assert dst == get_extension(src)


# def test_find_files():
#     assert len(find_files(DIR_PATH, ".py", depth=0)) > 0
#     assert len(find_files(DIR_PATH, "py", depth=0)) > 0
#     assert len(find_files(DIR_PATH, "py", depth=0)) < len(
#         find_files(DIR_PATH, "py", depth=1)
#     )
#     assert len(find_files(DIR_PATH, "ekfrhjweofgiuh", depth=100)) == 0
#     assert len(find_files(DIR_PATH, "jpeg", depth=0)) == 0
#     assert len(find_files(DIR_PATH, "jpeg", depth=1)) == 1
