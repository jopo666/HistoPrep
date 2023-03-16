from __future__ import annotations

from ._utils import (
    SLIDE_PATH_JPEG,
    SLIDE_PATH_SVS,
    TMP_DIRECTORY,
    clean_temporary_directory,
)


def create_metadata(unfinished: bool = False) -> None:  # noqa
    meta_path = TMP_DIRECTORY / "slide" / "metadata.parquet"
    meta_path.parent.mkdir(parents=True)
    if not unfinished:
        meta_path.touch()


def test_run(script_runner) -> None:  # noqa
    clean_temporary_directory()
    ret = script_runner.run(
        *f"poetry run HistoPrep -i {SLIDE_PATH_JPEG} -o {TMP_DIRECTORY}".split(" ")
    )
    assert ret.success
    assert [x.name for x in (TMP_DIRECTORY / "slide").iterdir()] == [
        "properties.json",
        "tiles",
        "metadata.parquet",
    ]
    assert ret.stdout.split("\n")[1:] == ["INFO: Processing 1 slides.", ""]
    clean_temporary_directory()


def test_skip_processed(script_runner) -> None:  # noqa
    clean_temporary_directory()
    create_metadata(unfinished=False)
    ret = script_runner.run(
        *f"poetry run HistoPrep -i {SLIDE_PATH_JPEG} -o {TMP_DIRECTORY}".split(" ")
    )
    assert not ret.success
    assert ret.stdout.split("\n")[1:] == [
        "INFO: Skipping 1 processed slides.",
        "ERROR: No slides to process.",
        "",
    ]
    clean_temporary_directory()


def test_overwrite(script_runner) -> None:  # noqa
    clean_temporary_directory()
    create_metadata(unfinished=False)
    ret = script_runner.run(
        *f"poetry run HistoPrep -i {SLIDE_PATH_JPEG} -o {TMP_DIRECTORY} -z true".split(
            " "
        )
    )
    assert ret.success
    assert [x.name for x in (TMP_DIRECTORY / "slide").iterdir()] == [
        "properties.json",
        "tiles",
        "metadata.parquet",
    ]
    assert ret.stdout.split("\n")[1:] == [
        "WARNING: Overwriting 1 slide outputs.",
        "INFO: Processing 1 slides.",
        "",
    ]
    clean_temporary_directory()


def test_unfinished(script_runner) -> None:  # noqa
    clean_temporary_directory()
    create_metadata(unfinished=True)
    ret = script_runner.run(
        *f"poetry run HistoPrep -i {SLIDE_PATH_JPEG} -o {TMP_DIRECTORY} -u true".split(
            " "
        )
    )
    assert ret.success
    assert [x.name for x in (TMP_DIRECTORY / "slide").iterdir()] == [
        "properties.json",
        "tiles",
        "metadata.parquet",
    ]
    assert ret.stdout.split("\n")[1:] == [
        "WARNING: Overwriting 1 unfinished slide outputs.",
        "INFO: Processing 1 slides.",
        "",
    ]
    clean_temporary_directory()
