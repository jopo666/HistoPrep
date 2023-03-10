from __future__ import annotations

import functools
import glob
from pathlib import Path

import mpire
import rich_click as click

from histoprep import SlideReader

from ._utils import error, info, warning

LOGO = """
██╗  ██╗██╗███████╗████████╗ ██████╗ ██████╗ ██████╗ ███████╗██████╗
██║  ██║██║██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗
███████║██║███████╗   ██║   ██║   ██║██████╔╝██████╔╝█████╗  ██████╔╝
██╔══██║██║╚════██║   ██║   ██║   ██║██╔═══╝ ██╔══██╗██╔══╝  ██╔═══╝
██║  ██║██║███████║   ██║   ╚██████╔╝██║     ██║  ██║███████╗██║
╚═╝  ╚═╝╚═╝╚══════╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝
                        by jopo666 (2023)
"""
# Rich-click options.
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.RANGE_STRING = ""
click.rich_click.HEADER_TEXT = LOGO
click.rich_click.STYLE_HEADER_TEXT = "dim"
click.rich_click.MAX_WIDTH = 120
click.rich_click.SHOW_METAVARS_COLUMN = False
click.rich_click.APPEND_METAVARS_HELP = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_CLICK_SHORT_HELP = False
IO_OPTIONS = ["--input", "--output", "--backend"]
TILE_OPTIONS = [
    "--level",
    "--width",
    "--height",
    "--overlap",
    "--max-background",
    "--out-of-bounds",
]
SAVE_OPTIONS = [
    "--save-paths",
    "--save-metrics",
    "--save-masks",
    "--overwrite",
    "--unfinished",
    "--format",
    "--quality",
    "--num-workers",
]
TISSUE_OPTIONS = [
    "--threshold",
    "--multiplier",
    "--ignore-white",
    "--ignore-black",
    "--tissue-level",
    "--max-dimension",
    "--sigma",
]
click.rich_click.OPTION_GROUPS = {
    "HistoPrep": [
        {"name": "Input/output", "options": IO_OPTIONS},
        {"name": "Tile extraction", "options": TILE_OPTIONS},
        {"name": "Tile saving", "options": SAVE_OPTIONS},
        {"name": "Tissue detection", "options": TISSUE_OPTIONS},
    ]
}


def glob_pattern(*args) -> list[Path]:
    pattern = args[-1]
    output = [
        z for z in (Path(x) for x in glob.glob(pattern, recursive=True)) if z.is_file()
    ]
    if len(output) == 0:
        error(f"Found no files matching pattern '{pattern}'.")
    info(f"Found {len(output)} files matching pattern '{pattern}'.")
    return output


@click.command()
# Required.
@click.option(  # input
    "-i",
    "--input",
    "paths",
    callback=glob_pattern,
    metavar="PATTERN",
    required=True,
    type=click.STRING,
    help="File pattern to glob.",
)
@click.option(  # output
    "-o",
    "--output",
    "parent_dir",
    metavar="DIRECTORY",
    required=True,
    callback=lambda *args: Path(args[-1]),
    type=click.Path(file_okay=False),
    help="Parent directory for all outputs.",
)
@click.option(  # backend
    "-b",
    "--backend",
    type=click.Choice(choices=["PIL", "OPENSLIDE", "CZI"], case_sensitive=False),
    default=None,
    show_default="automatic",
    help="Backend for reading slides.",
)
# Tiles.
@click.option(  # level
    "-l",
    "--level",
    metavar="INT",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Slide level for tile extraction.",
)
@click.option(  # width
    "-w",
    "--width",
    metavar="INT",
    type=click.IntRange(min=0, min_open=False),
    default=640,
    show_default=True,
    help="Tile width.",
)
@click.option(  # height
    "--height",
    metavar="INT",
    type=click.IntRange(min=0, min_open=False),
    default=None,
    show_default="width",
    help="Tile height.",
)
@click.option(  # overlap
    "-n",
    "--overlap",
    metavar="FLOAT",
    type=click.FloatRange(min=0, max=1, min_open=False, max_open=False),
    default=0.0,
    show_default=True,
    help="Overlap between neighbouring tiles.",
)
@click.option(  # background
    "-m",
    "--max-background",
    metavar="FLOAT",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    default=0.75,
    show_default=True,
    help="Maximum background in tile.",
)
@click.option(  # out-of-bounds
    "--out-of-bounds",
    type=click.BOOL,
    default=True,
    show_default=True,
    help="Allow tiles to go out-of-bounds. ",
)
# Saving.
@click.option(  # save_paths
    "--save-paths",
    type=click.BOOL,
    default=True,
    show_default=True,
    help="Save filepaths to metadata.",
)
@click.option(  # save_metrics
    "--save-metrics",
    type=click.BOOL,
    default=True,
    show_default=True,
    help="Save image metrics to metadata.",
)
@click.option(  # save_masks
    "--save-masks",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Save tissue masks for each tile.",
)
@click.option(  # overwrite
    "-z",
    "--overwrite",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Overwrite any existing slide outputs.",
)
@click.option(  # overwrite_unfinished
    "-u",
    "--unfinished",
    "overwrite_unfinished",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Overwrite only if metadata is missing.",
)
@click.option(  # format
    "--format",
    "image_format",
    type=click.STRING,
    default="jpeg",
    show_default=True,
    help="File format for tiles.",
)
@click.option(  # quality
    "--quality",
    metavar="INT",
    type=click.IntRange(min=0, max=100),
    default=80,
    show_default=True,
    help="Quality for jpeg-compression.",
)
@click.option(  # overwrite_unfinished
    "-c",
    "--use_csv",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Write metadata to csv-files instead of parquet files.",
)
@click.option(  # num_workers
    "-j",
    "--num-workers",
    metavar="INT",
    type=click.IntRange(min=0, min_open=False),
    default=None,
    show_default="CPU-count",
    help="Number of data saving workers.",
)
# Tissue.
@click.option(  # threshold
    "-t",
    "--threshold",
    metavar="INT",
    type=click.IntRange(min=0, max=255, min_open=False),
    default=None,
    show_default="Otsu's threshold",
    help="Global threshold value for tissue detection.",
)
@click.option(  # multiplier
    "-x",
    "--multiplier",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=False),
    default=1.05,
    show_default=True,
    help="Multiplier for Otsu's threshold. Ignored if threshold is set.",
)
@click.option(  # tissue_level
    "--tissue-level",
    metavar="INT",
    type=click.IntRange(min=0),
    default=None,
    show_default="max_dimension",
    help="Slide level for tissue detection.",
)
@click.option(  # max_dimension
    "--max-dimension",
    metavar="INT",
    type=click.IntRange(min=0),
    default=8192,
    show_default=True,
    help="Select first level with dimensions smaller than this value.",
)
@click.option(  # sigma
    "--sigma",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
    default=1.0,
    show_default=True,
    help="Sigma for gaussian blurring during tissue detection.",
)
def cut_slides(
    paths: list[str | Path],
    parent_dir: str | Path,
    *,
    backend: str | None = None,
    # Tissue detection.
    threshold: float | None = None,
    multiplier: float = 1.05,
    tissue_level: int | None = None,
    sigma: float = 1.0,
    max_dimension: int = 8192,
    # Tile extraction.
    level: int = 0,
    width: int = 640,
    height: int | None = None,
    overlap: float = 0.0,
    max_background: float = 0.75,
    out_of_bounds: bool = True,
    # Tile saving.
    save_paths: bool = True,
    save_metrics: bool = True,
    save_masks: bool = True,
    overwrite: bool = False,
    overwrite_unfinished: bool = False,
    image_format: str = "jpeg",
    quality: int = 80,
    use_csv: bool = False,
    num_workers: int | None = None,
) -> None:
    """CLI interface to extract tile images from slides."""
    # Filter slide paths.
    paths = filter_slide_paths(
        all_paths=[x if isinstance(x, Path) else Path(x) for x in paths],
        parent_dir=parent_dir,
        overwrite=overwrite,
        overwrite_unfinished=overwrite_unfinished,
    )
    # Define cut_slide kwargs.
    kwargs = {
        "reader_kwargs": {"backend": backend},
        "tissue_kwargs": {
            "level": tissue_level,
            "max_dimension": max_dimension,
            "threshold": threshold,
            "multiplier": multiplier,
            "sigma": sigma,
        },
        "tile_kwargs": {
            "width": width,
            "height": height,
            "overlap": overlap,
            "out_of_bounds": out_of_bounds,
            "max_background": max_background,
        },
        "save_kwargs": {
            "parent_dir": parent_dir,
            "level": level,
            "save_paths": save_paths,
            "save_metrics": save_metrics,
            "save_masks": save_masks,
            "image_format": image_format,
            "quality": quality,
            "use_csv": use_csv,
            "raise_exception": False,  # handled here.
            "num_workers": 0,  # slide-per-process.
            "overwrite": True,  # filtered earlier.
            "verbose": False,  # common progressbar.
        },
    }
    # Process.
    with mpire.WorkerPool(n_jobs=num_workers) as pool:
        for path, exception in pool.imap(
            func=functools.partial(cut_slide, **kwargs),
            iterable_of_args=paths,
            progress_bar=True,
            progress_bar_options={"desc": "Cutting slides"},
        ):
            if isinstance(exception, Exception):
                error(f"Could not process {path} due to exception: {exception}")


def filter_slide_paths(
    *,
    all_paths: list[Path],
    parent_dir: Path,
    overwrite: bool,
    overwrite_unfinished: bool,
) -> list[Path]:
    # Get processed and unprocessed slides.
    output, processed, interrupted = ([], [], [])
    for path in all_paths:
        if not path.is_file():
            continue
        output_dir = parent_dir / path.name.rstrip(path.suffix)
        if output_dir.exists():
            if (output_dir / "metadata.parquet").exists():
                processed.append(path)
            else:
                interrupted.append(path)
        else:
            output.append(path)
    # Add processed/unfinished to output.
    if overwrite:
        output += processed + interrupted
        if len(processed + interrupted) > 0:
            warning(f"Overwriting {len(interrupted)} slide outputs.")
    elif overwrite_unfinished:
        output += interrupted
        if len(interrupted) > 0:
            warning(f"Overwriting {len(interrupted)} unfinished slide outputs.")
    # Verbose.
    if len(output) == 0:
        error("No slides to process.")
    info(f"Processing {len(output)} slides.")
    return output


def cut_slide(
    path: Path,
    *,
    reader_kwargs: dict,
    tissue_kwargs: dict,
    tile_kwargs: dict,
    save_kwargs: dict,
) -> tuple[Path, Exception | None]:
    try:
        reader = SlideReader(path, **reader_kwargs)
        tissue_mask = reader.get_tissue_mask(**tissue_kwargs)
        coords = reader.get_tile_coordinates(tissue_mask=tissue_mask, **tile_kwargs)
        reader.save_tiles(coordinates=coords, **save_kwargs)
    except Exception as e:  # noqa
        return path, e
    return path, None
