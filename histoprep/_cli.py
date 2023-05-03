"""CLI interface for cutting slides into small tile images."""

__all__ = ["cut_slides"]

import functools
import glob
import sys
from pathlib import Path
from typing import NoReturn, Optional, Union

import mpire
import rich_click as click

from histoprep import SlideReader

# LOGO = """
# ██╗  ██╗██╗███████╗████████╗ ██████╗ ██████╗ ██████╗ ███████╗██████╗
# ██║  ██║██║██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗
# ███████║██║███████╗   ██║   ██║   ██║██████╔╝██████╔╝█████╗  ██████╔╝
# ██╔══██║██║╚════██║   ██║   ██║   ██║██╔═══╝ ██╔══██╗██╔══╝  ██╔═══╝
# ██║  ██║██║███████║   ██║   ╚██████╔╝██║     ██║  ██║███████╗██║
# ╚═╝  ╚═╝╚═╝╚══════╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝
#                         by jopo666 (2023)
# """
# Rich-click options.
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.RANGE_STRING = ""
# click.rich_click.HEADER_TEXT = LOGO
click.rich_click.STYLE_HEADER_TEXT = "dim"
click.rich_click.MAX_WIDTH = 100
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
    "--in-bounds",
]
SAVE_OPTIONS = [
    "--metrics",
    "--masks",
    "--overwrite",
    "--unfinished",
    "--image-format",
    "--quality",
    "--num-workers",
]
TISSUE_OPTIONS = [
    "--threshold",
    "--multiplier",
    "--tissue-level",
    "--max-dimension",
    "--sigma",
]
click.rich_click.OPTION_GROUPS = {
    "HistoPrep": [
        {"name": "Input/output", "options": IO_OPTIONS},
        {"name": "Tile extraction", "options": TILE_OPTIONS},
        {"name": "Tissue detection", "options": TISSUE_OPTIONS},
        {"name": "Tile saving", "options": SAVE_OPTIONS},
    ]
}

DEFAULT_OPTIONS = {
    # Input/output
    "backend": None,
    # Tile extraction.
    "level": 0,
    "width": 640,
    "height": None,
    "overlap": 0.0,
    "max_background": 0.75,
    "in_bounds": False,
    # Tissue detection.
    "threshold": None,
    "multiplier": 1.05,
    "tissue_level": None,
    "max_dimension": 8192,
    "sigma": 1.0,
    "save_metrics": False,
    "save_masks": False,
    "overwrite": False,
    "overwrite_unfinished": False,
    "image_format": "jpeg",
    "quality": 80,
    "num_workers": None,
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
    "--backend",
    type=click.Choice(choices=["PIL", "OPENSLIDE", "CZI"], case_sensitive=False),
    default=DEFAULT_OPTIONS["backend"],
    show_default="automatic",
    help="Backend for reading slides.",
)
# Tiles.
@click.option(  # level
    "-l",
    "--level",
    metavar="INT",
    type=click.IntRange(min=0),
    default=DEFAULT_OPTIONS["level"],
    show_default=True,
    help="Pyramid level for tile extraction.",
)
@click.option(  # width
    "-w",
    "--width",
    metavar="INT",
    type=click.IntRange(min=0, min_open=False),
    default=DEFAULT_OPTIONS["width"],
    show_default=True,
    help="Tile width.",
)
@click.option(  # height
    "-h",
    "--height",
    metavar="INT",
    type=click.IntRange(min=0, min_open=False),
    default=DEFAULT_OPTIONS["height"],
    show_default="width",
    help="Tile height.",
)
@click.option(  # overlap
    "-n",
    "--overlap",
    metavar="FLOAT",
    type=click.FloatRange(min=0, max=1, min_open=False, max_open=False),
    default=DEFAULT_OPTIONS["overlap"],
    show_default=True,
    help="Overlap between neighbouring tiles.",
)
@click.option(  # background
    "-b",
    "--max-background",
    metavar="FLOAT",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    default=DEFAULT_OPTIONS["max_background"],
    show_default=True,
    help="Maximum background for tiles.",
)
@click.option(  # out-of-bounds
    "--in-bounds",
    show_default="False",
    is_flag=True,
    help="Do not allow tiles to go out-of-bounds. ",
)
# Tissue.
@click.option(  # threshold
    "-t",
    "--threshold",
    metavar="INT",
    type=click.IntRange(min=0, max=255, min_open=False),
    default=DEFAULT_OPTIONS["threshold"],
    show_default="Otsu",
    help="Global thresholding value.",
)
@click.option(  # multiplier
    "-x",
    "--multiplier",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=False),
    default=DEFAULT_OPTIONS["multiplier"],
    show_default=True,
    help="Multiplier for Otsu's threshold.",
)
@click.option(  # tissue_level
    "--tissue-level",
    metavar="INT",
    type=click.IntRange(min=0),
    default=DEFAULT_OPTIONS["tissue_level"],
    show_default="max_dimension",
    help="Pyramid level for tissue detection.",
)
@click.option(  # max_dimension
    "--max-dimension",
    metavar="INT",
    type=click.IntRange(min=0),
    default=DEFAULT_OPTIONS["max_dimension"],
    show_default=True,
    help="Maximum dimension for tissue detection.",
)
@click.option(  # sigma
    "--sigma",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
    default=DEFAULT_OPTIONS["sigma"],
    show_default=True,
    help="Sigma for gaussian blurring.",
)
# Saving.
@click.option(  # save_metrics
    "--metrics",
    "save_metrics",
    show_default="False",
    is_flag=True,
    help="Save image metrics.",
)
@click.option(  # save_masks
    "--masks",
    "save_masks",
    show_default="False",
    is_flag=True,
    help="Save tissue masks.",
)
@click.option(  # overwrite
    "-z",
    "--overwrite",
    show_default="False",
    is_flag=True,
    help="Overwrite any existing slide outputs.",
)
@click.option(  # overwrite_unfinished
    "-u",
    "--unfinished",
    "overwrite_unfinished",
    show_default="False",
    is_flag=True,
    help="Overwrite only if metadata is missing.",
)
@click.option(  # format
    "--image-format",
    type=click.STRING,
    default=DEFAULT_OPTIONS["image_format"],
    show_default=True,
    help="File format for tile images.",
)
@click.option(  # quality
    "--quality",
    metavar="INT",
    type=click.IntRange(min=0, max=100),
    default=DEFAULT_OPTIONS["quality"],
    show_default=True,
    help="Quality for jpeg-compression.",
)
@click.option(  # num_workers
    "-j",
    "--num-workers",
    metavar="INT",
    type=click.IntRange(min=0, min_open=False),
    default=DEFAULT_OPTIONS["num_workers"],
    show_default="CPU-count",
    help="Number of data saving workers.",
)
def cut_slides(
    paths: list[Union[str, Path]],
    parent_dir: Union[str, Path],
    *,
    backend: Optional[str] = None,
    # Tissue detection.
    threshold: Optional[float] = None,
    multiplier: float = 1.05,
    tissue_level: Optional[int] = None,
    max_dimension: int = 8192,
    sigma: float = 1.0,
    # Tile extraction.
    level: int = 0,
    width: int = 640,
    height: Optional[int] = None,
    overlap: float = 0.0,
    max_background: float = 0.75,
    in_bounds: bool = False,
    # Tile saving.
    save_metrics: bool = False,
    save_masks: bool = False,
    overwrite: bool = False,
    overwrite_unfinished: bool = False,
    image_format: str = "jpeg",
    quality: int = 80,
    use_csv: bool = False,
    num_workers: Optional[int] = None,
) -> None:
    """Extract tile images from histological slides."""
    paths = filter_slide_paths(
        all_paths=[x if isinstance(x, Path) else Path(x) for x in paths],
        parent_dir=parent_dir,
        overwrite=overwrite,
        overwrite_unfinished=overwrite_unfinished,
    )
    # Define cut_slide kwargs.
    kwargs = {
        "reader_kwargs": {"backend": backend},
        "max_dimension": max_dimension,
        "tissue_kwargs": {
            "level": tissue_level,
            "threshold": threshold,
            "multiplier": multiplier,
            "sigma": sigma,
        },
        "tile_kwargs": {
            "width": width,
            "height": height,
            "overlap": overlap,
            "out_of_bounds": not in_bounds,
            "max_background": max_background,
        },
        "save_kwargs": {
            "parent_dir": parent_dir,
            "level": level,
            "save_metrics": save_metrics,
            "save_masks": save_masks,
            "save_thumbnails": True,
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
                warning(
                    f"Could not process {path} due to exception: {exception.__repr__()}"
                )


def filter_slide_paths(  # noqa
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
        output_dir = parent_dir / path.name.removesuffix(path.suffix)
        if output_dir.exists():
            if (output_dir / "metadata.parquet").exists() or (
                output_dir / "metadata.csv"
            ).exists():
                processed.append(path)
            else:
                interrupted.append(path)
        else:
            output.append(path)
    # Add processed/unfinished to output.
    if overwrite:
        output += processed + interrupted
        if len(processed + interrupted) > 0:
            warning(f"Overwriting {len(processed + interrupted)} slide outputs.")
    elif overwrite_unfinished:
        output += interrupted
        if len(interrupted) > 0:
            warning(f"Overwriting {len(interrupted)} unfinished slide outputs.")
    elif len(processed) > 0:
        info(f"Skipping {len(processed)} processed slides.")
    # Verbose.
    if len(output) == 0:
        error("No slides to process.")
    info(f"Processing {len(output)} slides.")
    return output


def cut_slide(
    path: Path,
    *,
    reader_kwargs: dict,
    max_dimension: int,
    tissue_kwargs: dict,
    tile_kwargs: dict,
    save_kwargs: dict,
) -> tuple[Path, Optional[Exception]]:
    try:
        reader = SlideReader(path, **reader_kwargs)
        if tissue_kwargs["level"] is None:
            tissue_kwargs["level"] = reader.level_from_max_dimension(max_dimension)
        threshold, tissue_mask = reader.get_tissue_mask(**tissue_kwargs)
        coords = reader.get_tile_coordinates(tissue_mask=tissue_mask, **tile_kwargs)
        reader.save_regions(coordinates=coords, threshold=threshold, **save_kwargs)
    except Exception as e:  # noqa
        return path, e
    return path, None


def info(msg: str) -> None:
    """Display info message."""
    prefix = click.style("INFO: ", bold=True, fg="cyan")
    click.echo(prefix + msg)


def warning(msg: str) -> None:
    """Display warning message."""
    prefix = click.style("WARNING: ", bold=True, fg="yellow")
    click.echo(prefix + msg)


def error(msg: str, exit_integer: int = 1) -> NoReturn:
    """Display error message and exit."""
    prefix = click.style("ERROR: ", bold=True, fg="red")
    click.echo(prefix + msg)
    sys.exit(exit_integer)
