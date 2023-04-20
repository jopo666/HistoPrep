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
    "--save-metrics",
    "--save-masks",
    "--save-thumbnails",
    "--overwrite",
    "--unfinished",
    "--format",
    "--quality",
    "--use-csv",
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
    help="Slide pyramid level for tile extraction.",
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
@click.option(  # save_thumbnails
    "--save-thumbnails",
    type=click.BOOL,
    default=True,
    show_default=True,
    help="Save thumbnail images.",
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
    "--use-csv",
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
    help="Slide pyramid level for tissue detection.",
)
@click.option(  # max_dimension
    "--max-dimension",
    metavar="INT",
    type=click.IntRange(min=0),
    default=8192,
    show_default=True,
    help="Select first pyramid level with both dimensions smaller than this value.",
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
    out_of_bounds: bool = True,
    # Tile saving.
    save_metrics: bool = True,
    save_masks: bool = True,
    save_thumbnails: bool = True,
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
            "out_of_bounds": out_of_bounds,
            "max_background": max_background,
        },
        "save_kwargs": {
            "parent_dir": parent_dir,
            "level": level,
            "save_metrics": save_metrics,
            "save_masks": save_masks,
            "save_thumbnails": save_thumbnails,
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
