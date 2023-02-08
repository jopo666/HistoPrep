import functools
import glob
import sys
from pathlib import Path
from typing import Optional

import click
import mpire
from histoprep import ReaderBackend

from ._cut import cut_slide
from ._files import convert_to_path
from ._verbose import error, info, warning


@click.command()
@click.option(
    "-i",
    "--input",
    "input_pattern",
    required=True,
    metavar="<file-pattern>",
    type=click.STRING,
    help="Input file pattern.",
)
@click.option(
    "-o",
    "--output",
    "parent_dir",
    required=True,
    callback=convert_to_path,
    metavar="<directory>",
    type=click.Path(file_okay=False),
    help="Parent directory for all outputs.",
)
@click.option(
    "--backend",
    metavar="<reader-backend>",
    type=click.Choice(
        choices=["AUTOMATIC", "PILLOW", "OPENSLIDE", "CZI"], case_sensitive=False
    ),
    default="automatic",
    show_default=True,
    help="Reader backend for opening slides.",
)
@click.option(
    "--width",
    metavar="<int>",
    type=click.IntRange(min=0, min_open=False),
    default=640,
    show_default=True,
    help="Tile width.",
)
@click.option(
    "--height",
    metavar="<int>",
    type=click.IntRange(min=0, min_open=False),
    default=None,
    show_default="width",
    help="Tile height.",
)
@click.option(
    "--overlap",
    metavar="<float>",
    type=click.FloatRange(min=0, max=1, min_open=False, max_open=False),
    default=0.0,
    show_default=True,
    help="Tile overlap.",
)
@click.option(
    "--background",
    metavar="<float>",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    default=0.6,
    show_default=True,
    help="Maximum background.",
)
@click.option(
    "--level",
    metavar="<int>",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Slide level.",
)
@click.option(
    "--out-of-bounds",
    metavar="<bool>",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Allow out-of-bounds tiles. ",
)
@click.option(
    "--save_paths",
    metavar="<bool>",
    type=click.BOOL,
    default=True,
    show_default=True,
    help="Save filepaths to metadata.",
)
@click.option(
    "--save_metrics",
    metavar="<bool>",
    type=click.BOOL,
    default=True,
    show_default=True,
    help="Save image metrics to metadata.",
)
@click.option(
    "--save_masks",
    metavar="<bool>",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Save tissue masks.",
)
@click.option(
    "--overwrite",
    metavar="<bool>",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Overwrite any existing slide outputs.",
)
@click.option(
    "--overwrite_unfinished",
    metavar="<bool>",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Overwrite any existing slide outputs if metadata is missing.",
)
@click.option(
    "--quality",
    metavar="<int>",
    type=click.IntRange(min=0, max=100),
    default=80,
    show_default=True,
    help="Quality for JPEG-compression.",
)
@click.option(
    "--format",
    "file_format",
    metavar="<str>",
    type=click.STRING,
    default="jpeg",
    show_default=True,
    help="File format for tiles.",
)
@click.option(
    "--threshold",
    metavar="<int>",
    type=click.IntRange(min=0, max=255, min_open=False),
    default=None,
    show_default="Otsu's threshold",
    help="Threshold for tissue detection.",
)
@click.option(
    "--num_workers",
    metavar="<int>",
    type=click.IntRange(min=0, min_open=False),
    default=None,
    show_default="CPU count",
    help="Threshold for tissue detection.",
)
@click.option(
    "--threshold_multiplier",
    metavar="<float>",
    type=click.FloatRange(min=0, min_open=False),
    default=1.05,
    show_default=True,
    help="Multiplier for Otsu's threshold. Ignored if threshold is set.",
)
@click.option(
    "--ignore_white",
    metavar="<bool>",
    type=click.BOOL,
    default=True,
    show_default=True,
    help="Ignore white pixels when finding Otsu's threshold.",
)
@click.option(
    "--ignore_black",
    metavar="<bool>",
    type=click.BOOL,
    default=True,
    show_default=True,
    help="Ignore black pixels when finding Otsu's threshold.",
)
@click.option(
    "--tissue_level",
    metavar="<int>",
    type=click.IntRange(min=0),
    default=None,
    show_default=True,
    help="Slide level for tissue detection.",
)
@click.option(
    "--tissue_level_dim",
    metavar="<int>",
    type=click.IntRange(min=0),
    default=8192,
    show_default=True,
    help=(
        "Select first level with both dimensions smaller than this value. "
        "Ignored if tissue_level is set."
    ),
)
@click.option(
    "--tissue_sigma",
    metavar="<float>",
    type=click.FloatRange(min=0),
    default=1.0,
    show_default=True,
    help="Sigma for gaussian blurring during tissue detection.",
)
def cut(
    *,
    # Slide paths.
    input_pattern: str,
    parent_dir: Path,
    backend: str,
    # Tile extraction.
    width: int,
    height: int,
    overlap: float,
    background: float,
    level: int,
    out_of_bounds: bool,
    # Tile saving.
    save_paths: bool,
    save_metrics: bool,
    save_masks: bool,
    overwrite: bool,
    overwrite_unfinished: bool,
    quality: int,
    file_format: str,
    num_workers: Optional[int],
    # Tissue detection.
    threshold: Optional[float],
    threshold_multiplier: float,
    ignore_white: bool,
    ignore_black: bool,
    tissue_level: Optional[int],
    tissue_level_dim: int,
    tissue_sigma: float,
) -> None:
    # Define slide paths.
    slide_paths = []
    for path in (Path(x) for x in glob.glob(input_pattern)):
        if path.is_file():
            slide_paths.append(path)
    if len(slide_paths) == 0:
        error(f"Could not find any files matching pattern '{input_pattern}'.")
    else:
        info(f"Found {len(slide_paths)} files matching pattern: {input_pattern}.")
    # Filter slide paths.
    slide_paths = filter_slide_paths(
        slide_paths=slide_paths,
        parent_dir=parent_dir,
        overwrite=overwrite,
        overwrite_unfinished=overwrite_unfinished,
    )
    # Define cutting function.
    cut_fn = functools.partial(
        cut_slide,
        parent_dir=parent_dir,
        backend=ReaderBackend[backend.upper()],
        # Tile extraction.
        width=width,
        height=height,
        overlap=overlap,
        level=level,
        out_of_bounds=out_of_bounds,
        max_background=background,
        # Tile saving.
        save_paths=save_paths,
        save_masks=save_masks,
        save_metrics=save_metrics,
        file_format=file_format,
        quality=quality,
        # Tissue detection.
        tissue_level=tissue_level,
        tissue_level_dim=tissue_level_dim,
        tissue_sigma=tissue_sigma,
        tissue_threshold=threshold,
        tissue_threshold_multiplier=threshold_multiplier,
        tissue_ignore_white=ignore_white,
        tissue_ignore_black=ignore_black,
        # Fixed inputs.
        overwrite=True,  # These have been filtered.
        num_workers=0,  # Each process handles one slide.
        verbose=False,  # Common progress bar.
    )
    # Process.
    try:
        with mpire.WorkerPool(n_jobs=num_workers) as pool:
            pool.map(
                cut_fn,
                slide_paths,
                progress_bar=True,
                progress_bar_options={"desc": "Cutting slides"},
            )
    except KeyboardInterrupt:
        error("Keyboard interruption detected.")
    except Exception as e:  # noqa
        error(f"Could not process slides due to exception: {e}")


def filter_slide_paths(
    *,
    slide_paths: list[Path],
    parent_dir: Path,
    overwrite: bool,
    overwrite_unfinished: bool,
) -> dict[str, list[Path]]:
    """Filter out processed slides."""
    output = []
    num_unfinished = 0
    num_processed = 0
    num_overwrite = 0
    for path in slide_paths:
        output_dir = parent_dir / path.name.rstrip(path.suffix)
        if overwrite:
            # All paths will be processed.
            output.append(path)
            if output_dir.exists():
                # All paths will be processed.
                num_overwrite += 1
        elif not output_dir.exists():
            output.append(path)
        elif overwrite_unfinished and not (output_dir / "metadata.parquet").exists():
            output.append(path)
            num_unfinished += 1
        else:
            num_processed += 1
    # Exit if all have been processed.
    if len(output) == 0:
        info("All slides have been processed.")
        sys.exit(0)
    # Verbose.
    if num_processed > 0:
        info(f"Skipping {num_processed} processed slides.")
    if num_overwrite:
        warning(f"Overwriting {num_overwrite} slide outputs.")
    if num_unfinished > 0:
        warning(f"Overwriting {num_unfinished} unfinished slide outputs.")
    return output
