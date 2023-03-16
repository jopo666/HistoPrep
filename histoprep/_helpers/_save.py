from __future__ import annotations

__all__ = [
    "load_region_data",
    "prepare_output_dir",
    "save_regions",
]

import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import tqdm
from PIL import Image

import histoprep.functional as F

ERROR_OUTPUT_DIR_IS_FILE = "Output directory exists but it is a file."
ERROR_CANNOT_OVERWRITE = "Output directory exists, but `overwrite=False`."


@dataclass
class RegionData:
    image: np.ndarray
    mask: np.ndarray | None
    metrics: dict[str, float]

    def save_data(
        self,
        *,
        image_dir: Path,
        mask_dir: Path,
        xywh: tuple[int, int, int, int],
        quality: int,
        image_format: str,
        prefix: str | None,
    ) -> dict[str, float]:
        """Save image (and mask) and return region metadata."""
        metadata = dict(zip("xywh", xywh))
        filename = "x{}_y{}_w{}_h{}".format(*xywh)
        if prefix is not None:
            filename = f"{prefix}_{filename}"
        # Save image.
        image_path = image_dir / f"{filename}.{image_format}"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.image).save(image_path, quality=quality)
        metadata["path"] = str(image_path.resolve())
        # Save mask.
        if self.mask is not None:
            mask_path = mask_dir / f"{filename}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(self.mask).save(mask_path)
            metadata["mask_path"] = str(mask_path.resolve())
        return {**metadata, **self.metrics}


def save_regions(
    output_dir: Path,
    iterable: Iterator[RegionData, tuple[int, int, int, int]],
    *,
    desc: str,
    total: int,
    quality: int,
    image_format: str,
    image_dir: str,
    file_prefixes: list[str],
    verbose: bool,
) -> pl.DataFrame:
    """Save region data to output directory.

    Args:
        output_dir: Output directory.
        iterable: Iterable yieldin RegionData and xywh-coordinates.
        desc: For the progress bar.
        total: For the progress bar.
        quality: Quality of jpeg-compression.
        image_format: Image extension.
        image_dir: Image directory name
        file_prefixes: List of file prefixes.
        verbose: Enable progress bar.

    Returns:
        Polars dataframe with metadata.
    """
    progress_bar = tqdm.tqdm(
        iterable=iterable,
        desc=desc,
        disable=not verbose,
        total=total,
    )
    rows = []
    num_failed = 0
    for i, (region_data, xywh) in enumerate(progress_bar):
        if isinstance(region_data, Exception):
            num_failed += 1
            progress_bar.set_postfix({"failed": num_failed}, refresh=False)
        rows.append(
            region_data.save_data(
                image_dir=output_dir / image_dir,
                mask_dir=output_dir / "masks",
                xywh=xywh,
                quality=quality,
                image_format=image_format,
                prefix=None if file_prefixes is None else file_prefixes[i],
            )
        )
    return pl.DataFrame(rows)


def prepare_output_dir(output_dir: str | Path, *, overwrite: bool) -> Path:
    """Prepare output directory."""
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if output_dir.exists():
        if output_dir.is_file():
            raise NotADirectoryError(ERROR_OUTPUT_DIR_IS_FILE)
        if len(list(output_dir.iterdir())) > 0 and not overwrite:
            raise ValueError(ERROR_CANNOT_OVERWRITE)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_region_data(
    image: np.ndarray,
    *,
    save_masks: bool,
    save_metrics: bool,
    threshold: int | None,
) -> RegionData:
    """Helper transform to add tissue mask and image metrics during tile loading."""
    tissue_mask = None
    metrics = {}
    if save_masks or save_metrics:
        __, tissue_mask = F.get_tissue_mask(image=image, threshold=threshold)
    if save_metrics:
        metrics = F.calculate_metrics(image=image, tissue_mask=tissue_mask)
    return RegionData(
        image=image, mask=tissue_mask if save_masks else None, metrics=metrics
    )
