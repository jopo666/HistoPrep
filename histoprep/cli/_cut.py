from pathlib import Path
from typing import Optional, Union

from histoprep import SlideReader

STRING_TO_BACKEND = {"AUTOMATIC"}


def cut_slide(
    slide_path: Union[str, Path],
    parent_dir: Union[str, Path],
    *,
    backend: ReaderBackend = ReaderBackend.AUTOMATIC,
    # Tile extraction.
    width: int = 640,
    height: int = None,
    overlap: float = 0.0,
    level: int = 0,
    allow_out_of_bounds: bool = False,
    max_background: float = 0.6,
    # Tile saving.
    save_paths: bool = False,
    save_masks: bool = False,
    save_metrics: bool = False,
    overwrite: bool = False,
    file_format: str = "jpeg",
    quality: int = 80,
    num_workers: Optional[int] = None,
    # Tissue detection.
    tissue_level: Optional[int] = None,
    tissue_level_dim: int = 8192,
    tissue_sigma: float = 1.0,
    tissue_threshold: Optional[float] = None,
    tissue_threshold_multiplier: float = 1.05,
    tissue_ignore_white: bool = True,
    tissue_ignore_black: bool = True,
    # Progress bar.
    verbose: bool = True,
) -> None:
    # Initialize reader.
    reader = SlideReader(path=slide_path, backend=backend)
    # Detect tissue.
    tissue_mask = reader.detect_tissue(
        level=tissue_level,
        level_dim=tissue_level_dim,
        sigma=tissue_sigma,
        threshold=tissue_threshold,
        multiplier=tissue_threshold_multiplier,
        ignore_white=tissue_ignore_white,
        ignore_black=tissue_ignore_black,
    )
    # Get coordinates.
    tile_coordinates = reader.get_tile_coordinates(
        tissue_mask=tissue_mask,
        width=width,
        height=height,
        overlap=overlap,
        level=level,
        allow_out_of_bounds=allow_out_of_bounds,
        max_background=max_background,
    )
    # Save tiles.
    reader.save_tiles(
        parent_dir=parent_dir,
        tile_coordinates=tile_coordinates,
        save_paths=save_paths,
        save_metrics=save_metrics,
        save_masks=save_masks,
        overwrite=overwrite,
        raise_exception=False,
        num_workers=num_workers,
        format=file_format,
        quality=quality,
        verbose=verbose,
    )
