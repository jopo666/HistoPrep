import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

import histoprep.functional as F

ERROR_OUTPUT_DIR_IS_FILE = "Output directory exists but it is a file."
ERROR_CANNOT_OVERWRITE = "Output directory exists, but `overwrite=False`."


@dataclass
class SlideRegionData:
    """Data class representing region data."""

    xywh: tuple[int, int, int, int]
    image: np.ndarray
    mask: np.ndarray
    metadata: dict

    def save_image(
        self,
        output_dir: Union[str, Path],
        image_format: str = "jpeg",
        quality: int = 80,
    ) -> str:
        """Save image to output directory."""
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        while image_format.startswith("."):
            image_format = image_format[1:]
        filename = "x{}_y{}_w{}_h{}".format(*self.xywh)
        filepath = output_dir / f"{filename}.{image_format}"
        filepath.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(self.image).save(filepath, quality=quality)
        return str(filepath)

    def save_mask(self, output_dir: Union[str, Path], **writer_kwargs) -> str:
        """Save mask to output directory."""
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        filepath = output_dir / "x{}_y{}_w{}_h{}.png".format(*self.xywh)
        filepath.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(self.mask).save(filepath, **writer_kwargs)
        return str(filepath)


def prepare_output_dir(
    *, parent_dir: Union[str, Path], slide_name: str, overwrite: bool
) -> Path:
    """Prepare output directory."""
    if not isinstance(parent_dir, Path):
        parent_dir = Path(parent_dir)
    output_dir = parent_dir / slide_name
    if output_dir.exists():
        if output_dir.is_file():
            raise NotADirectoryError(ERROR_OUTPUT_DIR_IS_FILE)
        if not overwrite:
            raise ValueError(ERROR_CANNOT_OVERWRITE)
        shutil.rmtree(output_dir)
    return output_dir


def worker_init(worker_state, reader_class, path: Path) -> None:  # noqa
    """Worker initialization function for `worker_save_region`."""
    worker_state["reader"] = reader_class(path)


def worker_save_region(
    worker_state: dict,
    xywh: tuple[int, int, int, int],
    *,
    output_dir: Path,
    level: int,
    threshold: int,
    sigma: float,
    save_paths: bool,
    save_metrics: bool,
    save_masks: bool,
    image_format: str,
    quality: int,
    raise_exception: bool,
    image_dir: str,
) -> Optional[dict]:
    """Worker function to read and save images and masks."""
    # Read region.
    region_data = read_region_data(
        reader=worker_state["reader"],
        xywh=xywh,
        level=level,
        threshold=threshold,
        sigma=sigma,
        skip_metrics=not save_metrics,
        raise_exception=raise_exception,
    )
    if isinstance(region_data, Exception):
        return region_data
    # Save images.
    paths = save_region_data(
        output_dir=output_dir,
        region_data=region_data,
        save_masks=save_masks,
        image_format=image_format,
        quality=quality,
        image_dir=image_dir,
        raise_exception=raise_exception,
    )
    if isinstance(paths, Exception):
        return paths
    if save_paths:
        return {**paths, **region_data.metadata}
    return region_data.metadata


def read_region_data(
    *,
    reader,  # noqa
    xywh: tuple[int, int, int, int],
    level: int,
    threshold: int,
    sigma: float,
    skip_metrics: bool,
    raise_exception: bool,
) -> Union[SlideRegionData, Exception]:
    """Read region image, generate mask and get image metrics safely."""
    try:
        x, y, w, h = xywh
        image = reader.read_region(xywh, level=level)
        __, mask = F.detect_tissue(image, threshold=threshold, sigma=sigma)
        metadata = {"x": x, "y": y, "w": w, "h": h}
        if not skip_metrics:
            metadata.update(F.calculate_metrics(image, mask))
    except KeyboardInterrupt:
        raise KeyboardInterrupt from None
    except Exception as catched_exception:  # noqa
        if raise_exception:
            raise catched_exception  # noqa
        return catched_exception
    return SlideRegionData(xywh=xywh, image=image, mask=mask, metadata=metadata)


def save_region_data(
    *,
    output_dir: Path,
    region_data: SlideRegionData,
    save_masks: bool,
    image_format: str,
    quality: int,
    image_dir: str,
    raise_exception: bool,
) -> Union[dict[str, str], Exception]:
    """Save image/mask safely."""
    try:
        output = {}
        output["path"] = region_data.save_image(
            output_dir=output_dir / image_dir,
            image_format=image_format,
            quality=quality,
        )
        if save_masks:
            output["mask_path"] = region_data.save_mask(output_dir=output_dir)
    except KeyboardInterrupt:
        raise KeyboardInterrupt from None
    except Exception as catched_exception:  # noqa
        if raise_exception:
            raise catched_exception  # noqa
        return catched_exception
    return output
