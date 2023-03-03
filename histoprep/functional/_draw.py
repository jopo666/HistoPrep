__all__ = ["draw_tiles"]

from typing import Optional, Union

import numpy as np
from matplotlib.font_manager import fontManager
from PIL import Image, ImageDraw, ImageFont

from histoprep.backend._functional import divide_xywh

from ._check import check_image

ERROR_TEXT_ITEM_LENGTH = (
    "Length of text items ({}) does not match length of coordinates ({})."
)


def draw_tiles(
    image: Union[np.ndarray, Image.Image],
    coordinates: list[tuple[int, int, int, int]],
    downsample: Union[float, tuple[float, float]],
    *,
    rectangle_outline: str = "red",
    rectangle_fill: Optional[str] = None,
    rectangle_width: int = 1,
    highlight_first: bool = False,
    highlight_outline: str = "blue",
    text_items: Optional[list[str]] = None,
    text_color: str = "black",
    text_proportion: float = 0.75,
    text_font: str = "monospace",
    alpha: float = 0.0,
) -> Image.Image:
    """Function to draw tiles to an image. Useful for visualising tiles/predictions.

    Args:
        image: Image to draw to.
        coordinates: Tile coordinates.
        downsample: Downsample for the image. If coordinates are from the same image,
            set this to 1.0.
        rectangle_outline: Outline color of each tile. Defaults to "red".
        rectangle_fill: Fill color of each tile. Defaults to None.
        rectangle_width: Width of each tile edges. Defaults to 1.
        highlight_first: Highlight first tile, useful when tiles overlap.
            Defaults to False.
        highlight_outline: Highlight color for the first tile. Defaults to "black".
        text_items: Text items for each tile. Length must match `coordinates`.
            Defaults to None.
        text_offset: Offset pixels from the lower left corner. Defaults to 5.
        text_color: Text color. Defaults to "black".
        text_proportion: Proportion of space the text takes in each tile.
            Defaults to 0.75.
        text_font: Passed to matplotlib's `fontManager.find_font` function. Defaults to
            "monospace".
        alpha: Alpha value for blending the original image and drawn image.
            Defaults to 0.0.

    Raises:
        ValueError: Text item length does not match length of coordinates.

    Returns:
        Annotated image.
    """
    # Check image and convert to PIL Image.
    image = Image.fromarray(check_image(image)).convert("RGB")
    # Check arguments.
    if text_items is not None:
        if len(text_items) != len(coordinates):
            raise ValueError(
                ERROR_TEXT_ITEM_LENGTH.format(len(text_items), len(coordinates))
            )
    else:
        text_items = [None] * len(coordinates)
    # Draw tiles.
    font = None
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for idx, (xywh, text) in enumerate(zip(coordinates, text_items)):
        # Downscale coordinates.
        x, y, w, h = divide_xywh(xywh, downsample)
        # Draw rectangle.
        draw.rectangle(
            ((x, y), (x + w, y + h)),
            fill=rectangle_fill,
            outline=rectangle_outline,
            width=rectangle_width,
        )
        if text is not None:
            # Define font.
            if font is None:
                max_length = max(3, max(len(str(x)) for x in text_items))
                font_path = fontManager.findfont(text_font)
                # Figure width coefficient to find correct font size.
                font_32 = ImageFont.FreeTypeFont(font_path, size=32).getbbox("W")
                font_64 = ImageFont.FreeTypeFont(font_path, size=64).getbbox("W")
                font_coeff = font_64[2] / font_32[2]
                # Create font.
                font = ImageFont.FreeTypeFont(
                    font_path, size=round(font_coeff * text_proportion * w / max_length)
                )
            # Write text.
            draw.text(
                xy=(x + rectangle_width, y + rectangle_width),
                text=str(text_items[idx]),
                font=font,
                fill=text_color,
            )
    # Highlight first.
    if highlight_first and len(coordinates) > 0:
        x, y, w, h = divide_xywh(coordinates[0], downsample)
        draw.rectangle(
            ((x, y), (x + w, y + h)),
            fill=rectangle_fill,
            outline=highlight_outline,
            width=rectangle_width,
        )
    # Blend.
    return Image.blend(annotated, image, alpha)
