from __future__ import annotations

__all__ = ["allowed_dimensions", "divide_xywh", "multiply_xywh"]


def allowed_dimensions(
    xywh: tuple[int, int, int, int], dimensions: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Get allowed height and width which are inside dimensions."""
    x, y, w, h = xywh
    height, width = dimensions
    if y > height or x > width:
        # x or y is outside of dimensions.
        return (0, 0)
    if y + h > height:
        h = height - y
    if x + w > width:
        w = width - x
    return h, w


def divide_xywh(
    xywh: tuple[int, int, int, int], divisor: float | tuple[float, float]
) -> tuple[int, int, int, int]:
    """Divide xywh-coordinates with divisor(s)."""
    if not isinstance(divisor, (tuple, list)):
        divisor = (divisor, divisor)
    w_div, h_div = divisor
    x, y, w, h = xywh
    return round(x / w_div), round(y / h_div), round(w / w_div), round(h / h_div)


def multiply_xywh(
    xywh: tuple[int, int, int, int], multiplier: float | tuple[float, float]
) -> tuple[int, int, int, int]:
    """Divide xywh-coordinates with divisor(s)."""
    if not isinstance(multiplier, (tuple, list)):
        multiplier = (multiplier, multiplier)
    w_mult, h_mult = multiplier
    x, y, w, h = xywh
    return round(x * w_mult), round(y * h_mult), round(w * w_mult), round(h * h_mult)
