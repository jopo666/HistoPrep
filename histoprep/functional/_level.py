__all__ = [
    "_format_level",
    "_level_from_dimensions",
    "_level_from_max_dimension",
]

ERROR_LEVEL = "Level {} could not be found, select from {}."


def _format_level(level: int, available: list[int]) -> int:
    """Format level."""
    if level < 0:
        if abs(level) > len(available):
            raise ValueError(ERROR_LEVEL.format(level, available))
        return available[level]
    if level in available:
        return level
    raise ValueError(ERROR_LEVEL.format(level, available))


def _level_from_dimensions(
    dimensions: tuple[int, int], level_dimensions: dict[int, tuple[int, int]]
) -> int:
    """Find level which is closest to `dimensions`.

    Args:
        dimensions: Height and width to match.

    Returns:
        Level which is closest to `dimensions`.

    Example:
        >>> level_dims = {1: (256, 256), 2: (128, 128), 3: (64, 64)}
        >>> _level_from_dimensions(dimensions=(100, 100), level_dimensions=level_dims)
        2
    """
    height, width = dimensions
    available = []
    distances = []
    for level, (level_h, level_w) in level_dimensions.items():
        available.append(level)
        distances.append(abs(level_h - height) + abs(level_w - width))
    return available[distances.index(min(distances))]


def _level_from_max_dimension(
    max_dimension: int, level_dimensions: dict[int, tuple[int, int]]
) -> int:
    """Find level with both dimensions less or equal to `max_dimension`.

    Args:
        max_dimension: Maximum dimension for the level. Defaults to 4096.

    Returns:
        Level with both dimensions less than `max_dimension`, or the smallest level.

    Example:
        >>> level_dims = {1: (256, 256), 2: (128, 128), 3: (64, 64)}
        >>> _level_from_max_dimension(max_dimension=100, level_dimensions=level_dims)
        3
    """
    for level, (level_h, level_w) in level_dimensions.items():
        if level_h <= max_dimension and level_w <= max_dimension:
            return level
    return list(level_dimensions.keys())[-1]
