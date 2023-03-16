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
    """Find level which is closest to `dimensions`."""
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
    """Find level with both dimensions less or equal to `max_dimension`."""
    for level, (level_h, level_w) in level_dimensions.items():
        if level_h <= max_dimension and level_w <= max_dimension:
            return level
    return list(level_dimensions.keys())[-1]
