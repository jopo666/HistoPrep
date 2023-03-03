__all__ = ["format_level", "level_from_index"]

ERROR_LEVEL = "Level {} could not be found, select from {}."


def format_level(level: int, available: list[int]) -> int:
    """Format level."""
    if level < 0:
        return level_from_index(level, available=available)
    if level in available:
        return level
    raise ValueError(ERROR_LEVEL.format(level, available))


def level_from_index(index: int, available: list[int]) -> int:
    """Check if a `index` exists in `available` and format it correctly."""
    if abs(index) > len(available):
        raise ValueError(ERROR_LEVEL.format(index, available))
    return available[index]
