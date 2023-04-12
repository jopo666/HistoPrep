ERROR_LEVEL = "Level {} could not be found, select from {}."


def format_level(level: int, available: list[int]) -> int:
    """Format level."""
    if level < 0:
        if abs(level) > len(available):
            raise ValueError(ERROR_LEVEL.format(level, available))
        return available[level]
    if level in available:
        return level
    raise ValueError(ERROR_LEVEL.format(level, available))
