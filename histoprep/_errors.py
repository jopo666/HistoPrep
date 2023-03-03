__all__ = ["LevelNotFoundError"]


class LevelNotFoundError(Exception):
    def __init__(self, *args: object) -> None:
        self.args = args

    def __str__(self, *args: object) -> None:
        return "Level {} not found, choose from {}.".format(*self.args)
