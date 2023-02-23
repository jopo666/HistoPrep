import sys
from typing import NoReturn

import click


def info(msg: str) -> None:
    """Display info message."""
    prefix = click.style("INFO: ", bold=True, fg="cyan")
    click.echo(prefix + msg)


def warning(msg: str) -> None:
    """Display warning message."""
    prefix = click.style("WARNING: ", bold=True, fg="yellow")
    click.echo(prefix + msg)


def error(msg: str, exit_integer: int = 1) -> NoReturn:
    """Display error message and exit."""
    prefix = click.style("ERROR: ", bold=True, fg="red")
    click.echo(prefix + msg)
    sys.exit(exit_integer)
