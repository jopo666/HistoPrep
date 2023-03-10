from __future__ import annotations

import polars as pl


class TileMetadata:
    def __init__(
        self,
        file_pattern: str,
        columns: list[str] = None,
        n_rows: int | None = None,
        csv_files: bool = False,
    ):
        pl.read_csv()
