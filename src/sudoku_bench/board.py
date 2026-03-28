from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class Board:
    cells: list[list[Optional[int]]]
    givens: frozenset[tuple[int, int]]
    box_rows: int
    box_cols: int

    @property
    def size(self) -> int:
        return self.box_rows * self.box_cols

    def is_given(self, row: int, col: int) -> bool:
        return (row, col) in self.givens

    @property
    def cells_filled(self) -> int:
        return sum(1 for row in self.cells for cell in row if cell is not None)

    @property
    def total_cells(self) -> int:
        return self.size * self.size

    def copy_with_cells(self, new_cells: list[list[Optional[int]]]) -> Board:
        return Board(
            cells=[row[:] for row in new_cells],
            givens=self.givens,
            box_rows=self.box_rows,
            box_cols=self.box_cols,
        )
