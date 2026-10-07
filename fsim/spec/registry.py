"""Task registry."""
from __future__ import annotations

from fsim.spec.square_table import ONE_LEG, SQUARE_TABLE
from fsim.spec import desk, round_table
from fsim.spec.types import TaskSpec

TASKS: dict[str, TaskSpec] = {t.name: t for t in (ONE_LEG, SQUARE_TABLE, desk.DESK, round_table.ROUND_TABLE)}


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}") from None
