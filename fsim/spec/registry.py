"""Task registry."""
from __future__ import annotations

from fsim.spec.chair import CHAIR
from fsim.spec.square_table import ONE_LEG, SQUARE_TABLE
from fsim.spec.types import TaskSpec

TASKS: dict[str, TaskSpec] = {t.name: t for t in (ONE_LEG, SQUARE_TABLE, CHAIR)}


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}") from None
