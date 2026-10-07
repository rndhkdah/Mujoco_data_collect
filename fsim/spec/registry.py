"""Task registry."""
from __future__ import annotations

from fsim.spec.lamp import LAMP
from fsim.spec.square_table import ONE_LEG, SQUARE_TABLE
from fsim.spec.stool import STOOL
from fsim.spec.types import TaskSpec

TASKS: dict[str, TaskSpec] = {t.name: t for t in (ONE_LEG, SQUARE_TABLE, STOOL, LAMP)}


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}") from None
