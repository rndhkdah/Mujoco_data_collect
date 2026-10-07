"""Task registry."""
from __future__ import annotations

from fsim.spec.cabinet import CABINET
from fsim.spec.chair import CHAIR
from fsim.spec.desk import DESK
from fsim.spec.drawer import DRAWER
from fsim.spec.lamp import LAMP
from fsim.spec.round_table import ROUND_TABLE
from fsim.spec.square_table import ONE_LEG, SQUARE_TABLE
from fsim.spec.stool import STOOL
from fsim.spec.types import TaskSpec

TASKS: dict[str, TaskSpec] = {t.name: t for t in (ONE_LEG, SQUARE_TABLE, DESK, ROUND_TABLE, STOOL, LAMP, DRAWER, CABINET, CHAIR)}


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}") from None
