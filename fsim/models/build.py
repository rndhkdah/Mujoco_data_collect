"""Generate the visual meshes for every registered task: python -m fsim.models.build [--force]"""
from __future__ import annotations

import argparse

from fsim.models.meshes import build_task_meshes
from fsim.spec import TASKS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="regenerate even if the files exist")
    p.add_argument("--out", default=None, help="output root (default fsim/assets/generated or $FSIM_GENERATED)")
    args = p.parse_args()
    for task in TASKS.values():
        for part, path in build_task_meshes(task, args.out, force=args.force).items():
            print(f"[{task.name}] {part}: {path}")


if __name__ == "__main__":
    main()
