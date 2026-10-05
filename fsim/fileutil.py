"""Atomic file writes that tolerate several processes or threads writing the same file
(e.g. the collection server and a replay tool building the same generated assets at once)."""
from __future__ import annotations

import os
import threading
import time


def unique_tmp(path: str) -> str:
    """A temp name next to `path` that no other process or thread uses."""
    return f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"


def atomic_replace(tmp: str, path: str, retries: int = 30):
    """os.replace, retried where a concurrent replace of the same target is refused (Windows)."""
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.01 * (attempt + 1))


def atomic_write_text(path: str, text: str):
    tmp = unique_tmp(path)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    atomic_replace(tmp, path)
