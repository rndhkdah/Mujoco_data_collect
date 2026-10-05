"""Atomic writes shared by several writers (envs starting at the same time rebuild the same assets)."""
import os
import threading

from fsim.fileutil import atomic_write_text, unique_tmp


def test_concurrent_writers_of_one_file_all_succeed(tmp_path):
    path = str(tmp_path / "part.xml")
    text = "<mujoco/>" * 2000
    errors = []

    def writer():
        try:
            for _ in range(40):
                atomic_write_text(path, text)
        except Exception as e:      # noqa: BLE001  (collected for the assert below)
            errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    with open(path, encoding="utf-8") as f:
        assert f.read() == text
    assert os.listdir(tmp_path) == ["part.xml"]          # no temp files left behind


def test_tmp_names_differ_between_threads():
    names = [unique_tmp("x")]
    t = threading.Thread(target=lambda: names.append(unique_tmp("x")))
    t.start()
    t.join()
    assert names[0] != names[1] and all(n.endswith(".tmp") for n in names)
