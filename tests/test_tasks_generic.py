"""Checks every registered task passes; a new task is covered just by registering it."""
import pytest

from fsim.spec.registry import TASKS
from task_checks import check_force_seat, check_hand_drive, check_reset, check_spec, compile_scene, mechanisms

NAMES = sorted(TASKS)
SPECS = [(n, i) for n in NAMES for i in range(len(mechanisms(TASKS[n])))]


@pytest.mark.parametrize("name", NAMES)
def test_spec_sanity(name):
    check_spec(TASKS[name])


@pytest.mark.parametrize("name", NAMES)
def test_scene_compiles_with_meshes(name):
    m = compile_scene(TASKS[name])
    assert m.nbody > len(TASKS[name].parts)


@pytest.mark.parametrize("level", ["low", "med", "high"])
@pytest.mark.parametrize("name", NAMES)
def test_reset_settles(name, level):
    if level == "high" and not TASKS[name].randomization.high_presets:
        pytest.skip("no high presets")
    check_reset(TASKS[name], level, seeds=(0,) if level == "high" else (0, 1))


@pytest.mark.parametrize("name", NAMES)
def test_force_seat_assembles(name):
    check_force_seat(TASKS[name])


@pytest.mark.parametrize("name,i", SPECS)
def test_hand_drive_seats_one_child(name, i):
    check_hand_drive(TASKS[name], mechanisms(TASKS[name])[i])


@pytest.mark.parametrize("name", NAMES)
def test_robosuite_env(name):
    pytest.importorskip("robosuite")
    from task_checks import check_env
    check_env(TASKS[name])
