"""Element names shared by the model builders and the runtime managers.

Names follow robosuite's MujocoXMLObject convention (object body renamed to "main", every
element prefixed with "<part>_", default free joint "joint0"), so the same names resolve in
the standalone scenes and inside robosuite.
"""


def body(part: str) -> str:
    return f"{part}_main"


def free_joint(part: str) -> str:
    return f"{part}_joint0"


def elem(part: str, local: str) -> str:
    return f"{part}_{local}"


def catch_geom(parent: str, hole: int) -> str:
    return elem(parent, f"catch{hole}")


def carriage(parent: str, hole: int) -> str:
    return elem(parent, f"hole{hole}_carriage")


def slide_joint(parent: str, hole: int) -> str:
    return elem(parent, f"hole{hole}_slide")


def hinge_joint(parent: str, hole: int) -> str:
    return elem(parent, f"hole{hole}_hinge")


def couple_eq(parent: str, hole: int) -> str:
    return elem(parent, f"hole{hole}_couple")


def weld_eq(parent: str, hole: int, child: str) -> str:
    return elem(parent, f"hole{hole}_weld_{child}")


def slot_carriage(parent: str, slot: int) -> str:
    return elem(parent, f"slot{slot}_carriage")


def slot_slide(parent: str, slot: int) -> str:
    return elem(parent, f"slot{slot}_slide")


def slot_lock_eq(parent: str, slot: int) -> str:
    return elem(parent, f"slot{slot}_lock")


def slot_weld_eq(parent: str, slot: int, child: str) -> str:
    return elem(parent, f"slot{slot}_weld_{child}")
