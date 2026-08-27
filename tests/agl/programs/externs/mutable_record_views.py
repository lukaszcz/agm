from agl import Callbacks, Mutable, Snapshot


def mutate(record: Mutable) -> None:
    record.value = 2


def reject_fixed_write(record: Mutable) -> bool:
    try:
        record.fixed = "changed"
    except AttributeError:
        return True
    return False


def return_received(record: Mutable) -> Mutable:
    record.value = 2
    return record


def construct() -> Mutable:
    return Mutable(value=9, fixed="made")


def mutate_member(member: object) -> None:
    member.value = 2


def snapshot_behavior(record: Snapshot) -> bool:
    try:
        record.value = 2
    except AttributeError:
        return hash(record) == hash(Snapshot(value=1)) and record == Snapshot(value=1)
    return False


_RETAINED: dict[str, object] = {}


def build_callbacks(seed):
    view = Callbacks(callback=seed)
    _RETAINED["view"] = view
    return view


def call_retained_callback(box: Callbacks) -> int:
    return _RETAINED["view"].callback(5)
