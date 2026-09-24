from agl import TypeContract, array


def build(target: TypeContract, depth: int, doc: str | None = None) -> object:
    """Build a value of *target*; *depth* picks enum members and bounds recursion."""
    match target.kind:
        case "bool":
            return depth % 2 == 1
        case "int":
            return depth
        case "text":
            return doc if doc is not None else target.label
        case "array":
            assert target.items is not None
            return array([build(target.items, depth)])
        case "enum":
            members = list(target.members.values())
            return build(members[max(0, min(depth, len(members) - 1))], depth)
    assert target.nominal is not None
    return target.nominal(
        **{
            field.name: build(field.contract, depth - 1, field.doc)
            for field in target.fields.values()
        }
    )


def guess(target: TypeContract, depth: int) -> object:
    return build(target, depth)


def divine(target: TypeContract, oracle: object) -> object:
    return build(target, getattr(oracle, "depth"))


def describe(target: TypeContract) -> str:
    return f"{target.label}: {target.doc}"
