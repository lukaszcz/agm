def dedup(xs: list[object]) -> list[object]:
    seen: list[object] = []
    result: list[object] = []
    for x in xs:
        if x not in seen:
            seen.append(x)
            result.append(x)
    return result
