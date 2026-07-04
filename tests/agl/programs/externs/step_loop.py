def next_step(n: int) -> dict[str, object]:
    if n >= 3:
        return {"$case": "Stop"}
    return {"$case": "Continue", "amount": n + 1}
