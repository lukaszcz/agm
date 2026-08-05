from agl import Step


def next_step(n: int) -> Step:
    if n >= 3:
        return Step.Stop()
    return Step.Continue(amount=n + 1)
