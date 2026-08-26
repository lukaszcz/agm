from agl import AglException, Problem


def invoke(f):
    return f()


def catch_and_reraise(f):
    try:
        return f()
    except AglException as error:
        raise error


def raise_from_companion():
    raise AglException(Problem(message="companion", detail="initiated"))
