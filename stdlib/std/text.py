"""Runtime interpolation companion for ``std/text``."""

from agm.util.interp import interp as _interp


def interp(template: str, vars: dict[str, str]) -> str:
    """Interpolate named runtime holes from ``vars``."""
    return _interp(template, vars)
