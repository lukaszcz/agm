"""Theme names and terminal-background detection for the AgL REPL — UI-free.

:data:`THEME_NAMES` lists the available theme names; :func:`detect_terminal_theme`
infers the terminal background from the ``$COLORFGBG`` environment variable (set
by most terminal emulators): the last semicolon-delimited segment is a 0-15 ANSI
colour index where ``15`` signals a white/light background. If the variable is
absent or unparseable the function defaults to ``"dark"``.

This leaf imports nothing from ``prompt_toolkit``, so anything that only needs
theme *names* or *detection* — the meta-command dispatcher, the plain REPL front
end — never pulls prompt_toolkit in. The actual ``Style`` objects and
``get_style`` (the prompt_toolkit-touching half) live in
:mod:`agm.agl.repl.themes`, which imports this module.
"""

from __future__ import annotations

import os

THEME_NAMES: tuple[str, ...] = ("dark", "light", "auto")


def detect_terminal_theme() -> str:
    """Infer whether the terminal background is dark or light.

    Reads ``$COLORFGBG`` (set by most terminal emulators as ``fg;bg`` or
    ``fg;unknown;bg``).  A trailing segment of ``15`` (white) indicates a light
    terminal; any other value, or an absent/malformed variable, returns ``"dark"``.
    """
    colorfgbg = os.environ.get("COLORFGBG", "")
    if colorfgbg:
        parts = colorfgbg.split(";")
        if parts[-1] == "15":
            return "light"
    return "dark"
