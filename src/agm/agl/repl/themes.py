"""Colour themes for the AgL REPL.

Two concrete themes are provided — ``dark`` (VS Code Dark+) and ``light``
(VS Code Light+).  :func:`detect_terminal_theme` infers the terminal background
from the ``$COLORFGBG`` environment variable (set by most terminal emulators):
the last semicolon-delimited segment is a 0–15 ANSI colour index where ``15``
signals a white/light background.  If the variable is absent or unparseable
the function defaults to ``"dark"``.

:func:`get_style` resolves an ``"auto"`` theme name to the detected theme before
returning the matching ``prompt_toolkit`` ``Style`` object.
"""

from __future__ import annotations

import os

from prompt_toolkit.styles import Style

THEME_NAMES: tuple[str, ...] = ("dark", "light", "auto")

DARK_THEME: Style = Style.from_dict(
    {
        "agl.keyword": "bold #569cd6",
        "agl.string": "#ce9178",
        "agl.number": "#b5cea8",
        "agl.operator": "#d4d4d4",
        "agl.type": "#4ec9b0",
        "agl.constructor": "#dcdcaa",
        "agl.name": "",
        "agl.banner": "italic #808080",
        "agl.prompt": "bold #569cd6",
    }
)

LIGHT_THEME: Style = Style.from_dict(
    {
        "agl.keyword": "bold #0000ff",
        "agl.string": "#a31515",
        "agl.number": "#098658",
        "agl.operator": "#000000",
        "agl.type": "#267f99",
        "agl.constructor": "#795e26",
        "agl.name": "",
        "agl.banner": "italic #767676",
        "agl.prompt": "bold #0000ff",
    }
)

_THEME_STYLES: dict[str, Style] = {"dark": DARK_THEME, "light": LIGHT_THEME}


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


def get_style(theme: str) -> Style:
    """Return the ``prompt_toolkit`` ``Style`` for *theme*.

    ``"auto"`` resolves via :func:`detect_terminal_theme`.  Unknown names fall
    back to the dark theme.
    """
    resolved = detect_terminal_theme() if theme == "auto" else theme
    return _THEME_STYLES.get(resolved, DARK_THEME)
