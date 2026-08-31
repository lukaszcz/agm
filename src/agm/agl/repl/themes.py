"""Colour themes for the AgL REPL — the prompt_toolkit-touching half.

Two concrete themes are provided — ``dark`` (VS Code Dark+) and ``light``
(VS Code Light+).  :func:`get_style` resolves an ``"auto"`` theme name via
:func:`agm.agl.repl.theme_selection.detect_terminal_theme` before returning the
matching ``prompt_toolkit`` ``Style`` object.

Theme *names* and terminal-background *detection* are UI-free and live in
:mod:`agm.agl.repl.theme_selection`, which this module imports; keeping that
half separate means an importer that only needs names/detection (the
meta-command dispatcher, the plain REPL front end) never pulls prompt_toolkit
in — only this module (and its sole importer, :mod:`agm.agl.repl.console`)
does.
"""

from __future__ import annotations

from prompt_toolkit.styles import Style

from agm.agl.repl.theme_selection import detect_terminal_theme

DARK_THEME: Style = Style.from_dict(
    {
        "agl.keyword": "bold #569cd6",
        "agl.string": "#ce9178",
        "agl.number": "#b5cea8",
        "agl.operator": "#d4d4d4",
        "agl.type": "#4ec9b0",
        "agl.constructor": "#dcdcaa",
        "agl.comment": "italic #6a9955",
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
        "agl.comment": "italic #008000",
        "agl.name": "",
        "agl.banner": "italic #767676",
        "agl.prompt": "bold #0000ff",
    }
)

_THEME_STYLES: dict[str, Style] = {"dark": DARK_THEME, "light": LIGHT_THEME}


def get_style(theme: str) -> Style:
    """Return the ``prompt_toolkit`` ``Style`` for *theme*.

    ``"auto"`` resolves via :func:`~agm.agl.repl.theme_selection.detect_terminal_theme`.
    Unknown names fall back to the dark theme.
    """
    resolved = detect_terminal_theme() if theme == "auto" else theme
    return _THEME_STYLES.get(resolved, DARK_THEME)
