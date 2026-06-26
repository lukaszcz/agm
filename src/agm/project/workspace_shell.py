"""Workspace shell wrapper managed under a user cache directory.

The wrapper launches the user's real interactive shell (``zsh``/``bash``/``sh``)
so that ``~/.zshrc``/``~/.bashrc``/``~/.shrc`` run normally — preserving the
user's keybindings, prompt, completions and aliases.  After the user's rc file
has been sourced, the wrapper appends ``eval "$(agm config env)"`` so the
workspace environment wins over anything the user's rc set.

Nothing is written under the project's ``.agent-files/``.  The wrapper and its
rc files live under ``$XDG_CACHE_HOME/agm/shell/<key>/`` (defaulting to
``~/.cache/agm/shell/<key>/``), keyed by the tmux session name.  This makes
shell reloads robust against deletion of ``.agent-files`` (the files no longer
live there) and against deletion of the rc subdirectories: the wrapper is
fully self-contained and regenerates any missing rc file inline before
exec'ing the real shell, so it never depends on the ``agm`` binary at
self-heal time.  ``agm workspace open`` cleans the per-session cache dir first
(so stale files from a previous open never linger), and ``agm workspace
close`` removes it.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from shlex import quote as shlex_quote

from agm.core.fs import chmod, mkdir, rmtree, write_text

SHELL_SUBDIR = "shell"
"""Subdirectory under the agm cache root that holds per-session shell wrappers."""

WRAPPER_NAME = "shell"
"""Filename of the wrapper script inside a per-session shell dir."""


def workspace_shell_root() -> Path:
    """Return the cache root holding per-session shell wrapper directories.

    Honors ``$XDG_CACHE_HOME``; falls back to ``~/.cache`` when unset.
    """

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        base = Path(xdg)
    else:
        base = Path.home() / ".cache"
    return base / "agm" / SHELL_SUBDIR


def _sanitize_session_key(session_name: str) -> str:
    """Return a filesystem-safe, collision-resistant key for *session_name*.

    tmux session names may contain ``/`` (e.g. ``proj/feat``) and other
    characters that are unsafe as path components.  Replace unsafe characters
    with ``__`` and append a short hash of the raw name to disambiguate keys
    that collide after sanitization.
    """

    safe = re.sub(r"[^A-Za-z0-9._-]+", "__", session_name).strip("._-")
    if not safe:
        safe = "session"
    digest = hashlib.sha1(session_name.encode("utf-8")).hexdigest()[:8]
    return f"{safe}-{digest}"


def workspace_shell_dir(session_name: str) -> Path:
    """Return the per-session shell directory for *session_name*."""

    return workspace_shell_root() / _sanitize_session_key(session_name)


def _real_shell(*, env: dict[str, str] | None) -> str:
    """Return the user's real interactive shell binary path.

    Prefers ``$AGM_REAL_SHELL`` (set by the wrapper, preserving the real shell
    across nested workspace shells) and falls back to ``$SHELL``; reads from
    *env* or the process environment when *env* is ``None``.  A candidate that
    points at our wrapper (a re-exec scenario — e.g. inside a workspace shell
    ``$SHELL`` is the wrapper) is skipped so we never loop back into ourselves.
    Falls back to ``/bin/sh`` when nothing usable is found.
    """

    source = os.environ if env is None else env
    wrapper_suffix = "/" + WRAPPER_NAME
    for name in ("AGM_REAL_SHELL", "SHELL"):
        candidate = source.get(name)
        if candidate and not candidate.endswith(wrapper_suffix):
            return candidate
    return "/bin/sh"


def _zshenv_body() -> str:
    # zsh sources $ZDOTDIR/.zshenv before .zshrc.  Re-source the user's
    # ~/.zshenv under the original ZDOTDIR so any env exported there (e.g.
    # PATH tweaks) still applies before the user's .zshrc runs.
    return "\n".join(
        [
            "# agm workspace shell: replay the user's ~/.zshenv",
            'if [ -z "${AGM_USER_ZDOTDIR:-}" ]; then',
            '  AGM_USER_ZDOTDIR="$HOME"',
            "fi",
            'if [ -f "$AGM_USER_ZDOTDIR/.zshenv" ]; then',
            '  . "$AGM_USER_ZDOTDIR/.zshenv"',
            "fi",
            "",
        ]
    )


def _zshrc_body() -> str:
    # Run the user's real ~/.zshrc with ZDOTDIR restored to $HOME (normal
    # behavior), so oh-my-zsh, bindkey maps, prompts and completions load.
    # Then append the agm workspace env so agm-set vars override the user's.
    return "\n".join(
        [
            "# agm workspace shell: source the user's ~/.zshrc, then apply agm env",
            'if [ -z "${AGM_USER_ZDOTDIR:-}" ]; then',
            '  AGM_USER_ZDOTDIR="$HOME"',
            "fi",
            '_agm_saved_zdotdir="$ZDOTDIR"',
            'export ZDOTDIR="$AGM_USER_ZDOTDIR"',
            'if [ -f "$ZDOTDIR/.zshrc" ]; then',
            '  . "$ZDOTDIR/.zshrc"',
            "fi",
            'ZDOTDIR="$_agm_saved_zdotdir"',
            "unset _agm_saved_zdotdir",
            'eval "$(agm config env)"',
            'export SHELL="$AGM_WORKSPACE_SHELL"',
            "",
        ]
    )


def _bashrc_body() -> str:
    return "\n".join(
        [
            "# agm workspace shell: source the user's ~/.bashrc, then apply agm env",
            'if [ -f "$HOME/.bashrc" ]; then',
            '  . "$HOME/.bashrc"',
            "fi",
            'eval "$(agm config env)"',
            'export SHELL="$AGM_WORKSPACE_SHELL"',
            "",
        ]
    )


def _shrc_body() -> str:
    # Replay the user's original $ENV startup file, captured by the wrapper as
    # $AGM_USER_ENV before it overwrote $ENV to point at this file.  Sourcing
    # $ENV here would source this file recursively (it now points at itself),
    # exhausting file descriptors ("Too many open files"); use the saved path.
    return "\n".join(
        [
            "# agm workspace shell: source the user's sh rc, then apply agm env",
            'if [ -n "${AGM_USER_ENV:-}" ] && [ -f "$AGM_USER_ENV" ]; then',
            '  . "$AGM_USER_ENV"',
            "fi",
            'if [ -f "$HOME/.shrc" ]; then',
            '  . "$HOME/.shrc"',
            "fi",
            'eval "$(agm config env)"',
            'export SHELL="$AGM_WORKSPACE_SHELL"',
            "",
        ]
    )


def _heredoc_lines(path: str, body: str) -> list[str]:
    """Return sh lines that write *body* to *path* (regenerating a rc file).

    Uses a quoted heredoc so the body is written verbatim.  The delimiter is
    chosen to never appear inside *body*.
    """

    delimiter = "AGM_EOF"
    assert delimiter not in body
    return [
        f"mkdir -p \"$(dirname {path})\"",
        f"cat > {path} <<'{delimiter}'",
        body,
        delimiter,
    ]


def _wrapper_content(
    *,
    shell_dir: Path,
    wrapper_path: Path,
    real_shell: str,
) -> str:
    # Self-heal: if any rc file is missing (e.g. a partial deletion of the
    # cache dir), regenerate it inline before exec'ing the real shell.  The
    # wrapper is fully self-contained — it never shells out to `agm`.
    self_heal_lines: list[str] = [
        "# agm workspace shell: self-heal missing rc files in place.",
        (
            'if [ ! -f "$AGM_WORKSPACE_SHELL_DIR/zsh/.zshenv" ] '
            '|| [ ! -f "$AGM_WORKSPACE_SHELL_DIR/zsh/.zshrc" ] '
            '|| [ ! -f "$AGM_WORKSPACE_SHELL_DIR/bash/bashrc" ] '
            '|| [ ! -f "$AGM_WORKSPACE_SHELL_DIR/sh/shrc" ]; then'
        ),
    ]
    zshenv_body = _zshenv_body()
    zshrc_body = _zshrc_body()
    bashrc_body = _bashrc_body()
    shrc_body = _shrc_body()
    self_heal_lines += _heredoc_lines(
        '"$AGM_WORKSPACE_SHELL_DIR/zsh/.zshenv"', zshenv_body
    )
    self_heal_lines += _heredoc_lines(
        '"$AGM_WORKSPACE_SHELL_DIR/zsh/.zshrc"', zshrc_body
    )
    self_heal_lines += _heredoc_lines(
        '"$AGM_WORKSPACE_SHELL_DIR/bash/bashrc"', bashrc_body
    )
    self_heal_lines += _heredoc_lines(
        '"$AGM_WORKSPACE_SHELL_DIR/sh/shrc"', shrc_body
    )
    self_heal_lines += ["fi"]

    return "\n".join(
        [
            "#!/bin/sh",
            "# agm workspace shell wrapper",
            f"AGM_WORKSPACE_SHELL_DIR={shlex_quote(str(shell_dir))}",
            f"export AGM_WORKSPACE_SHELL={shlex_quote(str(wrapper_path))}",
            *self_heal_lines,
            (
                'if [ -z "${AGM_REAL_SHELL:-}" ] '
                '|| [ "$AGM_REAL_SHELL" = "$AGM_WORKSPACE_SHELL" ]; then'
            ),
            f'  AGM_REAL_SHELL={shlex_quote(real_shell)}',
            "fi",
            "export AGM_REAL_SHELL",
            'case "$(basename "$AGM_REAL_SHELL")" in',
            "  zsh)",
            # Save the user's real ZDOTDIR once (mirroring AGM_USER_ENV for sh)
            # before overwriting it, so the generated .zshenv/.zshrc replay the
            # user's own config dir.  The +set guard keeps the captured value
            # across nested workspace shells (where ZDOTDIR already points at an
            # outer workspace dir).
            '    if [ -z "${AGM_USER_ZDOTDIR+set}" ]; then',
            '      export AGM_USER_ZDOTDIR="${ZDOTDIR:-$HOME}"',
            "    fi",
            '    export ZDOTDIR="$AGM_WORKSPACE_SHELL_DIR/zsh"',
            '    exec "$AGM_REAL_SHELL" -i',
            "    ;;",
            "  bash)",
            '    exec "$AGM_REAL_SHELL" --rcfile "$AGM_WORKSPACE_SHELL_DIR/bash/bashrc" -i',
            "    ;;",
            "  *)",
            # Save the user's original $ENV once (even if empty) so the sh rc can
            # replay it; without this the rc would source $ENV — itself — forever.
            '    if [ -z "${AGM_USER_ENV+set}" ]; then',
            '      export AGM_USER_ENV="${ENV:-}"',
            "    fi",
            '    export ENV="$AGM_WORKSPACE_SHELL_DIR/sh/shrc"',
            '    exec "$AGM_REAL_SHELL" -i',
            "    ;;",
            "esac",
            "",
        ]
    )


def _write_shell_files(
    *,
    shell_dir: Path,
    wrapper_path: Path,
    real_shell: str,
) -> None:
    zsh_dir = shell_dir / "zsh"
    bash_dir = shell_dir / "bash"
    sh_dir = shell_dir / "sh"
    mkdir(zsh_dir, parents=True, exist_ok=True)
    mkdir(bash_dir, parents=True, exist_ok=True)
    mkdir(sh_dir, parents=True, exist_ok=True)
    write_text(zsh_dir / ".zshenv", _zshenv_body())
    write_text(zsh_dir / ".zshrc", _zshrc_body())
    write_text(bash_dir / "bashrc", _bashrc_body())
    write_text(sh_dir / "shrc", _shrc_body())
    write_text(
        wrapper_path,
        _wrapper_content(shell_dir=shell_dir, wrapper_path=wrapper_path, real_shell=real_shell),
    )
    chmod(wrapper_path, 0o755)


def regenerate_workspace_shell(shell_dir: Path) -> None:
    """Rewrite the rc files and wrapper into an existing *shell_dir*.

    Used by ``agm workspace shell-regen`` for manual recovery.  The content is
    session-independent, so the directory alone is enough; the real shell is
    read from ``$AGM_REAL_SHELL`` (set by the wrapper) or ``$SHELL``.
    """

    if not shell_dir.is_dir():
        raise SystemExit(f"error: not a directory: {shell_dir}")
    real_shell = _real_shell(env=None)
    wrapper_path = shell_dir / WRAPPER_NAME
    _write_shell_files(
        shell_dir=shell_dir,
        wrapper_path=wrapper_path,
        real_shell=real_shell,
    )


def ensure_workspace_shell(
    session_name: str,
    *,
    env: dict[str, str] | None = None,
) -> Path:
    """Create (or recreate) the per-session shell wrapper and return its path.

    Cleans any existing per-session dir first so stale files from a prior open
    cannot linger, then writes the wrapper and rc files fresh.  Returns the
    path to the executable wrapper script.
    """

    shell_dir = workspace_shell_dir(session_name)
    if shell_dir.exists():
        rmtree(shell_dir)
    mkdir(shell_dir, parents=True, exist_ok=True)
    wrapper_path = shell_dir / WRAPPER_NAME
    real_shell = _real_shell(env=env)
    _write_shell_files(
        shell_dir=shell_dir,
        wrapper_path=wrapper_path,
        real_shell=real_shell,
    )
    return wrapper_path


def remove_workspace_shell(session_name: str) -> None:
    """Remove the per-session shell wrapper directory, if present."""

    shell_dir = workspace_shell_dir(session_name)
    if shell_dir.exists():
        rmtree(shell_dir)
