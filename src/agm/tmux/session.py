"""Tmux session creation."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from agm.core import dry_run
from agm.parser import exit_with_usage_error
from agm.tmux.layout import apply_layout

SKIP_NAMES: set[str] = {
    "TMUX",
    "TERM",
    "DISPLAY",
    "PWD",
    "OLDPWD",
    "SHELL",
    "SHLVL",
    "LOGNAME",
    "USER",
    "_",
}

SKIP_PREFIXES: tuple[str, ...] = ("TMUX_", "TERM_", "SSH_", "DBUS_", "XDG_")


def validate_pane_count(command_path: list[str], pane_count: str | None) -> int:
    """Validate an optional pane count and return the effective total."""

    if pane_count is None:
        return 4
    if not pane_count.isdigit() or int(pane_count) < 1:
        exit_with_usage_error(command_path, f"Invalid pane count: {pane_count}")
    return int(pane_count)


def _filter_env(env: dict[str, str]) -> list[tuple[str, str]]:
    filtered: list[tuple[str, str]] = []
    for name, value in env.items():
        if name in SKIP_NAMES:
            continue
        if any(name.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        filtered.append((name, value))
    return filtered


def create_tmux_session(
    *,
    detach: bool,
    pane_count: str | None,
    session_name: str | None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str | None:
    """Create a tmux session matching tmux.sh semantics."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    resolved_env = dict(os.environ if env is None else env)
    pane_total = validate_pane_count(["tmux", "open"], pane_count)

    tmux_env_args: list[str] = []
    for name, value in _filter_env(resolved_env):
        tmux_env_args.extend(["-e", f"{name}={value}"])

    create_detached_session = detach
    switch_to_session = False
    if resolved_env.get("TMUX") and not detach:
        create_detached_session = True
        switch_to_session = True

    session_name_args: list[str] = []
    if session_name:
        session_name_args = ["-s", session_name]

    if create_detached_session:
        if dry_run.enabled():
            planned_session = session_name or "0"
            dry_run.print_command(
                [
                    "tmux",
                    "new-session",
                    "-dP",
                    "-F",
                    "#{session_name}",
                    "-c",
                    str(current),
                    *tmux_env_args,
                    *session_name_args,
                ],
                cwd=current,
            )
            for _ in range(1, pane_total):
                dry_run.print_command(
                    [
                        "tmux",
                        "split-window",
                        "-d",
                        "-h",
                        "-t",
                        f"{planned_session}:0",
                        "-c",
                        str(current),
                    ],
                    cwd=current,
                )
            dry_run.print_operation("tmux-layout", f"{planned_session}:0 {pane_total} panes")
            dry_run.print_command(
                ["tmux", "select-pane", "-t", f"{planned_session}:0.0"],
                cwd=current,
            )
            if switch_to_session:
                dry_run.print_command(
                    ["tmux", "switch-client", "-t", planned_session],
                    cwd=current,
                )
            elif not detach:
                dry_run.print_command(
                    ["tmux", "attach-session", "-t", planned_session],
                    cwd=current,
                )
            print(f"Detached tmux session {planned_session} created")
            return planned_session
        result = subprocess.run(
            [
                "tmux",
                "new-session",
                "-dP",
                "-F",
                "#{session_name}",
                "-c",
                str(current),
                *tmux_env_args,
                *session_name_args,
            ],
            capture_output=True,
            text=True,
            cwd=current,
            env=resolved_env,
            check=False,
        )
        if result.returncode != 0:
            if result.stdout:
                sys.stdout.write(result.stdout)
            if result.stderr:
                sys.stderr.write(result.stderr)
            raise SystemExit(result.returncode)

        target_session = result.stdout.strip()
        for _ in range(1, pane_total):
            status = subprocess.run(
                [
                    "tmux",
                    "split-window",
                    "-d",
                    "-h",
                    "-t",
                    f"{target_session}:0",
                    "-c",
                    str(current),
                ],
                cwd=current,
                env=resolved_env,
                check=False,
            ).returncode
            if status != 0:
                raise SystemExit(status)

        def _display(format_string: str) -> str:
            display = subprocess.run(
                ["tmux", "display-message", "-p", "-t", f"{target_session}:0", format_string],
                capture_output=True,
                text=True,
                cwd=current,
                env=resolved_env,
                check=False,
            )
            if display.returncode != 0:
                if display.stdout:
                    sys.stdout.write(display.stdout)
                if display.stderr:
                    sys.stderr.write(display.stderr)
                raise SystemExit(display.returncode)
            return display.stdout.strip()

        apply_layout(
            pane_count=pane_total,
            window_id=_display("#{window_id}"),
            width=int(_display("#{window_width}")),
            height=int(_display("#{window_height}")),
        )
        status = subprocess.run(
            ["tmux", "select-pane", "-t", f"{target_session}:0.0"],
            cwd=current,
            env=resolved_env,
            check=False,
        ).returncode
        if status != 0:
            raise SystemExit(status)
        if switch_to_session:
            raise SystemExit(
                focus_tmux_session(
                    session_name=target_session,
                    cwd=current,
                    env=resolved_env,
                )
            )
        print(f"Detached tmux session {target_session} created")
        return target_session

    layout_command = " ".join(
        [
            shlex.quote(sys.executable),
            "-m",
            "agm.cli",
            "tmux",
            "layout",
            str(pane_total),
        ],
    )

    args = ["tmux", "new-session", "-c", str(current), *tmux_env_args, *session_name_args]
    for _ in range(1, pane_total):
        args.extend([";", "split-window", "-d", "-h", "-c", str(current)])
    args.extend([";", "run-shell", layout_command, ";", "select-pane", "-t", "0"])
    if dry_run.enabled():
        dry_run.print_command(args, cwd=current)
        return None
    raise SystemExit(subprocess.run(args, cwd=current, env=resolved_env, check=False).returncode)


def queue_command_in_session(
    *,
    session_name: str,
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Queue a shell command in the first pane of a tmux session."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    resolved_env = dict(os.environ if env is None else env)
    shell_command = " ".join(shlex.quote(part) for part in command)
    if dry_run.enabled():
        dry_run.print_command(
            ["tmux", "send-keys", "-t", f"{session_name}:0.0", shell_command, "C-m"],
            cwd=current,
        )
        return
    status = subprocess.run(
        ["tmux", "send-keys", "-t", f"{session_name}:0.0", shell_command, "C-m"],
        cwd=current,
        env=resolved_env,
        check=False,
    ).returncode
    if status != 0:
        raise SystemExit(status)


def focus_tmux_session(
    *,
    session_name: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Attach or switch to an existing tmux session."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    resolved_env = dict(os.environ if env is None else env)
    command = (
        ["tmux", "switch-client", "-t", session_name]
        if resolved_env.get("TMUX")
        else ["tmux", "attach-session", "-t", session_name]
    )
    if dry_run.enabled():
        dry_run.print_command(command, cwd=current)
        return 0
    return subprocess.run(command, cwd=current, env=resolved_env, check=False).returncode


def kill_tmux_session(
    *,
    session_name: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Kill an existing tmux session."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    resolved_env = dict(os.environ if env is None else env)
    if dry_run.enabled():
        dry_run.print_command(["tmux", "kill-session", "-t", session_name], cwd=current)
        return 0
    return subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        cwd=current,
        env=resolved_env,
        check=False,
    ).returncode


def close_tmux_session(
    *,
    session_name: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Kill an existing tmux session and report success."""

    status = kill_tmux_session(session_name=session_name, cwd=cwd, env=env)
    if status != 0:
        raise SystemExit(status)
    print(f"Closed session {session_name}")
