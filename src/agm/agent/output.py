"""Helpers for formatting agent command output."""

from __future__ import annotations

from datetime import datetime


def step_header_text(step: int) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = f"Step {step}  ({now})"
    sep = "-" * 61
    return (
        "\n"
        f"{sep}\n"
        f"{label.center(61)}\n"
        f"{sep}\n"
        "\n"
    )
