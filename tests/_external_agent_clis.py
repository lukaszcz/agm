"""External agent and sandbox CLIs that tests must never run for real."""

from __future__ import annotations

# ``srt`` is included so sandboxed runs never reach a real sandbox runtime.
EXTERNAL_AGENT_CLIS: tuple[str, ...] = ("claude", "codex", "opencode", "pi", "srt")
