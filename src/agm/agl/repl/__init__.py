"""AgL REPL package.

The UI-free incremental session core lives in :mod:`agm.agl.repl.session`.
Later milestones add the prompt_toolkit console, meta-command dispatch,
result rendering, and the confirming agent wrapper.
"""

from __future__ import annotations

from agm.agl.repl.session import EntryResult, ReplSession

__all__ = ["EntryResult", "ReplSession"]
