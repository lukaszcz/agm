"""WorkflowRuntime — the public façade for the AgL host runtime.

M0 shell: the constructor, ``register_agent``, and a ``run`` method that
always returns a pre-execution failure result (execution is not yet
implemented).  Later milestones fill in the full pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from agm.agl.diagnostics import Diagnostic

# Reserved agent names: cannot be registered by callers.
_RESERVED_AGENT_NAMES: frozenset[str] = frozenset({"prompt", "exec"})

# Type alias for an agent callable (relaxed for M0; tightened in M1).
AgentFn = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class RunError:
    """Structured representation of an uncaught AgL exception.

    ``type_name`` is the exception's declared type name (e.g. ``"AgentParseError"``).
    ``fields`` is a mapping from field names to JSON-shaped Python values.
    """

    type_name: str
    fields: dict[str, object]


@dataclass(slots=True)
class RunResult:
    """Result of a ``WorkflowRuntime.run`` call.

    ``ok``
        ``True`` iff there are no error-severity diagnostics **and** no
        uncaught AgL exception.  Warning-severity diagnostics (see
        ``Diagnostic.severity``) may be present while ``ok`` is ``True``.
    ``diagnostics``
        Pre-execution diagnostics (lex/parse/scope/typecheck/input-validation).
        Each entry has a ``.message`` (str), a ``.line`` (int, 1-based) and a
        ``.severity`` (``"error"`` or ``"warning"``).  When ``ok`` is ``True``
        any entries are warnings only.
    ``error``
        The uncaught AgL exception, or ``None``.  Set only when the program
        *started* executing but ended with an unhandled exception (exit code 2
        per the CLI contract).  ``None`` for pre-execution failures and for
        successful runs.
    """

    ok: bool
    diagnostics: list[Diagnostic]
    error: RunError | None


class WorkflowRuntime:
    """Host API for the AgL interpreter.

    Constructor parameters
    ----------------------
    default_loop_limit : int
        Default iteration bound for ``do[N]`` loops (design §2.11).
    default_strict_json : bool
        When ``True`` the JSON codec defaults to strict parsing (only a bare
        JSON value with surrounding whitespace is accepted).  The default
        ``False`` enables lenient JSON recovery (design §2.8, Q3).
    default_agent : callable or None
        The callable used for the built-in ``prompt`` agent.  ``None`` means
        no default agent is configured (only explicitly registered agents will
        be available).
    """

    def __init__(
        self,
        *,
        default_loop_limit: int = 5,
        default_strict_json: bool = False,
        default_agent: AgentFn | None = None,
    ) -> None:
        self._default_loop_limit = default_loop_limit
        self._default_strict_json = default_strict_json
        self._default_agent = default_agent
        self._agents: dict[str, AgentFn] = {}

    def register_agent(self, name: str, fn: AgentFn) -> None:
        """Register a named agent callable.

        Raises ``ValueError`` if ``name`` is a reserved name (``prompt`` or
        ``exec``) or if an agent with that name has already been registered.
        """
        if name in _RESERVED_AGENT_NAMES:
            raise ValueError(
                f"Cannot register agent with reserved name {name!r}. "
                f"Reserved names: {sorted(_RESERVED_AGENT_NAMES)}"
            )
        if name in self._agents:
            raise ValueError(
                f"An agent named {name!r} is already registered. "
                "Duplicate registrations are not allowed."
            )
        self._agents[name] = fn

    def run(self, source: str, *, inputs: Mapping[str, object] | None = None) -> RunResult:
        """Parse, analyse, and execute an AgL source program.

        M0 implementation: always returns a pre-execution failure with a
        single diagnostic explaining that execution is not yet implemented.
        The full pipeline (lex → parse → scope → typecheck → eval) is wired
        in M1–M5.
        """
        del source, inputs  # unused in M0
        diagnostic = Diagnostic(
            message="AgL execution is not implemented yet (M0 skeleton)",
            line=1,
        )
        return RunResult(ok=False, diagnostics=[diagnostic], error=None)

    @property
    def default_loop_limit(self) -> int:
        """Default iteration bound for ``do[N]`` loops."""
        return self._default_loop_limit

    @property
    def default_strict_json(self) -> bool:
        """Whether strict JSON parsing is the default."""
        return self._default_strict_json
