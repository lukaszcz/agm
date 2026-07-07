"""Pure result data types for one REPL entry evaluation (`EntryResult`, `EntryKind`).

Shared by the session and its graph-mode collaborator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agm.agl.diagnostics import Diagnostic

if TYPE_CHECKING:
    from pathlib import Path

    from agm.agl.pipeline import RunError
    from agm.agl.semantics.type_table import TypeTable
    from agm.agl.semantics.types import Type
    from agm.agl.semantics.values import Value


EntryKind = Literal["expression", "binding", "declaration", "statement", "type"]


@dataclass(frozen=True, slots=True)
class EntryResult:
    """Outcome of evaluating one REPL entry (pure data, no styled strings).

    ``kind``
        Classified by the entry's LAST item: a bare ``Expr`` → ``"expression"``
        (``value``/``value_type`` set); ``let``/``var`` → ``"binding"``
        (``name``/``value_type``/``value``); ``record``/``enum``/``type``/
        ``param``/``def``/``agent`` → ``"declaration"``; ``:=`` or side-
        effecting expr (``print``, etc.) → ``"statement"``; a REPL-only bare
        type expression (``int``, a declared type name, ``list[T]``) →
        ``"type"`` (``value_type`` set, no value, no state change).
    ``name``
        The bound/declared name, when meaningful (binding / declaration).
    ``value``
        The echoed runtime value (expression value or new binding value); ``None``
        for declarations, statements, ``check_only`` runs, and failures.
    ``value_type``
        The static type of the echoed value; ``None`` when not applicable.
    ``type_display``
        Pre-rendered type-focused display text for REPL-only type entries that
        are definitions rather than concrete semantic ``Type`` instances (for
        example a bare generic record or enum name).
    ``type_table``
        The shared ``TypeTable`` that resolves nominal record / enum shapes for
        ``value_type``; supplied whenever a type-focused echo may expand a
        record or enum handle to its field / constructor declarations.
    ``diagnostics``
        Pre-execution error diagnostics (parse/scope/typecheck/contract/unset
        param).  Empty on success.
    ``warnings``
        Advisory warnings from the type checker (e.g. non-exhaustive ``case``),
        surfaced on every non-parse/scope path.
    ``error``
        The uncaught AgL exception mapped to a ``RunError`` when the entry raised
        during evaluation; ``None`` otherwise.
    ``ok``
        ``True`` iff there are no error diagnostics AND no runtime error.
    ``trace_path``
        Path of the JSONL trace file the entry's records were appended to, or
        ``None`` when tracing is disabled (no ``--log-file``) or for a
        ``check_only`` (dry-run) entry, which writes no trace.
    ``installed``
        Names installed before a failed entry stopped. Empty for pre-execution
        failures and successful entries.
    ``quote_strings``
        Whether REPL echo should quote a top-level text value. This is normally
        ``True``. The only exception is a standalone ``ask`` builtin entry,
        whose response is echoed as display text rather than as an AgL string
        literal.
    """

    kind: EntryKind
    name: str | None
    value: "Value | None"
    value_type: "Type | None"
    diagnostics: list[Diagnostic]
    warnings: list[Diagnostic]
    error: "RunError | None"
    ok: bool
    trace_path: "Path | None" = None
    installed: tuple[str, ...] = ()
    quote_strings: bool = True
    type_display: str | None = None
    type_table: "TypeTable | None" = None
