"""CLI helpers for mapping AgL ``param`` declarations to exec CLI options.

Each ``param`` declaration in a program becomes a ``--<scope-path>`` option
on ``agm exec``. The module-qualified spelling is accepted and is required
when two inventory params share their scope-path spelling. If ordinary
qualified positive and bool-negative spellings collide, an ``@module::`` prefix
marks an unambiguous qualified form. A file entry is rendered by its file-stem module
route rather than the pipeline's internal ``<entry>`` sentinel; inline entries
use ``@entry``. If that stem collides with an imported module route, the entry
also falls back to ``@entry``. Bool params use ``--name/--no-name`` flag form.
One selected option map drives parsing, help, and completion.

Collision detection is **verbatim**: a param whose name is ``foo`` produces the
flag ``--foo``; that exact string is checked against ``RESERVED_FLAGS``
(re-exported here from ``program_options``, the host's own flag inventory).
There is no underscore↔hyphen normalisation — the engine keys all use
kebab-case, so a param named ``timeout`` (the exact engine key name) collides,
but one named ``timeout_val`` does not.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

from agm.agl.modules.roots import RootSet
from agm.agl.runtime.request import AgentResponse
from agm.agl.runtime.types import (
    ENTRY_PARAM_QUALIFIER,
    ParamDeclInfo,
    public_param_spelling,
)
from agm.agl.semantics.types import BoolType
from agm.cli_support.program_options import RESERVED_FLAGS

if TYPE_CHECKING:
    from agm.cli_support.exec_target import PackageProgramReference


def param_flag(name: str) -> str:
    """Return the CLI flag for a param name (verbatim spelling preserved)."""
    return f"--{name}"


def negative_param_flag(name: str) -> str:
    """Return the negative CLI flag for a bool param (``--no-<name>``)."""
    return f"--no-{name}"


_QUALIFIED_PARAM_QUALIFIER = "@module"


def _external_qualified_name(param: ParamDeclInfo) -> str:
    """Return the canonical shell-safe qualified spelling for a parameter flag."""
    return public_param_spelling(param.qualified_name, entry_qualifier=param.entry_qualifier)


def _collision_safe_qualified_names(
    params: tuple[ParamDeclInfo, ...],
) -> dict[ParamDeclInfo, str]:
    """Return qualified spellings without an entry/module route collision.

    A symlinked file entry's resolved stem can equal an imported module route.
    Keep the imported module's canonical spelling and fall back to the existing
    shell-safe ``@entry`` identity for the entry instead of choosing a winner by
    inventory order.
    """
    canonical = tuple((param, _external_qualified_name(param)) for param in params)
    counts: dict[str, int] = {}
    for _param, spelling in canonical:
        counts[spelling] = counts.get(spelling, 0) + 1
    return {
        param: (
            f"{ENTRY_PARAM_QUALIFIER}::{param.name}"
            if param.is_entry and counts[spelling] > 1
            else spelling
        )
        for param, spelling in canonical
    }


type _FlagTarget = tuple[ParamDeclInfo, bool | None]
type _FlagBinding = tuple[str, ParamDeclInfo, bool | None]


@dataclass(frozen=True)
class _ParamFlagMap:
    """One inventory's shared parsing, completion, and help option map."""

    bindings: tuple[_FlagBinding, ...]
    preferred: Mapping[ParamDeclInfo, tuple[_FlagBinding, ...]]
    ambiguous: Mapping[str, tuple[_FlagTarget, ...]]


def _flag_candidates(param: ParamDeclInfo, spelling: str) -> tuple[_FlagBinding, ...]:
    """Return the positive and, when applicable, negative flag for *spelling*."""
    positive = (param_flag(spelling), param, True if isinstance(param.type, BoolType) else None)
    if isinstance(param.type, BoolType):
        return positive, (negative_param_flag(spelling), param, False)
    return (positive,)


def _add_candidates(
    candidates: dict[str, list[_FlagTarget]], bindings: tuple[_FlagBinding, ...]
) -> None:
    """Add bindings to a flag candidate index, deduplicating identical targets."""
    for flag, param, value in bindings:
        target = (param, value)
        matches = candidates.setdefault(flag, [])
        if target not in matches:
            matches.append(target)


def _ambiguous_short_flags(
    params: tuple[ParamDeclInfo, ...],
) -> dict[str, tuple[ParamDeclInfo, ...]]:
    """Return short spellings that cannot safely select a param."""
    candidates: dict[str, list[_FlagTarget]] = {}
    for param in params:
        _add_candidates(candidates, _flag_candidates(param, param.name))
    return {
        flag: tuple(param for param, _value in matches)
        for flag, matches in candidates.items()
        if len(matches) > 1 or flag in RESERVED_FLAGS
    }


def external_param_keys(params: tuple[ParamDeclInfo, ...]) -> dict[ParamDeclInfo, str]:
    """Return each param's external key: its CLI flag/config-table spelling.

    A param's own name doubles as its external key everywhere — the CLI flag
    ``--<name>`` and the config-table key ``<name>`` — unless that spelling is
    ambiguous: shared with another param's short spelling, or colliding with a
    reserved built-in flag (``RESERVED_FLAGS``). An ambiguous param's external
    key is always its module-qualified spelling, so every layer that reads or
    writes its value (CLI flags, config tables) agrees on one key regardless
    of how the value was supplied. This is the single rule shared by CLI flag
    parsing (:func:`parse_param_tokens`) and qualified config-key resolution.
    """
    ambiguous_params = {
        param for candidates in _ambiguous_short_flags(params).values() for param in candidates
    }
    return {
        param: param.qualified_name if param in ambiguous_params else param.name for param in params
    }


def _escaped_qualified_name(qualified_name: str) -> str:
    """Return a spelling in the collision-free ``@module`` namespace."""
    return f"{_QUALIFIED_PARAM_QUALIFIER}::{qualified_name}"


def _form_is_safe(
    bindings: tuple[_FlagBinding, ...], candidates: Mapping[str, list[_FlagTarget]]
) -> bool:
    """Return whether every binding in one spelling form uniquely identifies its target."""
    return all(
        flag not in RESERVED_FLAGS and candidates[flag] == [(param, value)]
        for flag, param, value in bindings
    )


def _param_flag_map(params: tuple[ParamDeclInfo, ...]) -> _ParamFlagMap:
    """Select one canonical option form per param and every safe convenience alias.

    Short and ordinary module-qualified forms remain available when they name
    exactly one target across the complete inventory. If neither complete form
    is safe, an ``@module::`` prefix marks the qualified positive namespace;
    its bool negative is consequently ``--no-@module::...``. AgL identifiers
    cannot contain ``@``, so the two polarities cannot collide with another
    param spelling.
    """
    qualified_names = _collision_safe_qualified_names(params)
    short_forms = {param: _flag_candidates(param, param.name) for param in params}
    qualified_forms = {param: _flag_candidates(param, qualified_names[param]) for param in params}
    escaped_forms = {
        param: _flag_candidates(param, _escaped_qualified_name(qualified_names[param]))
        for param in params
    }

    convenience_candidates: dict[str, list[_FlagTarget]] = {}
    escaped_candidates: dict[str, list[_FlagTarget]] = {}
    for param in params:
        _add_candidates(convenience_candidates, short_forms[param])
        _add_candidates(convenience_candidates, qualified_forms[param])
        _add_candidates(escaped_candidates, escaped_forms[param])

    preferred: dict[ParamDeclInfo, tuple[_FlagBinding, ...]] = {}
    for param in params:
        if _form_is_safe(short_forms[param], convenience_candidates):
            preferred[param] = short_forms[param]
        elif _form_is_safe(qualified_forms[param], convenience_candidates):
            preferred[param] = qualified_forms[param]
        elif "::" in qualified_names[param] and _form_is_safe(
            escaped_forms[param], escaped_candidates
        ):
            preferred[param] = escaped_forms[param]

    selected: dict[str, _FlagTarget] = {}
    for param in params:
        for binding in (*qualified_forms[param], *short_forms[param]):
            flag, candidate_param, value = binding
            if flag not in RESERVED_FLAGS and convenience_candidates[flag] == [
                (candidate_param, value)
            ]:
                selected[flag] = (candidate_param, value)
        for flag, candidate_param, value in preferred.get(param, ()):
            selected[flag] = (candidate_param, value)

    all_candidates = {**convenience_candidates, **escaped_candidates}
    ambiguous = {
        flag: tuple(targets)
        for flag, targets in all_candidates.items()
        if len(targets) > 1 or flag in RESERVED_FLAGS
    }
    return _ParamFlagMap(
        bindings=tuple((flag, param, value) for flag, (param, value) in selected.items()),
        preferred=preferred,
        ambiguous=ambiguous,
    )


def param_option_flags(params: tuple[ParamDeclInfo, ...]) -> tuple[str, ...]:
    """Return the selected unambiguous CLI flags for *params*."""
    return tuple(flag for flag, _param, _bool_value in _param_flag_map(params).bindings)


def param_value_taking_flags(params: tuple[ParamDeclInfo, ...]) -> frozenset[str]:
    """Return every selected flag from *params* that consumes a following ``VALUE`` token.

    One half of the *value_flags*
    :func:`~agm.cli_support.program_options.short_help_requested` checks
    against, the other half being a selected program's own value parameters,
    projected through
    :meth:`~agm.cli_support.program_options.ProgramOptionMap.value_taking_flags`.
    """
    return frozenset(
        flag for flag, _param, bool_value in _param_flag_map(params).bindings if bool_value is None
    )


def discover_params_from_source(
    source: str,
    *,
    inline_source: bool = False,
    entry_path: Path | None = None,
    roots: RootSet | None = None,
    default_stdlib: bool = True,
) -> tuple[ParamDeclInfo, ...]:
    """Discover declared params from AgL *source*, degrading to ``()`` on error.

    Shared by the help and shell-completion paths, which both need only the
    discovered params and must tolerate unreadable/unparsable sources.
    Inline sources receive the same pure synthetic-main wrapper as ``agm exec
    -c``; file sources retain their ordinary unwrapped behavior. Supplying
    their *entry_path* lets the loader discover imports relative to that
    file.
    """
    try:
        from dataclasses import replace

        from agm.agl import PipelineDriver

        runtime = PipelineDriver(agent_dispatcher=lambda request: AgentResponse(content=""))
        if inline_source:
            parsed = runtime.parse_entry(source)
            if parsed.program is not None:
                from agm.agl.parser import wrap_inline_program

                program, next_id = wrap_inline_program(parsed.program, next_node_id=parsed.next_id)
                parsed = replace(parsed, program=program, next_id=next_id)
            prepared = runtime.prepare_parsed_entry(
                parsed, roots=roots, default_stdlib=default_stdlib
            )
        else:
            prepared = runtime.prepare_program(
                source, entry_path=entry_path, roots=roots, default_stdlib=default_stdlib
            )
        return runtime.discover_params(prepared).params
    except (Exception, SystemExit):
        return ()


def discover_params_from_installed_reference(
    resolved: "PackageProgramReference",
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    default_stdlib: bool = True,
) -> tuple[ParamDeclInfo, ...]:
    """Discover params for an already-resolved active package program reference.

    Help and completion resolve ``PACKAGE/MODULE::PROGRAM`` through the same
    active package selection as execution, then pass the result here. They
    are advisory surfaces, so an unreadable entry or a program that no
    longer parses degrades to no params.
    """
    try:
        from agm.cli_support.exec_roots import effective_exec_roots

        source = resolved.entry_path.read_text(encoding="utf-8")
        exec_roots = effective_exec_roots(
            entry_path=resolved.entry_path,
            module_paths=[],
            cwd=cwd,
            home=home,
            proj_dir=proj_dir,
        )
        return discover_params_from_source(
            source,
            entry_path=resolved.entry_path,
            roots=exec_roots.roots,
            default_stdlib=default_stdlib,
        )
    except (Exception, SystemExit):
        return ()


@overload
def parse_param_tokens(
    params: tuple[ParamDeclInfo, ...],
    tokens: list[str],
) -> dict[str, object]: ...


@overload
def parse_param_tokens(
    params: tuple[ParamDeclInfo, ...],
    tokens: list[str],
    *,
    collect_leftovers: Literal[True],
) -> tuple[dict[str, object], list[str]]: ...


def parse_param_tokens(
    params: tuple[ParamDeclInfo, ...],
    tokens: list[str],
    *,
    collect_leftovers: bool = False,
) -> dict[str, object] | tuple[dict[str, object], list[str]]:
    """Parse leftover CLI tokens into a param value dict.

    Returns a ``dict[str, object]`` mapping param names to their values:
    - bool params: native ``bool`` (``True`` for ``--name``, ``False`` for ``--no-name``)
    - all others: raw ``str`` (runtime ``convert_param_value`` handles type coercion)

    Raises ``ValueError`` for:
    - Unexpected positional or short-option tokens (unless *collect_leftovers*)
    - Unknown ``--xxx`` flags (unless *collect_leftovers*)
    - Missing value for a non-bool flag
    - Duplicate param flags

    With *collect_leftovers*, a token that names no declared param — a bare
    positional argument, or an unrecognized ``--xxx`` flag — is set aside into
    a leftover-token list instead of raising, and the return becomes ``(values,
    leftover_tokens)``. This lets callers reuse this function's own flag
    pairing and ambiguity-repair logic to split raw CLI tokens between
    declared params and a program's own value-parameter options, without a
    second, independently written token router. A token that *does* name a
    declared param is still fully validated: ambiguous spellings, missing
    values, and duplicate supplies still raise.
    """
    external_keys = external_param_keys(params)
    selected = _param_flag_map(params)
    flag_to_param = {flag: (param, bool_value) for flag, param, bool_value in selected.bindings}

    def ambiguity_repair(flag: str) -> str:
        """Render the selected canonical equivalent for every ambiguous target."""
        repairs: list[str] = []
        for param, value in selected.ambiguous[flag]:
            preferred = selected.preferred.get(param, ())
            repair = next(
                (
                    preferred_flag
                    for preferred_flag, _param, preferred_value in preferred
                    if preferred_value == value
                ),
                None,
            )
            if repair is not None:
                repairs.append(repair)
        return ", ".join(repairs)

    result: dict[str, object] = {}
    leftovers: list[str] = []

    def store(param: ParamDeclInfo, value: object) -> None:
        """Record *value* for *param*, rejecting a flag repeated for the same param."""
        key = external_keys[param]
        if key in result:
            raise ValueError(f"Option '{param_flag(param.name)}' specified more than once")
        result[key] = value

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            if collect_leftovers:
                leftovers.append(token)
                i += 1
                continue
            raise ValueError(f"Unexpected argument: {token!r}")

        # ``--name=value`` and ``--name`` share one flag lookup; only the
        # trailing value source differs.
        flag, has_equals, inline_value = token.partition("=")
        if flag in selected.ambiguous:
            raise ValueError(f"Option {flag!r} is ambiguous; use one of: {ambiguity_repair(flag)}")
        if flag not in flag_to_param:
            if collect_leftovers:
                leftovers.append(token)
                i += 1
                continue
            raise ValueError(f"Unknown option: {flag!r}")
        param, bool_val = flag_to_param[flag]

        if has_equals:
            if bool_val is not None:
                raise ValueError(f"Option {flag!r} does not take a value")
            store(param, inline_value)
            i += 1
        elif bool_val is not None:
            # Bool flag: --name → True, --no-name → False.
            store(param, bool_val)
            i += 1
        else:
            # Value-taking flag: next token is the value.
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                raise ValueError(f"Option {token!r} requires a value")
            store(param, tokens[i + 1])
            i += 2

    if collect_leftovers:
        return result, leftovers
    return result


def render_param_help_section(params: tuple[ParamDeclInfo, ...]) -> str:
    """Render the 'Program parameters:' help section for ``--help`` output.

    Returns a string starting with ``'Program parameters:\\n'`` followed by one
    line per param, or an empty string when there are no params.
    """
    if not params:
        return ""
    selected = _param_flag_map(params)
    lines: list[str] = ["Program parameters:"]
    for p in params:
        preferred = selected.preferred.get(p)
        if preferred is None:
            continue
        if isinstance(p.type, BoolType):
            flag_str = "/".join(flag for flag, _param, _value in preferred)
        else:
            flag_str = f"{preferred[0][0]} {p.type.kind.upper()}"
        req_str = "(required)" if not p.has_default else "(optional, has default)"
        lines.append(f"  {flag_str}  {req_str}")
    return "\n".join(lines) + "\n"
