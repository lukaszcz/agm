"""CLI helpers for mapping AgL ``param`` declarations to exec CLI options.

Each ``param`` declaration in a program becomes a ``--<scope-path>`` option
on ``agm exec``. The module-qualified spelling is always accepted; it is
required when two inventory params share their scope-path spelling. Bool
params use ``--name/--no-name`` flag form. This module provides pure,
unit-testable functions used by both the exec command and the help/completion
machinery.

Collision detection is **verbatim**: a param whose name is ``foo`` produces the
flag ``--foo``; that exact string is checked against ``RESERVED_FLAGS``.  There
is no underscore↔hyphen normalisation — the engine keys all use kebab-case, so
a param named ``timeout`` (the exact engine key name) collides, but one named
``timeout_val`` does not.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from agm.agl.modules.roots import RootSet
from agm.agl.runtime.request import AgentResponse
from agm.agl.runtime.types import ParamDeclInfo
from agm.agl.semantics.types import BoolType


def _build_engine_key_flags() -> frozenset[str]:
    """Derive the set of reserved CLI flag strings from the engine-key registry.

    For each engine key:
    - Always adds ``--<name>`` (positive flag).
    - Adds ``--no-<name>`` for bool-typed keys and Option-typed keys (which have
      an explicit ``--no-<name>`` negation to set the binding to ``none``).

    Derived at import time from ``ENGINE_KEY_NAMES`` + ``get_engine_key_type``
    so that adding a new engine key automatically appears here.
    """
    from agm.agl.semantics.engine_keys import ENGINE_KEY_NAMES, get_engine_key_type
    from agm.agl.semantics.types import EnumType

    flags: set[str] = set()
    for name in ENGINE_KEY_NAMES:
        flags.add(f"--{name}")
        key_type = get_engine_key_type(name)
        # Both bool keys and Option[T] keys have a ``--no-<name>`` counterpart.
        if key_type is not None and isinstance(key_type, (BoolType, EnumType)):
            flags.add(f"--no-{name}")
    return frozenset(flags)


# Non-engine built-in flags that are always reserved on ``agm exec``.
_BUILTIN_EXEC_FLAGS: frozenset[str] = frozenset(
    {
        "--command",
        "-c",
        "--program",
        "-p",
        "--module-path",
        "-I",
        "--max-call-depth",
        "--help",
        "-h",
        "--dry-run",
        "--no-stdlib",
    }
)

# Reserved flag strings: non-engine built-ins UNION engine-key flags (both polarities).
# Collision check is verbatim — no underscore↔hyphen normalisation.
RESERVED_FLAGS: frozenset[str] = _BUILTIN_EXEC_FLAGS | _build_engine_key_flags()


def param_flag(name: str) -> str:
    """Return the CLI flag for a param name (verbatim spelling preserved)."""
    return f"--{name}"


def negative_param_flag(name: str) -> str:
    """Return the negative CLI flag for a bool param (``--no-<name>``)."""
    return f"--no-{name}"


def _add_flag_candidates(
    candidates: dict[str, list[tuple[ParamDeclInfo, bool | None]]],
    param: ParamDeclInfo,
    spelling: str,
) -> None:
    """Add the positive and, when applicable, negative flag for *spelling*."""
    if isinstance(param.type, BoolType):
        candidates.setdefault(param_flag(spelling), []).append((param, True))
        candidates.setdefault(negative_param_flag(spelling), []).append((param, False))
    else:
        candidates.setdefault(param_flag(spelling), []).append((param, None))


def _ambiguous_short_flags(
    params: tuple[ParamDeclInfo, ...],
) -> dict[str, tuple[ParamDeclInfo, ...]]:
    """Return short spellings that cannot safely select a param.

    A short spelling is ambiguous both when several params generate it and when
    it belongs to AGM itself. The latter still has one param candidate, but it
    cannot be parsed as a param option by Click, so callers must use the
    module-qualified spelling.
    """
    candidates: dict[str, list[tuple[ParamDeclInfo, bool | None]]] = {}
    for param in params:
        _add_flag_candidates(candidates, param, param.name)
    return {
        flag: tuple(param for param, _value in matches)
        for flag, matches in candidates.items()
        if len(matches) > 1 or flag in RESERVED_FLAGS
    }


def _param_value_name(
    param: ParamDeclInfo, ambiguous_short_flags: Mapping[str, tuple[ParamDeclInfo, ...]]
) -> str:
    """Return the external key for *param*, qualifying ambiguous leaves."""
    if any(param in candidates for candidates in ambiguous_short_flags.values()):
        return param.qualified_name
    return param.name


def _param_flag_selection(
    params: tuple[ParamDeclInfo, ...],
) -> tuple[tuple[str, ParamDeclInfo, bool | None], ...]:
    """Return the valid, unambiguous parameter flags for one inventory."""
    ambiguous = _ambiguous_short_flags(params)
    flags: dict[str, tuple[ParamDeclInfo, bool | None]] = {}
    for param in params:
        qualified_candidates: dict[str, list[tuple[ParamDeclInfo, bool | None]]] = {}
        _add_flag_candidates(qualified_candidates, param, param.qualified_name)
        flags.update({flag: candidates[0] for flag, candidates in qualified_candidates.items()})
        short_candidates: dict[str, list[tuple[ParamDeclInfo, bool | None]]] = {}
        _add_flag_candidates(short_candidates, param, param.name)
        flags.update(
            {
                flag: candidates[0]
                for flag, candidates in short_candidates.items()
                if flag not in ambiguous
            }
        )
    return tuple((flag, param, bool_value) for flag, (param, bool_value) in flags.items())


def param_option_flags(params: tuple[ParamDeclInfo, ...]) -> tuple[str, ...]:
    """Return valid CLI flags for *params*, without ambiguous short spellings."""
    return tuple(flag for flag, _param, _bool_value in _param_flag_selection(params))


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
    discovered params and must tolerate unreadable/unparsable sources. Inline
    sources receive the same pure synthetic-main wrapper as ``agm exec -c``;
    file sources retain their ordinary unwrapped behavior. Supplying their
    *entry_path* lets the loader discover imports relative to that file.
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
            prepared = (
                runtime.prepare_parsed_entry(parsed, default_stdlib=default_stdlib)
                if roots is None
                else runtime.prepare_parsed_entry(
                    parsed, roots=roots, default_stdlib=default_stdlib
                )
            )
        else:
            prepared = (
                runtime.prepare_program(
                    source, entry_path=entry_path, default_stdlib=default_stdlib
                )
                if roots is None
                else runtime.prepare_program(
                    source, entry_path=entry_path, roots=roots, default_stdlib=default_stdlib
                )
            )
        return runtime.discover_params(prepared).params
    except (Exception, SystemExit):
        return ()


def parse_param_tokens(
    params: tuple[ParamDeclInfo, ...],
    tokens: list[str],
) -> dict[str, object]:
    """Parse leftover CLI tokens into a param value dict.

    Returns a ``dict[str, object]`` mapping param names to their values:
    - bool params: native ``bool`` (``True`` for ``--name``, ``False`` for ``--no-name``)
    - all others: raw ``str`` (runtime ``convert_param_value`` handles type coercion)

    Non-option tokens (not starting with ``--``) are silently skipped so that
    the FILE positional argument landing in ``ctx.args`` does not cause errors.

    Raises ``ValueError`` for:
    - Unknown ``--xxx`` flags
    - Missing value for a non-bool flag
    - Duplicate param flags
    """
    # Qualified flags always participate; unsafe short flags remain in the
    # ambiguity map solely to produce an actionable diagnostic.
    ambiguous_flags = _ambiguous_short_flags(params)
    flag_to_param = {
        flag: (param, bool_value) for flag, param, bool_value in _param_flag_selection(params)
    }

    result: dict[str, object] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue  # skip positional tokens (e.g. FILE)

        # Handle ``--name=value`` form.
        if "=" in token:
            flag, _, value = token.partition("=")
            if flag in ambiguous_flags:
                spellings = ", ".join(
                    param_flag(param.qualified_name) for param in ambiguous_flags[flag]
                )
                raise ValueError(f"Option {flag!r} is ambiguous; use one of: {spellings}")
            if flag not in flag_to_param:
                raise ValueError(f"Unknown option: {flag!r}")
            param, bool_val = flag_to_param[flag]
            if bool_val is not None:
                raise ValueError(f"Option {flag!r} does not take a value")
            result_name = _param_value_name(param, ambiguous_flags)
            if result_name in result:
                raise ValueError(f"Option '{param_flag(param.name)}' specified more than once")
            result[result_name] = value
            i += 1
            continue

        # Handle ``--name`` form.
        if token in ambiguous_flags:
            spellings = ", ".join(
                param_flag(param.qualified_name) for param in ambiguous_flags[token]
            )
            raise ValueError(f"Option {token!r} is ambiguous; use one of: {spellings}")
        if token not in flag_to_param:
            raise ValueError(f"Unknown option: {token!r}")

        param, bool_val = flag_to_param[token]
        if bool_val is not None:
            # Bool flag: --name → True, --no-name → False.
            result_name = _param_value_name(param, ambiguous_flags)
            if result_name in result:
                raise ValueError(f"Option '{param_flag(param.name)}' specified more than once")
            result[result_name] = bool_val
            i += 1
        else:
            # Value-taking flag: next token is the value.
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                raise ValueError(f"Option {token!r} requires a value")
            result_name = _param_value_name(param, ambiguous_flags)
            if result_name in result:
                raise ValueError(f"Option '{param_flag(param.name)}' specified more than once")
            result[result_name] = tokens[i + 1]
            i += 2

    return result


def render_param_help_section(params: tuple[ParamDeclInfo, ...]) -> str:
    """Render the 'Program parameters:' help section for ``--help`` output.

    Returns a string starting with ``'Program parameters:\\n'`` followed by one
    line per param, or an empty string when there are no params.
    """
    if not params:
        return ""
    ambiguous = _ambiguous_short_flags(params)
    lines: list[str] = ["Program parameters:"]
    for p in params:
        display_name = _param_value_name(p, ambiguous)
        is_bool = isinstance(p.type, BoolType)
        if is_bool:
            flag_str = f"{param_flag(display_name)}/{negative_param_flag(display_name)}"
            type_label = "bool"
        else:
            type_label = p.type.kind.upper()
            flag_str = f"{param_flag(display_name)} {type_label}"
        req_str = "(required)" if not p.has_default else "(optional, has default)"
        lines.append(f"  {flag_str}  {req_str}")
    return "\n".join(lines) + "\n"
