"""CLI helpers for mapping AgL ``param`` declarations to exec CLI options.

Each ``param`` declaration in a program becomes a ``--<name>`` option on
``agm exec``.  Bool params use ``--name/--no-name`` flag form.  This module
provides pure, unit-testable functions used by both the exec command and the
help/completion machinery.

It also provides ``resolve_param_values`` for merging config-file values with
CLI values (CLI wins) and detecting undeclared config keys.  This helper is
reusable by M6 (REPL config resolution).
"""

from __future__ import annotations

from collections.abc import Mapping

from agm.agl.diagnostics import Diagnostic, format_diagnostic
from agm.agl.runtime.request import AgentResponse
from agm.agl.runtime.runtime import ParamDeclInfo
from agm.agl.semantics.types import BoolType

# Reserved built-in flag strings for the ``agm exec`` command.
RESERVED_FLAGS: frozenset[str] = frozenset(
    {
        "--command",
        "-c",
        "--runner",
        "--log",
        "--log-file",
        "--no-log",
        "--strict-json",
        "--no-strict-json",  # negation of --strict-json; both are real built-ins
        "--max-iters",
        "--module-path",
        "-I",
        "--help",
        "-h",
        "--dry-run",
    }
)


def param_flag(name: str) -> str:
    """Return the CLI flag for a param name (verbatim underscores preserved)."""
    return f"--{name}"


def negative_param_flag(name: str) -> str:
    """Return the negative CLI flag for a bool param (``--no-<name>``)."""
    return f"--no-{name}"


def discover_params_from_source(source: str) -> tuple[ParamDeclInfo, ...]:
    """Discover declared params from AgL *source*, degrading to ``()`` on error.

    Shared by the help and shell-completion paths, which both need only the
    discovered params and must tolerate unreadable/unparsable sources.
    """
    try:
        from agm.agl import WorkflowRuntime

        prepared = WorkflowRuntime.prepare(source)
        discovery = WorkflowRuntime(
            default_agent=lambda request: AgentResponse(content="")
        ).discover_params(prepared)
        return discovery.params
    except (Exception, SystemExit):
        return ()


def _normalize_flag(flag: str) -> str:
    """Normalize a flag by replacing underscores with hyphens for collision detection."""
    return flag.replace("_", "-")


_NORMALIZED_RESERVED: frozenset[str] = frozenset(_normalize_flag(f) for f in RESERVED_FLAGS)


def _format_param_collision(
    param: ParamDeclInfo, flag: str, *, source_name: str | None
) -> str:
    """Return a formatted diagnostic for a param flag collision."""
    return format_diagnostic(
        Diagnostic(
            message=(
                f"param '{param.name}' generates flag '{flag}' which collides "
                "with a built-in exec option; rename the param."
            ),
            line=param.line,
            column=param.col,
        ),
        source_name=source_name,
    )


def check_param_collisions(
    params: tuple[ParamDeclInfo, ...], *, source_name: str | None = "<agl>"
) -> list[str]:
    """Check for collisions between param-generated flags and reserved built-in flags.

    Returns a list of error messages (empty = no collisions).  A collision is
    either an exact match against ``RESERVED_FLAGS`` or a normalized
    (underscore → hyphen) match.  For bool params, ``--no-<name>`` is also checked.
    """
    errors: list[str] = []
    for param in params:
        flag = param_flag(param.name)
        norm_flag = _normalize_flag(flag)
        if flag in RESERVED_FLAGS or norm_flag in _NORMALIZED_RESERVED:
            errors.append(_format_param_collision(param, flag, source_name=source_name))
        if isinstance(param.type, BoolType):
            no_flag = negative_param_flag(param.name)
            norm_no_flag = _normalize_flag(no_flag)
            if no_flag in RESERVED_FLAGS or norm_no_flag in _NORMALIZED_RESERVED:
                errors.append(
                    _format_param_collision(param, no_flag, source_name=source_name)
                )
    return errors


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
    # Build forward-lookup table: flag string → (param, bool_value)
    # bool_value is True/False for bool flags, None for value-taking flags.
    flag_to_param: dict[str, tuple[ParamDeclInfo, bool | None]] = {}
    for p in params:
        if isinstance(p.type, BoolType):
            # Positive bool flag → True; negative → False.
            flag_to_param[param_flag(p.name)] = (p, True)
            flag_to_param[negative_param_flag(p.name)] = (p, False)
        else:
            flag_to_param[param_flag(p.name)] = (p, None)

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
            if flag not in flag_to_param:
                raise ValueError(f"Unknown option: {flag!r}")
            param, bool_val = flag_to_param[flag]
            if bool_val is not None:
                raise ValueError(f"Option {flag!r} does not take a value")
            if param.name in result:
                raise ValueError(
                    f"Option '{param_flag(param.name)}' specified more than once"
                )
            result[param.name] = value
            i += 1
            continue

        # Handle ``--name`` form.
        if token not in flag_to_param:
            raise ValueError(f"Unknown option: {token!r}")

        param, bool_val = flag_to_param[token]
        if bool_val is not None:
            # Bool flag: --name → True, --no-name → False.
            if param.name in result:
                raise ValueError(
                    f"Option '{param_flag(param.name)}' specified more than once"
                )
            result[param.name] = bool_val
            i += 1
        else:
            # Value-taking flag: next token is the value.
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                raise ValueError(f"Option {token!r} requires a value")
            if param.name in result:
                raise ValueError(
                    f"Option '{param_flag(param.name)}' specified more than once"
                )
            result[param.name] = tokens[i + 1]
            i += 2

    return result


def render_param_help_section(params: tuple[ParamDeclInfo, ...]) -> str:
    """Render the 'Program parameters:' help section for ``--help`` output.

    Returns a string starting with ``'Program parameters:\\n'`` followed by one
    line per param, or an empty string when there are no params.
    """
    if not params:
        return ""
    lines: list[str] = ["Program parameters:"]
    for p in params:
        is_bool = isinstance(p.type, BoolType)
        if is_bool:
            flag_str = f"{param_flag(p.name)}/{negative_param_flag(p.name)}"
            type_label = "bool"
        else:
            type_label = p.type.kind.upper()
            flag_str = f"{param_flag(p.name)} {type_label}"
        req_str = "(required)" if not p.has_default else "(optional, has default)"
        lines.append(f"  {flag_str}  {req_str}")
    return "\n".join(lines) + "\n"


def resolve_param_values(
    declared_names: set[str] | frozenset[str],
    config_values: Mapping[str, object],
    cli_values: Mapping[str, object],
    *,
    program_name: str | None = None,
) -> tuple[dict[str, object], list[str]]:
    """Merge config and CLI param values, with CLI taking precedence.

    Precedence: CLI > config.  Config keys that are not declared as params
    produce warning messages (non-fatal, O4) and are excluded from the result.
    CLI values are assumed to be all-declared (unknown CLI options are a hard
    error, handled upstream in M4).

    Args:
        declared_names: The set of param names declared in the program.
        config_values: Raw TOML-native values from ``[params.<program>]`` config.
        cli_values: Values parsed from CLI ``--param`` tokens.
        program_name: The program name used for the config table key, used in
            warning messages.  When ``None``, the table name is omitted.

    Returns:
        A ``(external_dict, warning_messages)`` tuple where:
        - ``external_dict`` maps declared param names to their resolved values
          (config values for declared params, overridden by CLI values).
        - ``warning_messages`` lists human-readable warnings for each config key
          that is not declared as a param.
    """
    warnings: list[str] = []
    # Start from config values that ARE declared, warn on undeclared.
    external: dict[str, object] = {}
    table_ref = f"[params.{program_name}]" if program_name is not None else "[params.*]"
    for key, value in config_values.items():
        if key in declared_names:
            external[key] = value
        else:
            warnings.append(
                f"warning: config key '{key}' in {table_ref} is not a declared param "
                f"and will be ignored"
            )
    # CLI values override config values (CLI wins).
    external.update(cli_values)
    return external, warnings
