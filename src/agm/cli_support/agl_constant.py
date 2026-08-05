"""Host-side parsing of AgL constant expressions from CLI/config text.

:func:`parse_constant` drives the whole AgL pipeline to parse, typecheck, and
evaluate one constant expression supplied as a plain host string — the form
``agm exec``/``agm repl`` accept for ``--agent`` and the ``[exec]
default-agent`` configuration value (see :mod:`agm.cli_support.engine_seeds`,
its only production caller). It lives here, beside that caller, rather than
in ``agm.agl``, because it drives the whole pipeline: hosting it inside the
language tree put a pipeline-level entry point within reach of the compiler
passes. The AgL imports below are deferred to function scope (or
``TYPE_CHECKING``) so that importing this module costs nothing until a
constant is actually parsed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agm.agl.semantics.types import Type
    from agm.agl.semantics.values import Value

__all__ = ["ConstantExpressionError", "parse_constant"]


class ConstantExpressionError(ValueError):
    """A source string cannot produce a constant value of its expected type."""


def parse_constant(source: str, expected_type: "Type") -> "Value":
    """Parse, typecheck, and evaluate one constant AgL expression.

    The expression is checked through the ordinary program pipeline, which
    applies ``is_constant_expression`` (``agm.agl.syntax.constants``) using
    its resolved constructor metadata. The resulting runtime value is
    therefore exactly the value an AgL program would construct. The expected
    type is spliced into the wrapper program through
    ``agm.agl.semantics.type_syntax.render_type_syntax``, the renderer that
    owns the ``Type`` -> AgL type-syntax mapping.

    Raises:
        ConstantExpressionError: If *source* is not one expression, fails to
            parse, fails to resolve or typecheck, or is not constant. Every
            error includes the supplied source string so config and CLI
            callers can identify the failing input, and names the pass that
            rejected it where the diagnostic identifies one.
    """
    from agm.agl.diagnostics import DiagnosticPhase
    from agm.agl.modules.ids import ENTRY_ID
    from agm.agl.parser import AglSyntaxError, parse_program
    from agm.agl.pipeline import PipelineDriver
    from agm.agl.semantics.type_syntax import render_type_syntax
    from agm.agl.syntax import Expr, LetDecl
    from agm.agl.syntax.constants import is_constant_expression

    try:
        program = parse_program(source)
    except AglSyntaxError as exc:
        message = f"Invalid AgL constant {source!r}: parse error: {exc}"
        raise ConstantExpressionError(message) from exc

    if len(program.body.items) != 1 or not isinstance(program.body.items[0], Expr):
        raise ConstantExpressionError(
            f"Invalid AgL constant {source!r}: expected exactly one expression."
        )

    # The standard library is opened: a constant expression may name any
    # constructor its declared type comes from, including ``std/core`` ones such
    # as ``Option``'s, not only the builtin prelude's.
    driver = PipelineDriver()
    prepared = driver.prepare_program(
        f"let constant_value: {render_type_syntax(expected_type)} = (\n{source}\n)\nconstant_value",
        default_stdlib=True,
    )
    discovery = driver.discover_params(prepared)
    if discovery.diagnostics:
        diagnostic = discovery.diagnostics[0]
        if diagnostic.phase is DiagnosticPhase.SCOPE:
            kind = "scope error"
        elif diagnostic.phase is DiagnosticPhase.TYPECHECK:
            kind = "type error"
        else:
            # Lexing, parsing, module loading, and match compilation leave the
            # phase tag unset; none of those is a type error, so report them
            # under the neutral umbrella rather than mislabelling them.
            kind = "static error"
        raise ConstantExpressionError(
            f"Invalid AgL constant {source!r}: {kind}: {diagnostic.message}"
        )
    checked = discovery.checked
    compiled = discovery.compiled
    assert checked is not None
    assert compiled is not None

    entry_module = checked.modules[ENTRY_ID]
    constant_decl = next(
        item for item in entry_module.resolved.program.body.items if isinstance(item, LetDecl)
    )
    if not is_constant_expression(
        constant_decl.value,
        is_constructor=lambda node_id: entry_module.constructor_ref_for(node_id) is not None,
    ):
        raise ConstantExpressionError(
            f"Invalid AgL constant {source!r}: non-constant expression "
            "(constructors and literals only)."
        )

    result = driver.run_prepared(prepared, compiled=compiled)
    return result.bindings["constant_value"]
