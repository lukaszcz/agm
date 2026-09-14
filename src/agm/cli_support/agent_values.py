"""Normalize host-facing ``Agent`` values into AgL constructor source."""

from __future__ import annotations

from agm.agent.spec import AGENT_SPECS, AgentClaude, AgentCodex, AgentCommand, AgentSpec
from agm.agent.values import parse_agent_shorthand
from agm.agl.parser import AglSyntaxError, parse_program
from agm.agl.syntax import Call, VarRef
from agm.agl.value_syntax.lexical import quote_text

__all__ = ["normalize_agent_source"]


def _is_agent_constructor_source(source: str) -> bool:
    """Return whether *source* is one direct standard Agent constructor call."""
    try:
        program = parse_program(source)
    except AglSyntaxError:
        return False
    if len(program.body.items) != 1:
        return False
    expression = program.body.items[0]
    return (
        isinstance(expression, Call)
        and isinstance(expression.callee, VarRef)
        and expression.callee.name in AGENT_SPECS
    )


def _render_agent(spec: AgentSpec) -> str:
    """Render one host specification as canonical AgL constructor source."""
    if isinstance(spec, AgentCommand):
        return f"AgentCommand({quote_text(spec.command)})"
    if isinstance(spec, AgentClaude):
        return f"AgentClaude({quote_text(spec.model)}, {quote_text(spec.thinking)})"
    if isinstance(spec, AgentCodex):
        return f"AgentCodex({quote_text(spec.model)}, {quote_text(spec.thinking)})"
    return (
        f"AgentPi({quote_text(spec.provider)}, {quote_text(spec.model)}, "
        f"{quote_text(spec.thinking)})"
    )


def normalize_agent_source(source: str) -> str:
    """Normalize shorthand or command text while preserving canonical AgL syntax."""
    shorthand = parse_agent_shorthand(source)
    if shorthand is not None:
        return _render_agent(shorthand)
    if _is_agent_constructor_source(source):
        return source
    return _render_agent(AgentCommand(source))
