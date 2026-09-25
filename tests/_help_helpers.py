"""Shared assertions over rendered CLI help."""

from __future__ import annotations

from agm.cli_support.execution_options import EXECUTION_OPTION_SPECS

#: Every flag spelling of the shared execution options.
EXECUTION_OPTION_FLAGS = frozenset(
    flag
    for spec in EXECUTION_OPTION_SPECS
    for declaration in spec.declarations
    for flag in declaration.split("/")
)


def execution_options_section(text: str) -> str:
    """Return the body of *text*'s ``Execution options`` section."""
    _, heading, rest = text.partition("Execution options:\n")
    assert heading, "no Execution options section"
    section, _, _ = rest.partition("\n\n")
    return section


def execution_options_is_last(text: str) -> bool:
    """Report whether *text*'s ``Execution options`` section ends it."""
    return "\n\n" not in text.partition("Execution options:\n")[2].rstrip("\n")


def assert_lists_execution_options(text: str) -> None:
    """Assert *text*'s ``Execution options`` section lists exactly the shared flags."""
    section = execution_options_section(text)
    listed = {token for token in section.split() if token.startswith("--")}
    assert listed == EXECUTION_OPTION_FLAGS
    assert "--help" not in section
