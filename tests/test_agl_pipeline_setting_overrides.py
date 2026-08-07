"""Tests for host-supplied engine-setting overrides on ``PipelineDriver``.

An engine setting's value can be supplied as AgL source text from the host
(``PipelineDriver.prepare_program``'s ``setting_overrides``) and spliced in as
its ``std/config`` ``builtin var`` declaration's default before scope
resolution, so the override is resolved, type-checked, and constant-checked by
the program's own single compilation pass rather than a separate throwaway
one. This module also covers the ``parse_entry`` / ``prepare_parsed_entry``
split that makes the splice point reachable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import format_diagnostic
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import ParsedEntry, PipelineDriver, PreparedProgram
from agm.agl.semantics.values import EnumValue, TextValue
from agm.agl.setting_overrides import SettingOverride

_STDLIB = Path(__file__).resolve().parent.parent / "stdlib"


def _roots() -> RootSet:
    return RootSet(roots=frozenset({_STDLIB}))


def _prepare(
    source: str,
    *,
    overrides: dict[str, SettingOverride] | None = None,
    default_stdlib: bool = True,
) -> tuple[PipelineDriver, PreparedProgram]:
    driver = PipelineDriver()
    prepared = driver.prepare_program(
        source,
        entry_path=None,
        roots=_roots(),
        default_stdlib=default_stdlib,
        setting_overrides=overrides,
    )
    return driver, prepared


def _diagnostic_text(prepared: PreparedProgram) -> str:
    assert prepared.diagnostics
    return format_diagnostic(prepared.diagnostics[0])


# ---------------------------------------------------------------------------
# parse_entry: the pipeline-level split
# ---------------------------------------------------------------------------


class TestParseEntry:
    def test_exposes_parsed_program_and_next_id(self) -> None:
        parsed = PipelineDriver.parse_entry("let x = 1\nx")
        assert isinstance(parsed, ParsedEntry)
        assert parsed.diagnostics == ()
        assert parsed.program is not None
        assert parsed.next_id > 0

    def test_exposes_program_name_without_scope_resolution(self) -> None:
        parsed = PipelineDriver.parse_entry("program my-tool\nlet x = 1\nx")
        assert parsed.program_name == "my-tool"

    def test_no_program_declaration_is_none(self) -> None:
        parsed = PipelineDriver.parse_entry("let x = 1\nx")
        assert parsed.program_name is None

    def test_syntax_error_is_captured_as_a_diagnostic(self) -> None:
        parsed = PipelineDriver.parse_entry("let x = (")
        assert parsed.program is None
        assert parsed.diagnostics

    def test_prepare_parsed_entry_matches_prepare_program(self) -> None:
        """``prepare_program`` is a thin wrapper: same result either way."""
        source = "open import std/config\nlet value = std/config::default-agent\nvalue"
        driver = PipelineDriver()

        parsed = PipelineDriver.parse_entry(source, entry_path=None)
        via_split = PipelineDriver.prepare_parsed_entry(parsed, roots=_roots(), default_stdlib=True)
        via_wrapper = driver.prepare_program(source, entry_path=None, roots=_roots())

        assert via_split.diagnostics == via_wrapper.diagnostics == ()
        assert via_split.resolved is not None
        assert via_wrapper.resolved is not None


# ---------------------------------------------------------------------------
# Regression: no overrides behaves exactly as before
# ---------------------------------------------------------------------------


class TestNoOverridesRegression:
    def test_prepare_without_overrides_reads_the_declared_default(self) -> None:
        driver, prepared = _prepare(
            "open import std/config\nlet value = std/config::default-agent\nvalue"
        )
        assert prepared.diagnostics == ()
        result = driver.run_prepared(prepared)
        assert result.ok, f"expected success but got: {result.diagnostics or result.error!r}"
        value = result.bindings["value"]
        assert isinstance(value, EnumValue)
        assert value.variant == "AgentClaude"


# ---------------------------------------------------------------------------
# A successful override, observed by the program, compiled exactly once
# ---------------------------------------------------------------------------


class TestOverrideApplied:
    def test_default_agent_override_observed_by_the_program(self) -> None:
        driver, prepared = _prepare(
            "open import std/config\nlet value = std/config::default-agent\nvalue",
            overrides={
                "default-agent": SettingOverride(
                    source='AgentCommand("overridden")', origin="--agent"
                )
            },
        )
        assert prepared.diagnostics == ()
        result = driver.run_prepared(prepared)
        assert result.ok, f"expected success but got: {result.diagnostics or result.error!r}"
        value = result.bindings["value"]
        assert isinstance(value, EnumValue)
        assert value.variant == "AgentCommand"
        assert value.fields["command"] == TextValue("overridden")

    def test_program_module_graph_is_built_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The override is spliced into the one graph the program itself loads.

        No second throwaway compilation (as a separate pipeline driving a
        synthetic wrapper program would need) is ever triggered.
        """
        import agm.agl.modules.loader as loader_mod

        calls: list[int] = []
        original = loader_mod.build_repl_graph

        def spy(*args: object, **kwargs: object) -> object:
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(loader_mod, "build_repl_graph", spy)

        _driver, prepared = _prepare(
            "open import std/config\nlet value = std/config::default-agent\nvalue",
            overrides={
                "default-agent": SettingOverride(
                    source='AgentCommand("overridden")', origin="--agent"
                )
            },
        )
        assert prepared.diagnostics == ()
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Rejections that go through ordinary program checking
# ---------------------------------------------------------------------------


class TestOverrideCheckedByTheProgram:
    def test_type_mismatched_override_is_a_type_error_naming_the_origin(self) -> None:
        origin = "--agent"
        driver, prepared = _prepare(
            "()",
            overrides={"default-agent": SettingOverride(source='"not-an-agent"', origin=origin)},
        )
        assert prepared.diagnostics == ()
        assert prepared.resolved is not None
        result = driver.run_prepared(prepared)
        assert not result.ok
        assert result.diagnostics
        assert origin in format_diagnostic(result.diagnostics[0])

    def test_non_constant_override_is_rejected(self) -> None:
        origin = "--agent"
        driver, prepared = _prepare(
            "()",
            overrides={
                "default-agent": SettingOverride(
                    source='AgentCommand("not " + "constant")', origin=origin
                )
            },
        )
        assert prepared.diagnostics == ()
        assert prepared.resolved is not None
        result = driver.run_prepared(prepared)
        assert not result.ok
        assert result.diagnostics


# ---------------------------------------------------------------------------
# Rejections diagnosed while applying the override itself
# ---------------------------------------------------------------------------


class TestOverrideApplicationRejections:
    def test_unparseable_override_reports_diagnostic_naming_origin(self) -> None:
        origin = "[exec] default-agent"
        _driver, prepared = _prepare(
            "()",
            overrides={"default-agent": SettingOverride(source="(", origin=origin)},
        )
        assert prepared.resolved is None
        assert origin in _diagnostic_text(prepared)

    def test_multi_expression_override_reports_diagnostic_naming_origin(self) -> None:
        origin = "--agent"
        _driver, prepared = _prepare(
            "()",
            overrides={
                "default-agent": SettingOverride(
                    source='AgentClaude("sonnet", "medium")\nAgentClaude("sonnet", "medium")',
                    origin=origin,
                )
            },
        )
        assert prepared.resolved is None
        assert origin in _diagnostic_text(prepared)

    def test_non_expression_override_reports_diagnostic_naming_origin(self) -> None:
        origin = "--agent"
        _driver, prepared = _prepare(
            "()",
            overrides={"default-agent": SettingOverride(source="let x = 1", origin=origin)},
        )
        assert prepared.resolved is None
        assert origin in _diagnostic_text(prepared)

    def test_unknown_engine_key_reports_diagnostic_naming_origin(self) -> None:
        origin = "[exec] bogus-key"
        _driver, prepared = _prepare(
            "()",
            overrides={"bogus-key": SettingOverride(source="1", origin=origin)},
        )
        assert prepared.resolved is None
        assert origin in _diagnostic_text(prepared)

    def test_missing_stdlib_reports_diagnostic_naming_origin(self) -> None:
        origin = "--agent"
        _driver, prepared = _prepare(
            "()",
            overrides={
                "default-agent": SettingOverride(
                    source='AgentClaude("sonnet", "medium")', origin=origin
                )
            },
            default_stdlib=False,
        )
        assert prepared.resolved is None
        assert origin in _diagnostic_text(prepared)

    def test_no_exception_escapes_a_rejected_override(self) -> None:
        """Every rejection path is a diagnostic, never a raised exception."""
        for source, origin in (
            ("(", "a"),
            ("1\n2", "b"),
            ("let x = 1", "c"),
        ):
            _driver, prepared = _prepare(
                "()", overrides={"default-agent": SettingOverride(source=source, origin=origin)}
            )
            assert prepared.resolved is None
            assert prepared.diagnostics


# ---------------------------------------------------------------------------
# A well-typed override whose command text does not shell-split: caught only
# when ``run_prepared`` constructs the interpreter and materializes the
# winning ``default-agent`` value, not by ``prepare_program``'s static passes.
# ---------------------------------------------------------------------------


class TestMalformedAgentCommandAtConstruction:
    """``AgentCommand("...")`` is a valid constant ``Agent`` expression even when
    its command text does not shell-split, so ``prepare_program`` accepts it
    cleanly; the malformed text is only caught eagerly when the interpreter
    is constructed, before the program's first statement runs.
    """

    def test_unparseable_command_text_is_a_pre_execution_diagnostic(self) -> None:
        origin = "--agent"
        driver, prepared = _prepare(
            "()",
            overrides={
                "default-agent": SettingOverride(
                    source='AgentCommand("nonexistent-bin -p \'oops")', origin=origin
                )
            },
        )
        assert prepared.diagnostics == ()
        assert prepared.resolved is not None
        result = driver.run_prepared(prepared)
        assert not result.ok
        assert result.diagnostics
        assert result.error is None

    def test_well_formed_command_text_runs_normally(self) -> None:
        driver, prepared = _prepare(
            "open import std/config\nlet value = std/config::default-agent\nvalue",
            overrides={
                "default-agent": SettingOverride(source='AgentCommand("echo hi")', origin="--agent")
            },
        )
        result = driver.run_prepared(prepared)
        assert result.ok, f"expected success but got: {result.diagnostics or result.error!r}"
        value = result.bindings["value"]
        assert isinstance(value, EnumValue)
        assert value.variant == "AgentCommand"
        assert value.fields["command"] == TextValue("echo hi")

    def test_seeded_malformed_command_text_is_also_a_pre_execution_diagnostic(self) -> None:
        """A directly host-seeded value (e.g. ``[exec] runner``'s decode target)
        is validated the same way as a declared/spliced default: whichever
        value wins at construction is checked, regardless of which channel
        supplied it.
        """
        from tests._agl_helpers import agent_value

        driver, prepared = _prepare("()")
        result = driver.run_prepared(
            prepared,
            builtin_host_settings={
                "default-agent": agent_value("AgentCommand", command="nonexistent-bin -p 'oops")
            },
        )
        assert not result.ok
        assert result.diagnostics
        assert result.error is None
