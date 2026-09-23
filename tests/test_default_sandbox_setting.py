"""Behavior tests for the typed ``std/config::default-sandbox`` engine setting.

Modeled on ``tests/test_default_agent_setting.py``, which covers the sibling
``default-agent`` engine key with the same rigor; the shared ``run_program``/
``assert_shape``/``shapes_match``/``write_file_program``/REPL helpers live in
``tests/_agl_helpers.py`` and are imported rather than duplicated here. The
generic restamp-mechanism tests and the reserved-field-default drift guard
live in ``tests/test_engine_setting_restamp.py`` and
``tests/test_agl_reserved_nominals.py`` respectively, not here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner
from typer.main import get_command

import agm.cli as cli
from agm.agl.ir.ids import NominalId
from agm.agl.ir.reserved_nominals import (
    require_reserved_enum_member_id,
    require_reserved_nominal_id,
)
from agm.agl.runtime.engine_config import build_engine_config_seeds
from agm.agl.semantics.values import BoolValue, RecordValue
from agm.cli_support.args import CheckArgs
from agm.commands import check as check_command
from agm.commands import exec as exec_command
from agm.commands import exec_program as exec_engine
from agm.config.context import ConfigContext
from tests._agl_helpers import (
    assert_shape as _assert_shape,
)
from tests._agl_helpers import (
    eval_ok as _ok,
)
from tests._agl_helpers import (
    read_config_result as _read_result,
)
from tests._agl_helpers import (
    record_variant as _variant,
)
from tests._agl_helpers import (
    repl_session as _session,
)
from tests._agl_helpers import (
    run_program as _run,
)
from tests._agl_helpers import (
    unopened_repl_session as _unopened_session,
)
from tests._agl_helpers import (
    write_file_program,
)
from tests.test_exec_command import _exec_args_no_trace

# ---------------------------------------------------------------------------
# Reserved-fallback value builders
# ---------------------------------------------------------------------------


def _sandbox_default_value() -> RecordValue:
    """Build the reserved-fallback ``Sandbox`` value with its four declared defaults."""
    optional_default = RecordValue(
        nominal=NominalId(require_reserved_enum_member_id("Optional", "Default")), fields={}
    )
    option_none = RecordValue(
        nominal=NominalId(require_reserved_enum_member_id("Option", "None")), fields={}
    )
    return RecordValue(
        nominal=NominalId(require_reserved_nominal_id("Sandbox")),
        fields={
            "memory": optional_default,
            "swap": optional_default,
            "settings": option_none,
            "patch": BoolValue(True),
        },
    )


def _agent_sandbox_member_value(member_name: str) -> RecordValue:
    """Build the reserved-fallback ``AgentSandbox`` inline member value for *member_name*."""
    return RecordValue(
        nominal=NominalId(require_reserved_enum_member_id("AgentSandbox", member_name)),
        fields={},
    )


# ---------------------------------------------------------------------------
# Engine-key identity
# ---------------------------------------------------------------------------


def test_engine_key_uses_the_agent_sandbox_nominal_type() -> None:
    from agm.agl.semantics.engine_keys import get_engine_key_type

    assert repr(get_engine_key_type("default-sandbox")) == "AgentSandbox"


# ---------------------------------------------------------------------------
# Initializer, qualified write, and host-seed override
# ---------------------------------------------------------------------------


def test_default_sandbox_initializer_and_qualified_write_are_visible() -> None:
    result = _run(
        "import std/config\n"
        "let initial = std/config::default-sandbox\n"
        "let initial-is-sandbox = initial is Sandbox\n"
        "std/config::default-sandbox := AgentSandbox::Native\n"
        "let updated = std/config::default-sandbox\n"
        "let updated-is-native = updated is AgentSandbox::Native\n"
        "updated\n"
    )

    assert result.ok
    _assert_shape(
        result.bindings["initial"],
        result.bindings["initial-is-sandbox"],
        _sandbox_default_value(),
    )
    _assert_shape(
        result.bindings["updated"],
        result.bindings["updated-is-native"],
        _agent_sandbox_member_value("Native"),
    )


def test_host_seed_overrides_initializer_until_source_write() -> None:
    seed = build_engine_config_seeds({"default-sandbox": "Native"})
    result = _run(
        "import std/config\n"
        "let seeded = std/config::default-sandbox\n"
        "let seeded-is-native = seeded is AgentSandbox::Native\n"
        "std/config::default-sandbox := AgentSandbox::Disabled\n"
        "let written = std/config::default-sandbox\n"
        "let written-is-disabled = written is AgentSandbox::Disabled\n"
        "written\n",
        seed=seed,
    )

    assert result.ok
    _assert_shape(
        result.bindings["seeded"],
        result.bindings["seeded-is-native"],
        _agent_sandbox_member_value("Native"),
    )
    _assert_shape(
        result.bindings["written"],
        result.bindings["written-is-disabled"],
        _agent_sandbox_member_value("Disabled"),
    )


def test_default_sandbox_write_does_not_reconfigure_host_services() -> None:
    """Only ``trace``/``trace-file`` writes reconfigure a live host service."""
    from agm.agl.runtime.host_settings import HostSettingsPolicy

    trace_settings: list[tuple[bool, str | None]] = []

    def resolve_trace_path(enabled: bool, trace_file: str | None) -> None:
        trace_settings.append((enabled, trace_file))
        return None

    result = _run(
        "import std/config\nstd/config::default-sandbox := AgentSandbox::Native\n()\n",
        host_settings_policy=HostSettingsPolicy(resolve_trace_path=resolve_trace_path),
    )

    assert result.ok
    assert trace_settings == [(False, None)]


# ---------------------------------------------------------------------------
# Full precedence chain: source > CLI > program table > @config > [exec] > declared
# ---------------------------------------------------------------------------


class TestDefaultSandboxPrecedenceChain:
    """Each precedence step immediately adjacent to its neighbor, tier by tier."""

    def test_declared_initializer_used_with_no_configuration(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        program = tmp_path / "program.agl"
        write_file_program(
            program, "import std/config\nprint(std/config::default-sandbox is Sandbox)\n"
        )
        assert exec_command.run(_exec_args_no_trace(program)) is None
        assert capsys.readouterr().out == "true\n"

    def test_exec_table_beats_declared_initializer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[exec]\ndefault-sandbox = "Native"\n')
        program = tmp_path / "prog.agl"
        write_file_program(
            program,
            "import std/config\nprint(std/config::default-sandbox is AgentSandbox::Native)\n",
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        assert exec_command.run(_exec_args_no_trace(program)) is None
        assert capsys.readouterr().out == "true\n"

    def test_config_attribute_beats_exec_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[exec]\ndefault-sandbox = "Native"\n')
        program = tmp_path / "prog.agl"
        write_file_program(
            program,
            "import std/config\n\n"
            "@config(config::default-sandbox = AgentSandbox::Disabled)\n"
            "program def main() -> unit = "
            "print(config::default-sandbox is AgentSandbox::Disabled)\n",
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        assert exec_command.run(_exec_args_no_trace(program)) is None
        assert capsys.readouterr().out == "true\n"

    def test_program_table_beats_config_attribute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[prog.main]\ndefault-sandbox = "Native"\n')
        program = tmp_path / "prog.agl"
        write_file_program(
            program,
            "import std/config\n\n"
            "@config(config::default-sandbox = AgentSandbox::Disabled)\n"
            "program def main() -> unit = "
            "print(config::default-sandbox is AgentSandbox::Native)\n",
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        assert exec_command.run(_exec_args_no_trace(program)) is None
        assert capsys.readouterr().out == "true\n"

    def test_cli_beats_program_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[prog.main]\ndefault-sandbox = "Native"\n')
        program = tmp_path / "prog.agl"
        write_file_program(
            program,
            "import std/config\nprint(std/config::default-sandbox is AgentSandbox::Disabled)\n",
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        assert exec_command.run(_exec_args_no_trace(program, default_sandbox="Disabled")) is None
        assert capsys.readouterr().out == "true\n"

    def test_source_write_beats_cli(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        program = tmp_path / "program.agl"
        write_file_program(
            program,
            "import std/config\n"
            "std/config::default-sandbox := AgentSandbox::Disabled\n"
            "print(std/config::default-sandbox is AgentSandbox::Disabled)\n",
        )
        assert exec_command.run(_exec_args_no_trace(program, default_sandbox="Native")) is None
        assert capsys.readouterr().out == "true\n"


# ---------------------------------------------------------------------------
# Every accepted spelling, from the CLI and from config
# ---------------------------------------------------------------------------


class TestDefaultSandboxAcceptedSpellingsFromCli:
    @pytest.mark.parametrize(
        ("cli_literal", "expected_substrings"),
        [
            ("Disabled", ("AgentSandbox::Disabled",)),
            ("Native", ("AgentSandbox::Native",)),
            ("Sandbox(patch = false)", ("patch = false", "Optional::Default")),
            (
                'Sandbox(memory = Some("8G"), patch = false)',
                ("8G", "patch = false"),
            ),
        ],
    )
    def test_exec_accepts_every_default_sandbox_spelling_from_cli(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        cli_literal: str,
        expected_substrings: tuple[str, ...],
    ) -> None:
        program = tmp_path / "program.agl"
        write_file_program(program, "import std/config\nprint std/config::default-sandbox\n")
        assert exec_command.run(_exec_args_no_trace(program, default_sandbox=cli_literal)) is None
        rendered = capsys.readouterr().out
        for substring in expected_substrings:
            assert substring in rendered


class TestDefaultSandboxAcceptedSpellingsFromConfig:
    @pytest.mark.parametrize(
        ("config_body", "expected_substrings"),
        [
            ('[exec]\ndefault-sandbox = "Disabled"\n', ("AgentSandbox::Disabled",)),
            ('[exec]\ndefault-sandbox = "Native"\n', ("AgentSandbox::Native",)),
            (
                '[exec]\ndefault-sandbox = "Sandbox(patch = false)"\n',
                ("patch = false", "Optional::Default"),
            ),
            (
                "[exec]\ndefault-sandbox = 'Sandbox(memory = Some(\"8G\"), patch = false)'\n",
                ("8G", "patch = false"),
            ),
            (
                "[exec]\n"
                'default-sandbox = \'{"$case": "Sandbox", '
                '"memory": {"$case": "Some", "value": "8G"}, "patch": false}\'\n',
                ("8G", "patch = false"),
            ),
            (
                '[exec.default-sandbox]\n"$case" = "Sandbox"\npatch = false\n\n'
                '[exec.default-sandbox.memory]\n"$case" = "Some"\nvalue = "8G"\n',
                ("8G", "patch = false"),
            ),
        ],
        ids=[
            "disabled",
            "native",
            "sandbox-bare",
            "sandbox-fields-text",
            "json-tagged-object",
            "toml-native-table",
        ],
    )
    def test_exec_accepts_every_default_sandbox_spelling_from_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        config_body: str,
        expected_substrings: tuple[str, ...],
    ) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(config_body)
        program = tmp_path / "program.agl"
        write_file_program(program, "import std/config\nprint std/config::default-sandbox\n")
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        assert exec_command.run(_exec_args_no_trace(program)) is None
        rendered = capsys.readouterr().out
        for substring in expected_substrings:
            assert substring in rendered


# ---------------------------------------------------------------------------
# Rejection: malformed text and wrong-shape config values
# ---------------------------------------------------------------------------


class TestDefaultSandboxRejection:
    def test_exec_rejects_malformed_default_sandbox_literal_from_cli(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A syntactically broken CLI literal exits 1, before the module graph loads.

        The program carries an unresolvable import: if seed decoding ran
        after module loading, the module-not-found diagnostic would surface
        instead. It never does, pinning the decode-before-load ordering.
        """
        program = tmp_path / "program.agl"
        write_file_program(program, 'import nonexistent/mod\nprint "not-run"\n')

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_trace(program, default_sandbox="Sandbox("))

        assert exc_info.value.code == 1
        out, err = capsys.readouterr()
        assert out == ""
        assert "default-sandbox" in err
        assert "--default-sandbox" in err
        assert "nonexistent" not in err

    def test_exec_rejects_malformed_default_sandbox_literal_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A syntactically broken config-table literal exits 1, before the module graph loads.

        See :meth:`test_exec_rejects_malformed_default_sandbox_literal_from_cli`
        for why the program carries an unresolvable import.
        """
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[exec]\ndefault-sandbox = "Sandbox("\n')
        program = tmp_path / "program.agl"
        write_file_program(program, 'import nonexistent/mod\nprint "not-run"\n')
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_trace(program))

        assert exc_info.value.code == 1
        out, err = capsys.readouterr()
        assert out == ""
        assert "default-sandbox" in err
        assert "configuration key" in err
        assert "nonexistent" not in err

    def test_exec_rejects_wrong_shape_default_sandbox_value_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A well-formed but non-``AgentSandbox`` value (a bare integer) is rejected."""
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text("[exec]\ndefault-sandbox = 7\n")
        program = tmp_path / "program.agl"
        write_file_program(program, 'print "not-run"\n')
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_trace(program))

        assert exc_info.value.code == 1
        out, err = capsys.readouterr()
        assert out == ""
        assert "default-sandbox" in err


# ---------------------------------------------------------------------------
# REPL: ``--default-sandbox`` seed threading, cross-entry persistence, and
# ``restamp_engine_setting`` in both directions
# ---------------------------------------------------------------------------


class TestReplDefaultSandboxSeedThreading:
    """Mirrors ``test_agl_repl_builtin_settings.py``'s ``TestDefaultAgentSeedThreading``."""

    def test_seed_is_observed_and_persists_across_entries(self) -> None:
        s = _unopened_session(engine_base=build_engine_config_seeds({"default-sandbox": "Native"}))
        _ok(s, "import std/config")
        result = _read_result(s, "default-sandbox")
        assert isinstance(result.value, RecordValue)
        assert result.descriptors is not None
        # ``result.value``'s nominal is restamped onto the running session's own
        # program identity when it is read back (host seed -> program direction
        # of ``restamp_engine_setting``).
        assert _variant(result.value, result.descriptors) == "Native"

        _ok(s, "let unrelated = 1")
        later = _read_result(s, "default-sandbox")
        assert isinstance(later.value, RecordValue)
        assert later.descriptors is not None
        assert _variant(later.value, later.descriptors) == "Native"

    def test_source_write_still_overrides_it_afterward(self) -> None:
        s = _unopened_session(engine_base=build_engine_config_seeds({"default-sandbox": "Native"}))
        _ok(s, "import std/config")
        initial = _read_result(s, "default-sandbox")
        assert isinstance(initial.value, RecordValue)
        assert initial.descriptors is not None
        assert _variant(initial.value, initial.descriptors) == "Native"

        _ok(s, "std/config::default-sandbox := AgentSandbox::Disabled")
        result = _read_result(s, "default-sandbox")
        assert isinstance(result.value, RecordValue)
        assert result.descriptors is not None
        assert _variant(result.value, result.descriptors) == "Disabled"


class TestCrossEntryPersistenceForDefaultSandbox:
    """A write in entry N is visible to a read two entries later (N+2).

    Together with :class:`TestResetHostSeedPrecedenceForDefaultSandbox`, this
    exercises both directions of
    :func:`~agm.agl.runtime.engine_config.restamp_engine_setting`: a value
    read back here carries the *host seed -> program* restamp (the seed's
    reserved-fallback identity onto this session's own running program), and
    the write persisted for the later read carries the *program ->
    reserved-fallback* direction (the post-run register, restamped back onto
    the reserved-fallback table so it survives past this program's own
    lifetime -- see ``IrInterpreter.builtin_host_settings``).
    """

    def test_write_persists_two_entries_later(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        _ok(s, "std/config::default-sandbox := AgentSandbox::Disabled")
        _ok(s, "let unrelated = 1")
        result = _read_result(s, "default-sandbox")
        assert isinstance(result.value, RecordValue)
        assert result.descriptors is not None
        assert _variant(result.value, result.descriptors) == "Disabled"


class TestResetHostSeedPrecedenceForDefaultSandbox:
    """``:reset`` discards a source write and restores the host-seeded value.

    ``ReplSession.reset`` restores ``_engine_seed``, which is already held in
    reserved-fallback identity, so the source write made before the reset
    never reaches it -- the read afterward sees the original host seed, not
    the write.
    """

    def test_host_seed_survives_a_source_write(self) -> None:
        s = _session(engine_base=build_engine_config_seeds({"default-sandbox": "Native"}))
        _ok(s, "import std/config")
        _ok(s, "std/config::default-sandbox := AgentSandbox::Disabled")
        s.reset()
        _ok(s, "import std/config")
        result = _read_result(s, "default-sandbox")
        assert isinstance(result.value, RecordValue)
        assert result.descriptors is not None
        assert _variant(result.value, result.descriptors) == "Native"


# ---------------------------------------------------------------------------
# ``agm check`` never seeds engine settings
# ---------------------------------------------------------------------------


class TestCheckNeverSeedsEngineSettings:
    def test_default_sandbox_flag_is_rejected_on_check(self, tmp_path: Path) -> None:
        """``agm check`` has no ``--default-sandbox`` flag: passing one is a usage error.

        A user-visible CLI-surface check that ``check`` can never build a
        seed for an engine key, rather than an assertion on ``CheckArgs``'s
        internal field set.
        """
        agl_file = tmp_path / "program.agl"
        agl_file.write_text('program def main() -> unit =\n  print "hi"\n')

        result = CliRunner().invoke(
            get_command(cli.app),
            ["check", "--default-sandbox", "Native", str(agl_file)],
            prog_name="agm",
        )

        assert result.exit_code != 0

    def test_default_sandbox_read_type_checks_without_being_seeded(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A program reading ``default-sandbox`` checks cleanly with no output.

        ``check`` never reaches evaluation, so a ``default-sandbox`` read
        type-checks without any seed ever being built or consumed, and
        without the read ever actually running.
        """
        agl_file = tmp_path / "program.agl"
        agl_file.write_text(
            "import std/config\nprogram def main() -> unit =\n  print std/config::default-sandbox\n"
        )

        check_command.run(CheckArgs(files=[str(agl_file)]))

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
