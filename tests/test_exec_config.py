"""Tests for exec configuration loading."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agm.cli_support.args import ExecArgs
from agm.commands import exec_program as exec_engine
from agm.config.context import ConfigContext
from agm.config.general import (
    ExecConfig,
    exec_config_from_merged,
    load_merged_config,
)
from tests._package_helpers import write_installed_package


class TestExecConfig:
    def test_explicit_values(self) -> None:
        cfg = ExecConfig(
            strict_json=False,
            timeout=None,
            trace=False,
            trace_file=None,
        )
        assert cfg.strict_json is False
        assert cfg.timeout is None
        assert cfg.trace is False
        assert cfg.trace_file is None

    def test_frozen(self) -> None:
        cfg = ExecConfig(
            strict_json=False,
            timeout=None,
            trace=False,
            trace_file=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            cfg.strict_json = True


class TestExecConfigFromConfigFiles:
    """The ``[exec]`` section as every exec host reads it: merge, then resolve."""

    def _config(self, home: Path, cwd: Path, proj_dir: Path | None = None) -> ExecConfig:
        return exec_config_from_merged(load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd))

    def test_load_defaults_when_no_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        cfg = self._config(home, tmp_path)
        assert cfg.strict_json is False
        assert cfg.timeout is None
        assert cfg.trace is False
        assert cfg.trace_file is None

    def test_exec_settings_load_from_toml(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        config = home / ".agm" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text('[exec]\nstrict-json = true\ntimeout = "30m"\n')
        cfg = self._config(home, tmp_path)
        assert cfg.strict_json is True
        assert cfg.timeout == pytest.approx(1800.0)

    def test_project_config_overrides_home(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text("[exec]\ntimeout = 3\n")
        proj_dir = tmp_path / "proj"
        (proj_dir / "config").mkdir(parents=True)
        (proj_dir / "config" / "config.toml").write_text("[exec]\ntimeout = 7\n")
        cfg = self._config(home, tmp_path, proj_dir)
        assert cfg.timeout == pytest.approx(7.0)

    def test_a_nested_exec_sub_table_supplies_nothing(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        config = home / ".agm" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text("[exec]\ntimeout = 3\n\n[exec.myflow]\ntimeout = 7\n")
        cfg = self._config(home, tmp_path)
        assert cfg.timeout == pytest.approx(3.0)

    def test_numeric_timeout(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        config = home / ".agm" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text("[exec]\ntimeout = 60\n")
        cfg = self._config(home, tmp_path)
        assert cfg.timeout == pytest.approx(60.0)

    def test_trace_settings_loaded_from_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        config = home / ".agm" / "config.toml"
        config.parent.mkdir(parents=True)
        trace_path = tmp_path / "trace.jsonl"
        config.write_text(f"[exec]\ntrace = true\ntrace-file = {str(trace_path)!r}\n")
        cfg = self._config(home, tmp_path)
        assert cfg.trace is True
        assert cfg.trace_file == str(trace_path)

    def test_escaped_trace_file_interpolation_is_not_reapplied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NAME", "expanded")
        home = tmp_path / "home"
        config_dir = home / ".agm"
        literal_dir = config_dir / "%{NAME}"
        literal_dir.mkdir(parents=True)
        (literal_dir / "base.jsonl").touch()
        (config_dir / "config.toml").write_text('[exec]\ntrace-file = "\\\\%{NAME}/base.jsonl"\n')

        cfg = self._config(home, tmp_path)

        assert cfg.trace_file == str(literal_dir / "base.jsonl")


class TestExecConfigProgramTableOverride:
    def test_program_table_overrides_exec_engine_keys(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        config = home / ".agm" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text("[exec]\ntimeout = 5\n")
        merged = load_merged_config(home=home, proj_dir=None, cwd=tmp_path)
        cfg = exec_config_from_merged(merged, program_table={"timeout": 10})
        assert cfg.timeout == pytest.approx(10.0)

    def test_program_table_partial_override(self, tmp_path: Path) -> None:
        merged = {"exec": {"timeout": 7, "strict-json": False}}
        cfg = exec_config_from_merged(merged, program_table={"strict-json": True})
        assert cfg.timeout == pytest.approx(7.0)
        assert cfg.strict_json is True

    @pytest.mark.parametrize(("value", "expected"), [(7, 7), (0, None), (True, None)])
    def test_max_call_depth_accepts_only_a_positive_integer(
        self, value: object, expected: int | None
    ) -> None:
        cfg = exec_config_from_merged({"exec": {"max-call-depth": value}})
        assert cfg.max_call_depth == expected


class TestModuleParameterConfigRoutes:
    """Module parameter config routes feed the ordinary exec host."""

    def _program(self, tmp_path: Path) -> tuple[Path, Path]:
        modules = tmp_path / "modules"
        module = modules / "A" / "logging.agl"
        module.parent.mkdir(parents=True)
        module.write_text(
            "@param let verbose: bool = false\n"
            "scope debug\n"
            "  @param let trace: bool = false\n"
            "end debug\n",
            encoding="utf-8",
        )
        source = tmp_path / "main.agl"
        source.write_text(
            "import A/logging\n"
            "program def main() -> unit =\n"
            "  print A/logging::verbose\n"
            "  print A/logging::debug::trace\n",
            encoding="utf-8",
        )
        return source, modules

    def _configure_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, project: bool = False
    ) -> tuple[Path, Path | None]:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        proj_dir = tmp_path / "project" if project else None
        if proj_dir is not None:
            (proj_dir / "config").mkdir(parents=True)
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=proj_dir, cwd=tmp_path),
        )
        return home, proj_dir

    def _run(self, source: Path, modules: Path) -> None:
        exec_engine.run(
            replace(
                ExecArgs(file=str(source), strict_json=None, no_trace=True, trace_file=None),
                module_paths=[str(modules)],
            )
        )

    @pytest.mark.parametrize(
        "table",
        ["[A.logging.debug]", '["A/logging".debug]'],
    )
    def test_scope_region_config_routes_accept_dotted_and_anchor_spellings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        table: str,
    ) -> None:
        source, modules = self._program(tmp_path)
        home, _ = self._configure_context(tmp_path, monkeypatch)
        (home / ".agm" / "config.toml").write_text(f"{table}\ntrace = true\n")

        self._run(source, modules)

        assert capsys.readouterr().out == "false\ntrue\n"

    def test_program_table_overrides_module_route_across_config_layers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        source, modules = self._program(tmp_path)
        home, project = self._configure_context(tmp_path, monkeypatch, project=True)
        assert project is not None
        (home / ".agm" / "config.toml").write_text("[A.logging]\nverbose = false\n")
        (project / "config" / "config.toml").write_text("[main.main]\nverbose = true\n")

        self._run(source, modules)

        assert capsys.readouterr().out == "true\nfalse\n"

    def test_program_table_accepts_a_qualified_module_parameter_leaf(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        source, modules = self._program(tmp_path)
        home, _ = self._configure_context(tmp_path, monkeypatch)
        (home / ".agm" / "config.toml").write_text(
            '[main.main]\n"A.logging.verbose" = true\n"A.logging.debug.trace" = true\n'
        )

        self._run(source, modules)

        assert capsys.readouterr().out == "true\ntrue\n"

    def test_undeclared_keys_are_reported_for_module_and_program_routes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        source, modules = self._program(tmp_path)
        home, _ = self._configure_context(tmp_path, monkeypatch)
        (home / ".agm" / "config.toml").write_text(
            "[A.logging]\nunknown-module = true\n\n"
            "[main.main]\nunknown-program = true\nmax-iters = 1\n"
        )

        self._run(source, modules)

        err = capsys.readouterr().err
        assert "unknown-module" in err
        assert "unknown-program" in err
        assert "max-iters" in err


class TestPackageEntryConfigRoute:
    """Every spelling of one package program reads the same config table.

    A program that belongs to a package is addressed by its package-qualified
    module route, so its configuration lives under that route whether it was
    reached by file path, by installed reference, or as a registered command.
    """

    _SOURCE = 'program def main(level: text = "unset") -> unit = print level\n'

    def _install(self, tmp_path: Path) -> tuple[Path, Path]:
        """Install a one-module ``tools`` package and configure its program argument."""
        home = tmp_path / "home"
        module = write_installed_package(home, "tools", source=self._SOURCE)
        (home / ".agm" / "config.toml").write_text('[tools.main.main]\nlevel = "prod"\n')
        return home, module

    def _use_home(self, monkeypatch: pytest.MonkeyPatch, home: Path, cwd: Path) -> None:
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=cwd),
        )

    def test_store_package_file_path_reads_the_qualified_table(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        home, module = self._install(tmp_path)
        self._use_home(monkeypatch, home, tmp_path)

        exec_engine.run(
            ExecArgs(file=str(module), strict_json=None, no_trace=False, trace_file=None)
        )

        assert capsys.readouterr().out == "prod\n"

    def test_installed_reference_reads_the_qualified_table(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        home, _module = self._install(tmp_path)
        self._use_home(monkeypatch, home, tmp_path)

        exec_engine.run_registered("tools/main::main", [])

        assert capsys.readouterr().out == "prod\n"

    def _install_registered(self, tmp_path: Path, table: str) -> tuple[Path, Path]:
        """Install a ``tools`` package registering ``agm dev review``, with *table* config."""
        home = tmp_path / "home"
        module = write_installed_package(
            home, "tools", source=self._SOURCE, commands={"dev review": "tools/main::main"}
        )
        (home / ".agm" / "config.toml").write_text(table)
        return home, module

    def test_registered_command_path_addresses_the_program(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``agm dev review`` reads the ``[dev.review]`` table its registration names."""
        home, _module = self._install_registered(tmp_path, '[dev.review]\nlevel = "cmd"\n')
        self._use_home(monkeypatch, home, tmp_path)

        exec_engine.run_registered(
            "tools/main::main", [], package="tools", command_path="dev review"
        )

        assert capsys.readouterr().out == "cmd\n"

    def test_command_path_table_addresses_the_program_by_any_spelling(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The command table addresses the program, not one way of reaching it."""
        home, module = self._install_registered(tmp_path, '[dev.review]\nlevel = "cmd"\n')
        self._use_home(monkeypatch, home, tmp_path)

        exec_engine.run(
            ExecArgs(file=str(module), strict_json=None, no_trace=False, trace_file=None)
        )
        exec_engine.run_registered("tools/main::main", [])

        assert capsys.readouterr().out == "cmd\ncmd\n"

    def test_rejects_a_command_table_conflicting_with_the_module_route(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home, module = self._install_registered(
            tmp_path, '[dev.review]\nlevel = "cmd"\n\n[tools.main.main]\nlevel = "route"\n'
        )
        self._use_home(monkeypatch, home, tmp_path)

        with pytest.raises(SystemExit):
            exec_engine.run(
                ExecArgs(file=str(module), strict_json=None, no_trace=False, trace_file=None)
            )

    def test_command_table_key_is_not_reported_as_undeclared(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        home, module = self._install_registered(tmp_path, '[dev.review]\nlevel = "cmd"\n')
        self._use_home(monkeypatch, home, tmp_path)

        exec_engine.run(
            ExecArgs(file=str(module), strict_json=None, no_trace=False, trace_file=None)
        )

        assert capsys.readouterr().err == ""

    def test_undeclared_command_table_key_is_reported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        home, module = self._install_registered(tmp_path, "[dev.review]\nbogus = 1\n")
        self._use_home(monkeypatch, home, tmp_path)

        exec_engine.run(
            ExecArgs(file=str(module), strict_json=None, no_trace=False, trace_file=None)
        )

        assert "bogus" in capsys.readouterr().err

    def test_loose_file_outside_a_package_keeps_its_stem_route(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A file that no selected package owns is still addressed by its stem."""
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[loose.main]\nlevel = "stem"\n')
        loose = tmp_path / "loose.agl"
        loose.write_text(self._SOURCE, encoding="utf-8")
        self._use_home(monkeypatch, home, tmp_path)

        exec_engine.run(
            ExecArgs(file=str(loose), strict_json=None, no_trace=False, trace_file=None)
        )

        assert capsys.readouterr().out == "stem\n"

    def test_package_entry_program_parameter_keeps_its_bare_cli_spelling(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A package-owned entry's parameters are still spelled as the program's own.

        The entry module carries its package identity internally; that must not
        leak into the selected program's CLI surface.
        """
        home, module = self._install(tmp_path)
        self._use_home(monkeypatch, home, tmp_path)

        exec_engine.run(
            ExecArgs(
                file=str(module),
                strict_json=None,
                no_trace=False,
                trace_file=None,
                argument_tokens=["--level", "bare"],
            )
        )
        assert capsys.readouterr().out == "bare\n"
