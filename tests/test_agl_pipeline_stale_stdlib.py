"""Regression tests for the default (``roots=None``) stdlib-resolution path.

``PipelineDriver.prepare_program``/``prepare_parsed_entry`` are documented as
non-raising: every load, override, or scope failure must be captured into
``PreparedProgram.diagnostics`` rather than raised.  When no explicit
``roots`` is supplied, the pipeline resolves default module roots itself via
``resolve_stdlib_root``, which can raise ``StaleStdlibError`` for a stdlib
tree whose ``STDLIB_CONTRACT`` marker is missing or mismatched.  That raise
happened outside every ``try`` block, breaking the non-raising contract for
this one failure mode; these tests guard against a regression.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agm.config.module_roots as module_roots
from agm.agl.pipeline import PipelineDriver
from agm.config.module_roots import StaleStdlibError


def _raise_stale(*, home: Path) -> Path:
    raise StaleStdlibError(home / ".agm" / "stdlib")


class TestPrepareProgramDefaultRootsStaleStdlib:
    def test_stale_stdlib_is_captured_as_a_diagnostic_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``prepare_program(source)`` with ``roots=None`` must not raise."""
        monkeypatch.setattr(module_roots, "resolve_stdlib_root", _raise_stale)

        prepared = PipelineDriver.prepare_program("1")

        assert prepared.resolved is None
        assert prepared.diagnostics
        assert "out of date" in prepared.diagnostics[0].message

    def test_stale_stdlib_diagnostic_surfaces_from_discover_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``discover_params`` on a stale-stdlib ``PreparedProgram`` reports why.

        Params genuinely cannot be discovered without a usable stdlib (type
        checking depends on it), so the empty result must carry the
        diagnostic explaining the failure rather than looking identical to a
        clean, param-less program.
        """
        monkeypatch.setattr(module_roots, "resolve_stdlib_root", _raise_stale)

        prepared = PipelineDriver.prepare_program("param x: int\nx")
        discovery = PipelineDriver().discover_params(prepared)

        assert discovery.params == ()
        assert discovery.diagnostics
        assert "out of date" in discovery.diagnostics[0].message


class TestDiscoverParamsFromSourceStaleStdlib:
    def test_help_param_discovery_degrades_to_empty_without_crashing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ``--help``/completion path still degrades gracefully, not by luck.

        Before the fix, ``discover_params_from_source``'s broad
        ``except (Exception, SystemExit)`` happened to swallow the raised
        ``StaleStdlibError`` too; now the underlying pipeline call already
        returns cleanly, so this path no longer depends on catching a raise
        that was never supposed to happen.
        """
        monkeypatch.setattr(module_roots, "resolve_stdlib_root", _raise_stale)
        from agm.cli_support.exec_params import discover_params_from_source

        params = discover_params_from_source("param x: int\nx")

        assert params == ()
