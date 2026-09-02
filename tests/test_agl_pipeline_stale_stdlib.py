"""Regression tests for default stdlib version-mismatch diagnostics.

``PipelineDriver.prepare_program``/``prepare_parsed_entry`` are non-raising:
every load, override, or scope failure becomes a ``PreparedProgram``
diagnostic. A mismatched active store ``std`` package must follow that same
contract rather than escaping from default root resolution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agm.config.module_roots as module_roots
from agm.agl.pipeline import PipelineDriver
from agm.config.module_roots import StdlibResolutionError, StdlibVersionMismatchError


def _raise_version_mismatch(*, home: Path, anchor: Path | None = None) -> Path:
    raise StdlibVersionMismatchError("0.0.1", "0.1.0")


class TestPrepareProgramDefaultRootsVersionMismatch:
    def test_version_mismatch_is_captured_as_a_diagnostic_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``prepare_program(source)`` with ``roots=None`` must not raise."""
        monkeypatch.setattr(module_roots, "resolve_stdlib_root", _raise_version_mismatch)

        prepared = PipelineDriver.prepare_program("program def main() -> unit = ()")

        assert prepared.resolved is None
        assert prepared.diagnostics
        assert "0.0.1" in prepared.diagnostics[0].message

    def test_activation_resolution_error_is_captured_as_a_diagnostic_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_resolution_error(*, home: Path, anchor: Path | None = None) -> Path:
            raise StdlibResolutionError("corrupt active std package")

        monkeypatch.setattr(module_roots, "resolve_stdlib_root", raise_resolution_error)

        prepared = PipelineDriver.prepare_program("program def main() -> unit = ()")

        assert prepared.resolved is None
        assert prepared.diagnostics
        assert "corrupt active std package" in prepared.diagnostics[0].message

    def test_version_mismatch_diagnostic_surfaces_from_discover_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``discover_params`` reports why its default stdlib could not load."""
        monkeypatch.setattr(module_roots, "resolve_stdlib_root", _raise_version_mismatch)

        prepared = PipelineDriver.prepare_program("param x: int\nprogram def main() -> unit = ()")
        discovery = PipelineDriver().discover_params(prepared)

        assert discovery.params == ()
        assert discovery.diagnostics
        assert "0.0.1" in discovery.diagnostics[0].message


class TestDiscoverParamsFromSourceVersionMismatch:
    def test_help_param_discovery_degrades_to_empty_without_crashing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ``--help``/completion path degrades gracefully on a mismatch."""
        monkeypatch.setattr(module_roots, "resolve_stdlib_root", _raise_version_mismatch)
        from agm.cli_support.exec_params import discover_params_from_source

        params = discover_params_from_source("param x: int\nprogram def main() -> unit = ()")

        assert params == ()
