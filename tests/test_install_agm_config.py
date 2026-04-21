"""Tests for the AGM config installer helper script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_install_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "tools" / "install_agm_config.py"
    spec = importlib.util.spec_from_file_location("install_agm_config", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_uses_custom_prefix_when_provided(monkeypatch) -> None:
    module = _load_install_module()
    calls: list[tuple[Path, Path, bool]] = []

    def fake_install_user_config(
        *,
        repo_root: Path,
        install_root: Path,
        force: bool = False,
    ) -> object:
        calls.append((repo_root, install_root, force))
        return module.InstallUserConfigResult(installed=[], skipped=[])

    monkeypatch.setattr(module, "install_user_config", fake_install_user_config)

    exit_code = module.main(["/usr/local", "--force"])

    assert exit_code == 0
    assert calls == [(Path(module.__file__).resolve().parents[1], Path("/usr/local"), True)]


def test_main_defaults_to_home_when_prefix_not_provided(monkeypatch, tmp_path: Path) -> None:
    module = _load_install_module()
    calls: list[tuple[Path, Path, bool]] = []

    def fake_install_user_config(
        *,
        repo_root: Path,
        install_root: Path,
        force: bool = False,
    ) -> object:
        calls.append((repo_root, install_root, force))
        return module.InstallUserConfigResult(installed=[], skipped=[])

    monkeypatch.setattr(module, "install_user_config", fake_install_user_config)
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)

    exit_code = module.main(["--force"])

    assert exit_code == 0
    assert calls == [(Path(module.__file__).resolve().parents[1], tmp_path, True)]


def test_install_user_config_installs_prompts(tmp_path: Path) -> None:
    module = _load_install_module()
    repo_root = tmp_path / "repo"
    install_root = tmp_path / "install"

    (repo_root / "config" / "sandbox").mkdir(parents=True)
    (repo_root / "config" / "sandbox" / "default.json").write_text("{}")
    (repo_root / "config" / "prompts").mkdir(parents=True)
    (repo_root / "config" / "prompts" / "loop.md").write_text("loop prompt\n")
    (repo_root / "config" / "config.toml").write_text("[run]\n")

    result = module.install_user_config(repo_root=repo_root, install_root=install_root)

    prompt_path = install_root / ".agm" / "prompts" / "loop.md"
    assert prompt_path in result.installed
    assert prompt_path.read_text() == "loop prompt\n"
