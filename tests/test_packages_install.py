"""Tests for package installation, activation, and removal."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

import agm.packages.install as package_install
from agm.core import dry_run
from agm.packages.activation import PackageActivationError, load_activation_index
from agm.packages.install import (
    PackageInstallError,
    install_directory,
    installed_packages,
    uninstall_package,
)
from agm.packages.record import verify_record


@pytest.fixture(autouse=True)
def _restore_dry_run() -> Generator[None, None, None]:
    previous = dry_run.enabled()
    dry_run.set_enabled(False)
    yield
    dry_run.set_enabled(previous)


def _package(root: Path, name: str, version: str, dependencies: str = "") -> Path:
    root.mkdir()
    (root / name).mkdir()
    (root / "package.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "{version}"\n' + dependencies,
        encoding="utf-8",
    )
    (root / name / "main.agl").write_text("program def main() -> unit = ()\n", encoding="utf-8")
    return root


def test_install_copies_package_writes_record_and_activates_it(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    home = tmp_path / "home"

    installed = install_directory(source, home=home, env={})

    assert installed.root == home / ".agm" / "packages" / "alpha" / "1.0.0"
    assert (installed.root / "alpha" / "main.agl").is_file()
    assert verify_record(installed.root)
    active = load_activation_index(home=home, env={}).packages["alpha"]
    assert active.version == installed.manifest.version


def test_install_resolves_path_dependencies_before_activation(tmp_path: Path) -> None:
    bravo = _package(tmp_path / "bravo", "bravo", "1.2.0")
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1.1", path = "../bravo" }\n',
    )
    home = tmp_path / "home"

    install_directory(alpha, home=home, env={})

    index = load_activation_index(home=home, env={})
    assert set(index.packages) == {"alpha", "bravo"}
    assert index.packages["bravo"].version.major == 1
    assert (home / ".agm" / "packages" / "bravo" / "1.2.0") == (
        home / ".agm" / "packages" / "bravo" / "1.2.0"
    )
    assert bravo.is_dir()


def test_install_uses_an_installed_satisfying_dependency_before_path_source(tmp_path: Path) -> None:
    home = tmp_path / "home"
    install_directory(_package(tmp_path / "older", "bravo", "1.5.0"), home=home, env={})
    installed_dependency = _package(tmp_path / "installed", "bravo", "2.0.0")
    install_directory(installed_dependency, home=home, env={})
    install_directory(_package(tmp_path / "prerelease", "bravo", "2.0.0-alpha"), home=home, env={})
    _package(tmp_path / "path-bravo", "bravo", "1.0.0")
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1.0", path = "../path-bravo" }\n',
    )

    install_directory(alpha, home=home, env={})

    assert load_activation_index(home=home, env={}).packages["bravo"].version.major == 2


def test_editable_install_mounts_live_tree_without_a_record(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    home = tmp_path / "home"

    installed = install_directory(source, home=home, env={}, editable=True)

    assert installed.root == source.resolve()
    assert not (source / "RECORD").exists()
    active = load_activation_index(home=home, env={}).packages["alpha"]
    assert active.editable == source.resolve()


def test_uninstall_verifies_record_then_removes_only_the_requested_version(tmp_path: Path) -> None:
    home = tmp_path / "home"
    one = install_directory(_package(tmp_path / "one", "alpha", "1.0.0"), home=home, env={})
    two = install_directory(_package(tmp_path / "two", "alpha", "2.0.0"), home=home, env={})

    uninstall_package("alpha", home=home, env={})

    assert not two.root.exists()
    assert one.root.exists()
    assert "alpha" not in load_activation_index(home=home, env={}).packages


def test_uninstall_refuses_tampered_record_without_removing_files(tmp_path: Path) -> None:
    home = tmp_path / "home"
    installed = install_directory(
        _package(tmp_path / "source", "alpha", "1.0.0"), home=home, env={}
    )
    module = installed.root / "alpha" / "main.agl"
    module.write_text("tampered", encoding="utf-8")

    with pytest.raises(PackageInstallError, match="integrity"):
        uninstall_package("alpha", home=home, env={})

    assert module.exists()


def test_uninstall_refuses_to_leave_an_active_package_with_unsatisfied_dependencies(
    tmp_path: Path,
) -> None:
    bravo = _package(tmp_path / "bravo", "bravo", "1.0.0")
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", path = "../bravo" }\n',
    )
    home = tmp_path / "home"
    install_directory(alpha, home=home, env={})

    with pytest.raises(PackageInstallError, match="bravo"):
        uninstall_package("bravo", home=home, env={})

    assert (home / ".agm" / "packages" / "bravo" / "1.0.0").is_dir()
    assert set(load_activation_index(home=home, env={}).packages) == {"alpha", "bravo"}
    assert bravo.is_dir()


def test_dry_run_uninstall_refuses_to_leave_unsatisfied_remaining_selection(tmp_path: Path) -> None:
    bravo = _package(tmp_path / "bravo", "bravo", "1.0.0")
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", path = "../bravo" }\n',
    )
    home = tmp_path / "home"
    install_directory(alpha, home=home, env={})
    dry_run.set_enabled(True)

    with pytest.raises(PackageInstallError, match="bravo"):
        uninstall_package("bravo", home=home, env={})

    assert (home / ".agm" / "packages" / "bravo" / "1.0.0").is_dir()
    assert set(load_activation_index(home=home, env={}).packages) == {"alpha", "bravo"}
    assert bravo.is_dir()


def test_uninstall_of_editable_package_only_removes_activation(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    home = tmp_path / "home"
    install_directory(source, home=home, env={}, editable=True)

    uninstall_package("alpha", home=home, env={})

    assert source.exists()
    assert "alpha" not in load_activation_index(home=home, env={}).packages


def test_install_rejects_invalid_and_linked_source_trees(tmp_path: Path) -> None:
    home = tmp_path / "home"
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    with pytest.raises(PackageInstallError):
        install_directory(invalid, home=home, env={})
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    linked = tmp_path / "linked"
    linked.symlink_to(source, target_is_directory=True)
    with pytest.raises(PackageInstallError):
        install_directory(linked, home=home, env={})


def test_install_and_uninstall_refuse_store_paths_redirected_outside_the_store(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    store = home / ".agm" / "packages"
    external = tmp_path / "external"
    external.mkdir()
    store.mkdir(parents=True)
    (store / "alpha").symlink_to(external, target_is_directory=True)

    with pytest.raises(PackageInstallError, match="store"):
        install_directory(source, home=home, env={})

    (store / "alpha").unlink()
    installed = install_directory(source, home=home, env={})
    external_version = external / "1.0.0"
    installed.root.rename(external_version)
    (store / "alpha").rmdir()
    (store / "alpha").symlink_to(external, target_is_directory=True)

    with pytest.raises(PackageInstallError, match="store"):
        uninstall_package("alpha", home=home, env={})

    assert external_version.is_dir()
    assert "alpha" in load_activation_index(home=home, env={}).packages


def test_install_refuses_cyclic_and_mismatched_path_dependencies(tmp_path: Path) -> None:
    home = tmp_path / "home"
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", path = "../bravo" }\n',
    )
    _package(
        tmp_path / "bravo",
        "bravo",
        "1.0.0",
        '\n[dependencies]\nalpha = { version = "1", path = "../alpha" }\n',
    )
    with pytest.raises(PackageInstallError, match="cyclic"):
        install_directory(alpha, home=home, env={})

    wrong = _package(tmp_path / "wrong", "wrong", "1.0.0")
    mismatched = _package(
        tmp_path / "mismatched",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", path = "../wrong" }\n',
    )
    with pytest.raises(PackageInstallError, match="bravo"):
        install_directory(mismatched, home=home, env={})
    assert wrong.is_dir()


def test_install_refuses_tampered_existing_tree_and_cleans_failed_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    installed = install_directory(source, home=home, env={})
    (installed.root / "alpha" / "main.agl").write_text("tampered", encoding="utf-8")
    with pytest.raises(PackageInstallError, match="integrity"):
        install_directory(source, home=home, env={})

    fresh = _package(tmp_path / "fresh", "bravo", "1.0.0")

    def fail_copy(_: Path, destination: Path, **__: object) -> None:
        destination.mkdir()
        raise OSError("full")

    monkeypatch.setattr(package_install.fs, "copy_tree", fail_copy)
    with pytest.raises(PackageInstallError, match="cannot install"):
        install_directory(fresh, home=home, env={})
    assert not (home / ".agm" / "packages" / "bravo" / "1.0.0").exists()

    another = _package(tmp_path / "another", "charlie", "1.0.0")
    monkeypatch.setattr(
        package_install.fs,
        "copy_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("full")),
    )
    with pytest.raises(PackageInstallError, match="cannot install"):
        install_directory(another, home=home, env={})


def test_store_dependency_integrity_error_is_reported(tmp_path: Path) -> None:
    home = tmp_path / "home"
    installed = install_directory(_package(tmp_path / "bravo", "bravo", "1.0.0"), home=home, env={})
    (installed.root / "bravo" / "main.agl").write_text("tampered", encoding="utf-8")
    source = _package(tmp_path / "alpha", "alpha", "1.0.0", '\n[dependencies]\nbravo = "1"\n')
    with pytest.raises(PackageInstallError, match="integrity"):
        install_directory(source, home=home, env={})


def test_url_dependency_fetches_only_to_the_deferred_archive_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", url = "https://example.test/bravo.agmpkg", '
        'hash = "sha256=' + "0" * 64 + '" }\n',
    )
    fetched: list[str] = []

    def fake_fetch(**kwargs: object) -> None:
        fetched.append(str(kwargs["requirement"]))
        handoff = kwargs["handoff"]
        assert callable(handoff)
        handoff(tmp_path / "archive.agmpkg")

    monkeypatch.setattr(package_install, "fetch_archive", fake_fetch)
    with pytest.raises(PackageInstallError, match=r"archive.*bravo >= 1\.0\.0"):
        install_directory(source, home=tmp_path / "home", env={})
    assert fetched == ["bravo >= 1.0.0"]


def test_fetch_and_activation_failures_become_package_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    monkeypatch.setattr(
        package_install,
        "load_activation_index",
        lambda **_: (_ for _ in ()).throw(PackageActivationError("broken")),
    )
    with pytest.raises(PackageInstallError, match="activation"):
        install_directory(source, home=tmp_path / "home", env={})

    monkeypatch.setattr(
        package_install,
        "load_activation_index",
        lambda **_: load_activation_index(home=tmp_path / "other", env={}),
    )
    monkeypatch.setattr(
        package_install,
        "write_activation_index",
        lambda *_, **__: (_ for _ in ()).throw(PackageActivationError("broken")),
    )
    with pytest.raises(PackageInstallError, match="activation"):
        install_directory(source, home=tmp_path / "home", env={})
    assert not (tmp_path / "home" / ".agm" / "packages" / "alpha" / "1.0.0").exists()


def test_installed_package_enumeration_rejects_invalid_store_identity(tmp_path: Path) -> None:
    home = tmp_path / "home"
    assert installed_packages(home=home, env={}) == ()
    root = home / ".agm" / "packages" / "alpha" / "1.0.0"
    (root / "alpha").mkdir(parents=True)
    (root / "package.toml").write_text(
        '[package]\nname = "bravo"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    with pytest.raises(PackageInstallError, match="identity"):
        installed_packages(home=home, env={})


def test_uninstall_reports_unknown_and_activation_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(PackageInstallError, match="not installed"):
        uninstall_package("alpha", home=tmp_path / "home", env={})
    monkeypatch.setattr(
        package_install,
        "load_activation_index",
        lambda **_: (_ for _ in ()).throw(PackageActivationError("broken")),
    )
    with pytest.raises(PackageInstallError, match="activation"):
        uninstall_package("alpha", home=tmp_path / "home", env={})


def test_uninstall_reports_activation_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    home = tmp_path / "home"
    install_directory(source, home=home, env={})
    monkeypatch.setattr(
        package_install,
        "write_activation_index",
        lambda *_, **__: (_ for _ in ()).throw(PackageActivationError("broken")),
    )
    with pytest.raises(PackageInstallError, match="activation"):
        uninstall_package("alpha", home=home, env={})


def test_installed_package_enumeration_skips_non_package_entries_and_bad_manifests(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    store = home / ".agm" / "packages"
    store.mkdir(parents=True)
    (store / "index.toml").write_text("", encoding="utf-8")
    (store / "alpha").mkdir()
    (store / "alpha" / "notes").write_text("not a version", encoding="utf-8")
    assert installed_packages(home=home, env={}) == ()
    broken = store / "alpha" / "1.0.0"
    broken.mkdir()
    (broken / "package.toml").write_text("[package\n", encoding="utf-8")
    with pytest.raises(PackageInstallError, match="cannot load"):
        installed_packages(home=home, env={})


def test_install_refuses_unsatisfied_and_different_existing_manifest(tmp_path: Path) -> None:
    home = tmp_path / "home"
    unsatisfied = _package(
        tmp_path / "unsatisfied", "alpha", "1.0.0", '\n[dependencies]\nbravo = "1"\n'
    )
    with pytest.raises(PackageInstallError, match="bravo"):
        install_directory(unsatisfied, home=home, env={})

    source = _package(tmp_path / "source", "bravo", "1.0.0")
    install_directory(source, home=home, env={})
    (source / "package.toml").write_text(
        '[package]\nname = "bravo"\nversion = "1.0.0"\ndescription = "changed"\n',
        encoding="utf-8",
    )
    with pytest.raises(PackageInstallError, match="disagrees"):
        install_directory(source, home=home, env={})


def test_dry_run_install_validates_the_resulting_activation_without_writing(tmp_path: Path) -> None:
    home = tmp_path / "home"
    install_directory(_package(tmp_path / "bravo-two", "bravo", "2.0.0"), home=home, env={})
    install_directory(
        _package(
            tmp_path / "alpha",
            "alpha",
            "1.0.0",
            '\n[dependencies]\nbravo = "2"\n',
        ),
        home=home,
        env={},
    )
    dry_run.set_enabled(True)

    with pytest.raises(PackageInstallError, match="bravo"):
        install_directory(_package(tmp_path / "bravo-one", "bravo", "1.0.0"), home=home, env={})

    assert not (home / ".agm" / "packages" / "bravo" / "1.0.0").exists()
    assert load_activation_index(home=home, env={}).packages["bravo"].version.major == 2


def test_dry_run_install_selects_prior_transient_dependencies(tmp_path: Path) -> None:
    bravo = _package(tmp_path / "bravo", "bravo", "1.0.0")
    _package(
        tmp_path / "charlie",
        "charlie",
        "1.0.0",
        '\n[dependencies]\nbravo = "1"\n',
    )
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", path = "../bravo" }\n'
        'charlie = { version = "1", path = "../charlie" }\n',
    )
    home = tmp_path / "home"
    dry_run.set_enabled(True)

    installed = install_directory(alpha, home=home, env={})

    assert installed.manifest.name == "alpha"
    assert bravo.is_dir()
    assert not home.exists()

    dry_run.set_enabled(False)
    actual = install_directory(alpha, home=home, env={})

    assert actual == installed
    assert set(load_activation_index(home=home, env={}).packages) == {"alpha", "bravo", "charlie"}


def test_dry_run_url_dependency_never_fetches_or_creates_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", url = "https://example.test/bravo.agmpkg", '
        'hash = "sha256=' + "0" * 64 + '" }\n',
    )
    fetched = False

    def fail_fetch(**_: object) -> None:
        nonlocal fetched
        fetched = True

    monkeypatch.setattr(package_install, "fetch_archive", fail_fetch)
    dry_run.set_enabled(True)

    with pytest.raises(PackageInstallError, match=r"archive.*bravo >= 1\.0\.0"):
        install_directory(source, home=tmp_path / "dry-home", env={})

    assert not fetched
    assert not (tmp_path / "dry-home").exists()


def test_url_fetch_scratch_creation_failure_names_the_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", url = "https://example.test/bravo.agmpkg", '
        'hash = "sha256=' + "0" * 64 + '" }\n',
    )
    monkeypatch.setattr(
        package_install.fs,
        "mkdir",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("full")),
    )

    with pytest.raises(PackageInstallError, match=r"fetch failed.*bravo >= 1\.0\.0"):
        install_directory(source, home=tmp_path / "home", env={})


def test_fetch_failure_and_dry_run_use_clean_dependency_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", url = "https://example.test/bravo.agmpkg", '
        'hash = "sha256=' + "0" * 64 + '" }\n',
    )
    monkeypatch.setattr(
        package_install,
        "fetch_archive",
        lambda **_: (_ for _ in ()).throw(
            package_install.FetchError("fetch failed for bravo >= 1.0.0: offline")
        ),
    )
    with pytest.raises(PackageInstallError, match=r"fetch failed.*bravo >= 1\.0\.0"):
        install_directory(source, home=tmp_path / "home", env={})

    dry_run.set_enabled(True)
    with pytest.raises(PackageInstallError, match=r"archive.*bravo >= 1\.0\.0"):
        install_directory(source, home=tmp_path / "dry-home", env={})


def test_install_and_uninstall_use_dry_run_filesystem_primitives(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    home = tmp_path / "home"
    dry_run.set_enabled(True)

    installed = install_directory(source, home=home, env={})

    assert not installed.root.exists()
    dry_run.set_enabled(False)
    install_directory(source, home=home, env={})
    dry_run.set_enabled(True)
    uninstall_package("alpha", home=home, env={})

    assert (home / ".agm" / "packages" / "alpha" / "1.0.0").exists()
