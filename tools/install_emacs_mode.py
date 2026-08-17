"""Install the AgL Emacs mode into the user's Emacs.

Stages the Elisp in ``config/emacs/`` as a package tar and installs it with
Emacs's own package machinery, so package.el generates the autoloads and
byte-compiles the sources.  The package version is taken from
:mod:`agm.version`, keeping it in lockstep with AGM.

The installer never edits the user's init file.  On Emacs 27+ an installed
package activates at the next start, so opening a ``.agl`` file selects
``agl-mode`` with no configuration.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, cast

from agm.version import AGM_VERSION

PACKAGE_NAME = "agl-mode"
PACKAGE_SUMMARY = "Major mode for AgL source files"
EMACS_REQUIREMENT = '((emacs "27.1"))'


class _InstallArgs(Protocol):
    force: bool
    prefix: str | None


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser, mirroring the config installer's shape."""
    parser = argparse.ArgumentParser(prog="python tools/install_emacs_mode.py")
    parser.add_argument(
        "prefix",
        nargs="?",
        help="Install into PREFIX's Emacs package directory instead of $HOME's.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Accepted for parity with the config installer; a reinstall always replaces.",
    )
    return parser


def package_sources(emacs_dir: Path) -> list[Path]:
    """Return the Elisp files that belong in the installed package.

    The ERT suites under ``tests/`` are development files and are excluded.
    """
    return sorted(path for path in emacs_dir.glob("*.el") if path.is_file())


def package_descriptor(version: str) -> str:
    """Return the ``agl-mode-pkg.el`` contents for VERSION."""
    return (
        f'(define-package "{PACKAGE_NAME}" "{version}"\n'
        f'  "{PACKAGE_SUMMARY}"\n'
        f"  '{EMACS_REQUIREMENT})\n"
    )


def build_package_tar(emacs_dir: Path, staging: Path, version: str) -> Path:
    """Stage the package under STAGING and return the built tar path.

    package.el expects a tar holding a single ``<name>-<version>/``
    directory, so the sources and the generated descriptor are placed there.
    """
    package_dir = staging / f"{PACKAGE_NAME}-{version}"
    package_dir.mkdir(parents=True)
    for source in package_sources(emacs_dir):
        shutil.copy2(source, package_dir / source.name)
    (package_dir / f"{PACKAGE_NAME}-pkg.el").write_text(
        package_descriptor(version), encoding="utf-8"
    )

    # Built with tar(1) rather than `tarfile`: package.el's untar check is
    # strict about how the package directory is recorded, and the archives
    # `tarfile` writes do not satisfy it.
    archive = staging / f"{PACKAGE_NAME}-{version}.tar"
    tar = shutil.which("tar")
    if tar is None:
        raise SystemExit("Error: tar was not found on PATH.")
    completed = subprocess.run(
        [tar, "cf", str(archive), package_dir.name],
        cwd=staging,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"Error: could not build the package archive:\n{completed.stderr.strip()}")
    return archive


def install_elisp(archive: Path) -> str:
    """Return the Elisp that removes any installed copy and installs ARCHIVE."""
    return f"""
(require 'package)
(package-initialize)
(dolist (installed (cdr (assq '{PACKAGE_NAME} package-alist)))
  (ignore-errors (package-delete installed t)))
(package-install-file "{archive}")
(princ (format "into %s\\n" package-user-dir))
"""


def run_emacs_install(archive: Path, *, home: Path | None) -> str:
    """Install ARCHIVE with ``emacs --batch``; return its output.

    *home* overrides ``HOME`` so the package lands under that prefix's
    ``package-user-dir`` — the parity the config installer's PREFIX gives.
    """
    emacs = shutil.which("emacs")
    if emacs is None:
        raise SystemExit(
            "Error: emacs was not found on PATH; install Emacs, or skip the "
            "Emacs mode (`just install` skips it automatically)."
        )

    with TemporaryDirectory(prefix="agm-emacs-install-") as scratch:
        script = Path(scratch) / "install.el"
        script.write_text(install_elisp(archive), encoding="utf-8")
        # Inherit the caller's environment and override only HOME: a
        # minimal environment leaves Emacs without the settings it expects.
        env = None
        if home is not None:
            env = dict(os.environ)
            env["HOME"] = str(home)
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [emacs, "--batch", "--load", str(script)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    if completed.returncode != 0:
        raise SystemExit(
            f"Error: emacs failed to install the package:\n{completed.stderr.strip()}"
        )
    return completed.stdout.strip() or completed.stderr.strip()


def manual_setup_notice(emacs_dir: Path) -> str:
    """Return the wiring instructions for setups that do not use package.el."""
    return (
        "Doom, straight.el, and other non-package.el setups can load the mode "
        f"directly instead:\n"
        f'  (add-to-list \'load-path "{emacs_dir}")\n'
        "  (require 'agl-mode)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Install the Emacs mode, printing what was installed."""
    args = cast(_InstallArgs, build_parser().parse_args(argv))
    del args.force  # A reinstall always replaces; accepted for parity.

    repo_root = Path(__file__).resolve().parents[1]
    emacs_dir = repo_root / "config" / "emacs"
    if not emacs_dir.is_dir():
        raise SystemExit(f"Error: {emacs_dir} does not exist.")

    home = None if args.prefix is None else Path(args.prefix).resolve()
    if home is not None:
        home.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="agm-emacs-stage-") as staging:
        archive = build_package_tar(emacs_dir, Path(staging), AGM_VERSION)
        output = run_emacs_install(archive, home=home)

    print(f"Installed {PACKAGE_NAME} {AGM_VERSION}")
    if output:
        print(output)
    print()
    print(manual_setup_notice(emacs_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
