"""The suite's own isolation from the machine it runs on.

Every other test trusts that AGM resolves its home, project, and shell state
from what that test set up.  These tests verify that trust: a developer with a
real ``~/.agm`` — installed packages, a personal ``config.toml``, prompts — must
get the same results as a clean checkout, and a suite launched from inside a
project directory or a workspace shell must not inherit either.
"""

from __future__ import annotations

import os
from pathlib import Path

from agm.config.context import current_config_context
from agm.config.general import agm_home_dir
from agm.packages.activation import load_activation_index
from tests.conftest import LAUNCH_ENVIRONMENT


def test_the_suite_does_not_run_against_the_launching_home() -> None:
    launched_from = LAUNCH_ENVIRONMENT.get("HOME")

    assert launched_from is not None
    assert os.environ["HOME"] != launched_from
    assert Path.home() != Path(launched_from)


def test_the_agm_home_is_empty_of_installed_state() -> None:
    context = current_config_context()
    home = agm_home_dir(home=context.home)

    assert not home.exists()
    index = load_activation_index(home=context.home)
    assert index.packages == {}
    assert index.commands == {}


def test_project_and_shell_variables_are_not_inherited() -> None:
    for name in ("PROJ_DIR", "REPO_DIR", "TMUX", "TMUX_PANE"):
        assert name not in os.environ
    assert not any(name.startswith("AGM_") for name in os.environ if name != "AGM_STDLIB")


def test_git_identity_does_not_depend_on_a_personal_gitconfig() -> None:
    assert os.environ["GIT_CONFIG_NOSYSTEM"] == "1"
    assert os.environ["GIT_AUTHOR_NAME"]
    assert os.environ["GIT_COMMITTER_EMAIL"]
    assert not (Path(os.environ["HOME"]) / ".gitconfig").exists()
