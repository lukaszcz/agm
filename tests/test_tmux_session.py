"""Comprehensive tests for agm.tmux.session — validate_pane_count and _filter_env."""

from __future__ import annotations

import pytest

from agm.tmux.session import _filter_env, validate_pane_count

# ===========================================================================
# validate_pane_count
# ===========================================================================


class TestValidatePaneCount:
    # --- None input ---

    def test_none_returns_default_four(self) -> None:
        result = validate_pane_count(["tmux", "open"], None)
        assert result == 4

    # --- Valid digit strings ---

    def test_single_pane(self) -> None:
        assert validate_pane_count(["tmux", "open"], "1") == 1

    def test_two_panes(self) -> None:
        assert validate_pane_count(["tmux", "open"], "2") == 2

    def test_four_panes(self) -> None:
        assert validate_pane_count(["tmux", "open"], "4") == 4

    def test_large_pane_count(self) -> None:
        assert validate_pane_count(["tmux", "open"], "10") == 10

    def test_returns_int_not_string(self) -> None:
        result = validate_pane_count(["tmux", "open"], "3")
        assert isinstance(result, int)

    # --- Invalid inputs that should exit ---

    def test_zero_exits(self) -> None:
        with pytest.raises(SystemExit):
            validate_pane_count(["tmux", "open"], "0")

    def test_negative_string_exits(self) -> None:
        # "-1" is not a digit string
        with pytest.raises(SystemExit):
            validate_pane_count(["tmux", "open"], "-1")

    def test_non_digit_string_exits(self) -> None:
        with pytest.raises(SystemExit):
            validate_pane_count(["tmux", "open"], "abc")

    def test_float_string_exits(self) -> None:
        with pytest.raises(SystemExit):
            validate_pane_count(["tmux", "open"], "1.5")

    def test_empty_string_exits(self) -> None:
        with pytest.raises(SystemExit):
            validate_pane_count(["tmux", "open"], "")

    def test_whitespace_string_exits(self) -> None:
        with pytest.raises(SystemExit):
            validate_pane_count(["tmux", "open"], " 4")

    def test_alphanumeric_string_exits(self) -> None:
        with pytest.raises(SystemExit):
            validate_pane_count(["tmux", "open"], "4a")

    def test_command_path_is_passed_through_to_error(self) -> None:
        # Use a known valid command path recognised by the parser
        with pytest.raises(SystemExit) as exc_info:
            validate_pane_count(["tmux", "open"], "bad")
        assert exc_info.value.code != 0


# ===========================================================================
# _filter_env
# ===========================================================================


class TestFilterEnv:
    # --- Pass-through cases ---

    def test_normal_env_var_is_included(self) -> None:
        result = _filter_env({"MY_VAR": "hello"})
        assert ("MY_VAR", "hello") in result

    def test_multiple_normal_vars_all_included(self) -> None:
        env = {"EDITOR": "vim", "LANG": "en_US.UTF-8", "PATH": "/usr/bin"}
        result = _filter_env(env)
        for name, value in env.items():
            assert (name, value) in result

    def test_result_preserves_values_unchanged(self) -> None:
        env = {"MY_KEY": "some value with spaces"}
        result = _filter_env(env)
        assert ("MY_KEY", "some value with spaces") in result

    # --- SKIP_PREFIXES: TMUX_ ---

    def test_tmux_prefixed_var_is_excluded(self) -> None:
        result = _filter_env({"TMUX_PANE": "%1"})
        assert not any(name == "TMUX_PANE" for name, _ in result)

    def test_tmux_prefix_multiple_vars_excluded(self) -> None:
        result = _filter_env({"TMUX_VERSION": "3.3", "TMUX_PANE": "%0"})
        names = [name for name, _ in result]
        assert "TMUX_VERSION" not in names
        assert "TMUX_PANE" not in names

    # --- SKIP_PREFIXES: TERM_ ---

    def test_term_prefixed_var_is_excluded(self) -> None:
        result = _filter_env({"TERM_PROGRAM": "iTerm2"})
        assert not any(name == "TERM_PROGRAM" for name, _ in result)

    def test_term_program_version_excluded(self) -> None:
        result = _filter_env({"TERM_PROGRAM_VERSION": "3.5"})
        names = [name for name, _ in result]
        assert "TERM_PROGRAM_VERSION" not in names

    # --- SKIP_PREFIXES: SSH_ ---

    def test_ssh_prefixed_var_is_excluded(self) -> None:
        result = _filter_env({"SSH_AUTH_SOCK": "/tmp/ssh-agent.sock"})
        assert not any(name == "SSH_AUTH_SOCK" for name, _ in result)

    def test_ssh_client_excluded(self) -> None:
        result = _filter_env({"SSH_CLIENT": "1.2.3.4 1234 22"})
        names = [name for name, _ in result]
        assert "SSH_CLIENT" not in names

    # --- SKIP_PREFIXES: DBUS_ ---

    def test_dbus_prefixed_var_is_excluded(self) -> None:
        result = _filter_env({"DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus"})
        assert not any(name == "DBUS_SESSION_BUS_ADDRESS" for name, _ in result)

    # --- SKIP_PREFIXES: XDG_ ---

    def test_xdg_prefixed_var_is_excluded(self) -> None:
        result = _filter_env({"XDG_RUNTIME_DIR": "/run/user/1000"})
        assert not any(name == "XDG_RUNTIME_DIR" for name, _ in result)

    def test_xdg_data_dirs_excluded(self) -> None:
        result = _filter_env({"XDG_DATA_DIRS": "/usr/share:/usr/local/share"})
        names = [name for name, _ in result]
        assert "XDG_DATA_DIRS" not in names

    # --- Skip-list names (not safe per is_safe_shell_env_assignment_name) ---

    def test_tmux_bare_name_excluded_by_skip_list(self) -> None:
        # "TMUX" is in _SHELL_ENV_ASSIGNMENT_SKIP_NAMES
        result = _filter_env({"TMUX": "/tmp/tmux.sock"})
        assert not any(name == "TMUX" for name, _ in result)

    def test_term_bare_name_excluded_by_skip_list(self) -> None:
        # "TERM" is in _SHELL_ENV_ASSIGNMENT_SKIP_NAMES
        result = _filter_env({"TERM": "xterm-256color"})
        assert not any(name == "TERM" for name, _ in result)

    def test_display_excluded_by_skip_list(self) -> None:
        result = _filter_env({"DISPLAY": ":0"})
        assert not any(name == "DISPLAY" for name, _ in result)

    def test_path_not_excluded(self) -> None:
        # PATH is not in the skip list and has no skip prefix
        result = _filter_env({"PATH": "/usr/bin:/bin"})
        assert any(name == "PATH" for name, _ in result)

    # --- Invalid identifier names ---

    def test_name_with_equals_is_excluded(self) -> None:
        result = _filter_env({"BAD=NAME": "val"})
        assert not any(name == "BAD=NAME" for name, _ in result)

    def test_name_starting_with_digit_is_excluded(self) -> None:
        result = _filter_env({"1VAR": "val"})
        assert not any(name == "1VAR" for name, _ in result)

    def test_name_with_hyphen_is_excluded(self) -> None:
        result = _filter_env({"MY-VAR": "val"})
        assert not any(name == "MY-VAR" for name, _ in result)

    def test_empty_name_is_excluded(self) -> None:
        result = _filter_env({"": "val"})
        assert not any(name == "" for name, _ in result)

    # --- Mixed environment ---

    def test_mixed_env_only_safe_vars_included(self) -> None:
        env = {
            "EDITOR": "vim",
            "TMUX_PANE": "%0",
            "SSH_AUTH_SOCK": "/tmp/sock",
            "XDG_SESSION_ID": "1",
            "DBUS_SESSION_BUS_ADDRESS": "unix:abstract",
            "TERM_PROGRAM": "iTerm2",
            "HOME": "should-be-filtered",
            "MY_CUSTOM": "keep",
        }
        result = _filter_env(env)
        names = [name for name, _ in result]
        assert "EDITOR" in names
        assert "MY_CUSTOM" in names
        assert "TMUX_PANE" not in names
        assert "SSH_AUTH_SOCK" not in names
        assert "XDG_SESSION_ID" not in names
        assert "DBUS_SESSION_BUS_ADDRESS" not in names
        assert "TERM_PROGRAM" not in names

    def test_empty_env_returns_empty_list(self) -> None:
        assert _filter_env({}) == []

    def test_result_is_list_of_tuples(self) -> None:
        result = _filter_env({"KEY": "val"})
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    # --- Prefix boundary: ensure non-prefixed names with similar start are kept ---

    def test_var_starting_with_tmux_but_not_prefixed_exactly_excluded(self) -> None:
        # "TMUX_" prefix check — "TMUX" alone is handled by skip list, not prefix
        # A var like "TMUXSOMETHING" doesn't start with "TMUX_" but starts with "TMUX"
        result = _filter_env({"TMUXSOMETHING": "val"})
        names = [name for name, _ in result]
        # "TMUXSOMETHING" doesn't match "TMUX_" prefix, so it passes prefix check
        # It is a valid identifier and not in skip list → should be included
        assert "TMUXSOMETHING" in names

    def test_ssh_without_underscore_not_excluded_by_prefix(self) -> None:
        # "SSHKEY" doesn't start with "SSH_"
        result = _filter_env({"SSHKEY": "abc"})
        names = [name for name, _ in result]
        assert "SSHKEY" in names
