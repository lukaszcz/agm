"""Tests for agm.commands.run pure functions."""

from __future__ import annotations

from agm.commands.run import (
    _normalize_systemd_limit,
    _systemd_run_prefix,
    _systemd_scope_name,
    normalize_run_command,
)


class TestNormalizeRunCommand:
    """Tests for normalize_run_command."""

    def test_strips_leading_double_dash(self) -> None:
        assert normalize_run_command(["--", "echo", "hi"]) == ["echo", "hi"]

    def test_leaves_command_unchanged_without_leading_double_dash(self) -> None:
        assert normalize_run_command(["echo", "hi"]) == ["echo", "hi"]

    def test_empty_list_returns_empty(self) -> None:
        assert normalize_run_command([]) == []

    def test_only_double_dash_returns_empty(self) -> None:
        assert normalize_run_command(["--"]) == []

    def test_double_dash_not_at_start_is_left_alone(self) -> None:
        assert normalize_run_command(["echo", "--", "arg"]) == ["echo", "--", "arg"]

    def test_multiple_double_dashes_strips_only_first(self) -> None:
        assert normalize_run_command(["--", "--", "echo"]) == ["--", "echo"]


class TestNormalizeSystemdLimit:
    """Tests for _normalize_systemd_limit."""

    def test_unlimited_returns_infinity(self) -> None:
        assert _normalize_systemd_limit("unlimited") == "infinity"

    def test_unlimited_case_insensitive(self) -> None:
        assert _normalize_systemd_limit("UNLIMITED") == "infinity"
        assert _normalize_systemd_limit("Unlimited") == "infinity"

    def test_unlimited_with_whitespace(self) -> None:
        assert _normalize_systemd_limit("  unlimited  ") == "infinity"

    def test_other_values_unchanged(self) -> None:
        assert _normalize_systemd_limit("20G") == "20G"
        assert _normalize_systemd_limit("0") == "0"
        assert _normalize_systemd_limit("infinity") == "infinity"


class TestSystemdRunPrefix:
    """Tests for _systemd_run_prefix."""

    def test_includes_base_flags(self) -> None:
        prefix = _systemd_run_prefix(memory_limit=None, swap_limit=None)
        assert "systemd-run" in prefix
        assert "--user" in prefix
        assert "--scope" in prefix
        assert "-q" in prefix
        assert "-p" in prefix
        assert "Delegate=yes" in prefix

    def test_adds_memory_max_when_memory_limit_set(self) -> None:
        prefix = _systemd_run_prefix(memory_limit="10G", swap_limit=None)
        assert "MemoryMax=10G" in prefix

    def test_adds_swap_max_when_swap_limit_set(self) -> None:
        prefix = _systemd_run_prefix(memory_limit=None, swap_limit="2G")
        assert "MemorySwapMax=2G" in prefix

    def test_normalizes_unlimited_to_infinity(self) -> None:
        prefix = _systemd_run_prefix(memory_limit="unlimited", swap_limit=None)
        assert "MemoryMax=infinity" in prefix

    def test_omits_memory_max_when_none(self) -> None:
        prefix = _systemd_run_prefix(memory_limit=None, swap_limit="1G")
        assert not any("MemoryMax" in item for item in prefix)

    def test_omits_swap_max_when_none(self) -> None:
        prefix = _systemd_run_prefix(memory_limit="10G", swap_limit=None)
        assert not any("MemorySwapMax" in item for item in prefix)

    def test_includes_both_limits(self) -> None:
        prefix = _systemd_run_prefix(memory_limit="8G", swap_limit="4G")
        assert "MemoryMax=8G" in prefix
        assert "MemorySwapMax=4G" in prefix


class TestSystemdScopeName:
    """Tests for _systemd_scope_name."""

    def test_returns_agm_run_prefix(self) -> None:
        name = _systemd_scope_name()
        assert name.startswith("agm-run-")

    def test_ends_with_scope_suffix(self) -> None:
        name = _systemd_scope_name()
        assert name.endswith(".scope")

    def test_names_are_unique(self) -> None:
        names = {_systemd_scope_name() for _ in range(20)}
        assert len(names) == 20

    def test_hex_part_is_32_chars(self) -> None:
        name = _systemd_scope_name()
        # format: agm-run-{32 hex chars}.scope
        hex_part = name.removeprefix("agm-run-").removesuffix(".scope")
        assert len(hex_part) == 32
        assert all(c in "0123456789abcdef" for c in hex_part)