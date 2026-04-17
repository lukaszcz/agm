"""Sandbox template regression tests."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
SANDBOX_DIR = CONFIG_DIR / "sandbox"

COMMON_POLICY = {
    "network": {"deniedDomains": []},
    "filesystem": {
        "denyRead": ["~/.ssh", "~/.aws", "~/.gnupg"],
        "denyWrite": [".env", ".env.*", "*.pem", "*.key"],
    },
}


def _load_template(name: str) -> dict[str, object]:
    return json.loads((SANDBOX_DIR / name).read_text())


def test_repo_config_toml_exists_and_is_empty() -> None:
    assert (CONFIG_DIR / "config.toml").read_text() == ""


def test_codex_template_matches_pi_policy_with_openai_api_access() -> None:
    template = _load_template("codex.json")

    allowed_domains = template["network"]["allowedDomains"]

    assert "api.openai.com" in allowed_domains
    assert template["network"]["deniedDomains"] == []
    assert template["filesystem"] == {
        **COMMON_POLICY["filesystem"],
        "allowWrite": [".", "/tmp", "~/.codex"],
    }


def test_claude_template_matches_pi_policy_with_claude_code_domains() -> None:
    template = _load_template("claude.json")

    allowed_domains = template["network"]["allowedDomains"]

    assert "api.anthropic.com" in allowed_domains
    assert template["network"]["deniedDomains"] == []
    assert template["filesystem"] == {
        **COMMON_POLICY["filesystem"],
        "allowWrite": [".", "/tmp", "~/.claude"],
    }
