"""Sandbox template regression tests."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SANDBOX_DIR = REPO_ROOT / "sandbox"

COMMON_POLICY = {
    "network": {"deniedDomains": []},
    "filesystem": {
        "denyRead": ["~/.ssh", "~/.aws", "~/.gnupg"],
        "denyWrite": [".env", ".env.*", "*.pem", "*.key"],
    },
}


def _load_template(name: str) -> dict[str, object]:
    return json.loads((SANDBOX_DIR / name).read_text())


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
