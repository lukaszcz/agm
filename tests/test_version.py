"""Tests for AGM's release-version contract."""

from __future__ import annotations

import re

from agm.version import AGM_VERSION


def test_agm_version_has_plain_semver_shape() -> None:
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", AGM_VERSION)
