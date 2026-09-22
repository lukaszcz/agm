"""Scripted TypeSafe transport and ``sysone`` package mounting for network-free tests.

``JevTransport`` is an ``httpx2.MockTransport`` answering an ordered list of
outcomes, one per expected request, in the style of ``_http_helpers.FakeHttp``.
Each outcome is a JSON-friendly mapping, so an e2e scenario key can reuse the
shape verbatim::

    {"expect": {"body": {"state": "...", "model": "jev-latest", "questions": {...}},
                "url": "https://api.typesafe.ai/v1/systemone",
                "headers": {"authorization": "Bearer test-key"}, "timeout": 5.0},
     "status": 200, "json": {"model": "jev-1", "answers": {...}, "usage": {...}},
     "headers": {"x-typesafe-request-id": "req-1"}}

``expect`` is optional, and so is each of its keys: ``body`` is the exact
decoded JSON request body; ``url`` the exact URL; ``headers`` a
case-insensitive subset (a ``None`` value asserts absence); ``timeout`` the
seconds every phase of the request's timeout carries (``None`` for none).
A response carries ``status`` (default 200), ``json`` (the body; omitted for
an empty one), and ``headers``, such as a request id or ``retry-after``.
``fail: "connection" | "timeout"`` instead raises ``httpx2.ConnectError`` or
``httpx2.ReadTimeout``. An unexpected or mismatched request fails the test
immediately through ``pytest.fail``, whose ``BaseException`` escapes the SDK's
retries and the extern boundary; ``assert_complete()`` checks every outcome
was consumed.

``jev_roots()`` mounts ``packages/sysone`` as ``assemble_roots`` does for an
installed package. ``install_jev_transport()`` loads the ``sysone/jev``
companion into a caller's registry and swaps its ``open_client`` seam for one
whose clients send through a scripted transport; ``mount_jev()`` does so for a
fresh driver.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import httpx2
import pytest

from agm.agl.modules.ids import ModuleId
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PipelineDriver
from agm.agl.runtime.externs import ExternRegistry
from tests._agl_helpers import package_roots, run_inline_command
from tests._package_helpers import package_info

SYSONE_ROOT = Path(__file__).resolve().parents[1] / "packages" / "sysone"
JEV_MODULE = ModuleId(("sysone", "jev"))
TEST_API_KEY = "test-key"

_TIMEOUT_PHASES = ("connect", "read", "write", "pool")


class JevTransport(httpx2.MockTransport):
    """Answers scripted TypeSafe requests in order."""

    def __init__(self, outcomes: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(self._answer)
        self._outcomes = list(outcomes)
        self.requests: list[httpx2.Request] = []

    def _answer(self, request: httpx2.Request) -> httpx2.Response:
        index = len(self.requests)
        self.requests.append(request)
        if index >= len(self._outcomes):
            pytest.fail(f"unexpected request #{index + 1}: {request.method} {request.url}")
        outcome = self._outcomes[index]
        expected = outcome.get("expect")
        if expected is not None:
            _check_expectation(index, request, expected)
        failure = outcome.get("fail")
        if failure == "connection":
            raise httpx2.ConnectError("scripted connection failure", request=request)
        if failure == "timeout":
            raise httpx2.ReadTimeout("scripted timeout", request=request)
        headers = dict(outcome.get("headers", {}))
        status = int(outcome.get("status", 200))
        if "json" not in outcome:
            return httpx2.Response(status, headers=headers)
        return httpx2.Response(status, headers=headers, json=outcome["json"])

    def assert_complete(self) -> None:
        """Every scripted outcome was requested."""
        remaining = len(self._outcomes) - len(self.requests)
        assert remaining == 0, f"{remaining} scripted TypeSafe request(s) never sent"


def _check_expectation(index: int, request: httpx2.Request, expected: Mapping[str, Any]) -> None:
    wanted: dict[str, Any] = {}
    actual: dict[str, Any] = {}
    if "body" in expected:
        wanted["body"] = expected["body"]
        actual["body"] = json.loads(request.content)
    if "url" in expected:
        wanted["url"] = expected["url"]
        actual["url"] = str(request.url)
    if "headers" in expected:
        wanted["headers"] = {name.lower(): value for name, value in expected["headers"].items()}
        actual["headers"] = {name: request.headers.get(name) for name in wanted["headers"]}
    if "timeout" in expected:
        wanted["timeout"] = dict.fromkeys(_TIMEOUT_PHASES, expected["timeout"])
        phases = request.extensions.get("timeout", {})
        actual["timeout"] = dict.fromkeys(_TIMEOUT_PHASES) | dict(phases)
    if wanted != actual:
        pytest.fail(
            f"TypeSafe request #{index + 1} mismatch:\nexpected: {wanted!r}\nactual:   {actual!r}"
        )


def jev_roots() -> RootSet:
    """Roots mounting the repository ``sysone`` package over the repository stdlib."""
    return package_roots(package_info(SYSONE_ROOT), cwd=SYSONE_ROOT)


def load_jev_companion(driver: PipelineDriver, registry: ExternRegistry) -> ModuleType:
    """Load the ``sysone/jev`` companion into *registry* through *driver*."""
    result = run_inline_command(driver, "import sysone/jev", roots=jev_roots())
    assert result.ok, result.diagnostics
    companion = registry.loaded_companion(JEV_MODULE)
    assert companion is not None
    return companion


def install_jev_transport(
    monkeypatch: pytest.MonkeyPatch,
    driver: PipelineDriver,
    registry: ExternRegistry,
    outcomes: Sequence[Mapping[str, Any]],
) -> JevTransport:
    """Script *registry*'s ``sysone/jev`` clients with *outcomes*.

    Preloads the companion into *registry* (which *driver* owns) and swaps its
    ``open_client`` seam. The SDK's API key is ``TEST_API_KEY``.
    """
    monkeypatch.setenv("TYPESAFE_API_KEY", TEST_API_KEY)
    companion = load_jev_companion(driver, registry)
    transport = JevTransport(outcomes)
    real_open = companion.open_client
    monkeypatch.setattr(
        companion, "open_client", lambda settings: real_open(settings, transport=transport)
    )
    return transport


class JevMount(NamedTuple):
    driver: PipelineDriver
    transport: JevTransport
    companion: ModuleType


def mount_jev(monkeypatch: pytest.MonkeyPatch, outcomes: Sequence[Mapping[str, Any]]) -> JevMount:
    """A fresh driver whose ``sysone/jev`` clients are answered by scripted *outcomes*."""
    registry = ExternRegistry()
    driver = PipelineDriver(extern_registry=registry)
    transport = install_jev_transport(monkeypatch, driver, registry, outcomes)
    companion = registry.loaded_companion(JEV_MODULE)
    assert companion is not None
    return JevMount(driver, transport, companion)
