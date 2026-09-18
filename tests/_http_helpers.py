"""Scripted fake HTTP transport for network-free tests.

``FakeHttp`` is a ``requests.adapters.BaseAdapter`` mounted on a real
``requests.Session`` for both ``http://`` and ``https://``, in the style of
``FakeShell`` (see ``_process_helpers.py``): an ordered list of scripted
outcomes, one per expected request. ``fake_session(outcomes)`` builds such a
session directly; ``install(monkeypatch, outcomes)`` patches
``agm.core.http.open_session`` to return one and hands back the adapter for
assertions. ``assert_complete()`` checks every outcome was consumed.

Each outcome is a mapping, kept JSON-friendly (no Python ``bytes``) so an e2e
``http`` scenario key can reuse the shape verbatim::

    {"expect": {"method": "GET", "url": "https://x/y?a=1",
                "headers": {"accept": "application/json"}, "body": "...",
                "timeout": (5.0, 5.0), "verify": True},
     "status": 200, "headers": {"content-type": "text/plain"}, "body": "hi",
     "charset": "utf-8"}

``expect`` is optional and, when present, is checked against the actual call:
an exact ``method``, an exact ``url``, a case-insensitive header subset, an
exact request ``body`` text, the ``timeout`` tuple given to the transport,
and the ``verify`` flag given to the transport.  An unexpected or
out-of-order request fails the test immediately with a clear message.

A response carries ``status`` (default 200), ``headers`` (a mapping, or a
list of ``(name, value)`` pairs to script repeated headers realistically --
built through a real ``urllib3.HTTPHeaderDict`` so duplicates join with
``", "`` exactly as they do against a real transport), and exactly one body
form: ``body`` (+ optional ``charset``, default UTF-8) or ``body_hex`` (a hex
string, for bytes no text encoding can produce). ``response.cookies`` is
derived from scripted ``set-cookie`` header values, exactly as a real
transport populates it.

``fail: "<kind>"`` (one of ``url``, ``connection``, ``timeout``, ``tls``,
``redirect``) raises the matching ``requests`` exception before any response
is built, modelling a connect-time failure. ``fail_mid_stream: "<kind>"``
(``connection``, ``timeout``, ``tls``, or ``decode``) first yields the
scripted body then raises while it is being streamed, exercising the
failures that ``agm.core.http.perform`` must classify from what ``requests``
wraps them as when they occur mid-stream rather than at connect time.

A 3xx ``status`` with a ``location`` header is an ordinary outcome:
``requests``' own redirect handling sends the next request through this
adapter again, so script each hop as its own list entry.

``send()`` emits ``InsecureRequestWarning`` exactly when the transport is
called with ``verify=False``, like urllib3, so a test can assert
``agm.core.http.perform`` suppresses it precisely then and only for that one
call.
"""

from __future__ import annotations

import email.message
import http.cookies
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import pytest
import requests
import requests.adapters
import requests.structures
import urllib3
import urllib3.exceptions

from agm.core import http as agm_http


class FakeHttp(requests.adapters.BaseAdapter):
    """Fake transport adapter answering scripted requests in order."""

    def __init__(self, outcomes: Sequence[Mapping[str, Any]]) -> None:
        super().__init__()
        self._outcomes = outcomes
        self.sent: list[requests.PreparedRequest] = []
        self.timeouts: list[object] = []
        self.verifies: list[object] = []
        self.streams: list[object] = []
        self.proxies_sent: list[dict[str, str] | None] = []

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: object = None,
        verify: object = True,
        cert: object = None,
        proxies: dict[str, str] | None = None,
    ) -> requests.Response:
        del cert
        index = len(self.sent)
        self.sent.append(request)
        self.timeouts.append(timeout)
        self.verifies.append(verify)
        self.streams.append(stream)
        self.proxies_sent.append(proxies)
        assert index < len(self._outcomes), (
            f"unexpected HTTP request: {request.method} {request.url}"
        )
        outcome = self._outcomes[index]
        _check_expectation(request, outcome.get("expect"), timeout=timeout, verify=verify)
        if verify is False:
            warnings.warn(
                urllib3.exceptions.InsecureRequestWarning(
                    "fake: certificate verification disabled"
                ),
                stacklevel=2,
            )
        if "fail" in outcome:
            raise _connect_failure(outcome["fail"], request.url)
        return _build_response(request, outcome)

    def close(self) -> None:
        pass

    def assert_complete(self) -> None:
        """Assert every scripted outcome was consumed, in order."""
        assert len(self.sent) == len(self._outcomes), (
            f"expected {len(self._outcomes)} HTTP requests, got {len(self.sent)}"
        )


def fake_session(outcomes: Sequence[Mapping[str, Any]]) -> tuple[requests.Session, FakeHttp]:
    """A session built by ``agm.core.http.open_session`` with a fresh ``FakeHttp`` mounted.

    Building through ``open_session`` (rather than a bare ``requests.Session``)
    means every test exercises the seam's real session configuration --
    cookie non-persistence, blocked ``.netrc`` lookup -- not a stand-in for it.
    """
    session = agm_http.open_session()
    adapter = FakeHttp(outcomes)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session, adapter


def install(monkeypatch: pytest.MonkeyPatch, outcomes: Sequence[Mapping[str, Any]]) -> FakeHttp:
    """Patch ``agm.core.http.open_session`` to return a faked session."""
    session, adapter = fake_session(outcomes)
    monkeypatch.setattr("agm.core.http.open_session", lambda: session)
    return adapter


def _check_expectation(
    request: requests.PreparedRequest,
    expect: Mapping[str, Any] | None,
    *,
    timeout: object,
    verify: object,
) -> None:
    if expect is None:
        return
    if "method" in expect:
        assert request.method == expect["method"], (
            f"expected method {expect['method']!r}, got {request.method!r}"
        )
    if "url" in expect:
        assert request.url == expect["url"], f"expected url {expect['url']!r}, got {request.url!r}"
    if "headers" in expect:
        for name, value in expect["headers"].items():
            actual = request.headers.get(name)
            assert actual == value, f"expected header {name}={value!r}, got {actual!r}"
    if "body" in expect:
        body = request.body
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        assert body == expect["body"], f"expected body {expect['body']!r}, got {body!r}"
    if "timeout" in expect:
        assert timeout == expect["timeout"], (
            f"expected timeout {expect['timeout']!r}, got {timeout!r}"
        )
    if "verify" in expect:
        assert verify == expect["verify"], f"expected verify {expect['verify']!r}, got {verify!r}"


_CONNECT_FAILURES: dict[str, Callable[[str], requests.exceptions.RequestException]] = {
    "url": lambda url: requests.exceptions.InvalidURL(f"fake: unsupported url {url}"),
    "connection": lambda url: requests.exceptions.ConnectionError(
        f"fake: connection refused: {url}"
    ),
    "timeout": lambda url: requests.exceptions.ConnectTimeout(f"fake: connect timed out: {url}"),
    "tls": lambda url: requests.exceptions.SSLError(f"fake: certificate verify failed: {url}"),
    "redirect": lambda url: requests.exceptions.TooManyRedirects(
        f"fake: too many redirects: {url}"
    ),
}


def _connect_failure(kind: str, url: str | None) -> requests.exceptions.RequestException:
    assert kind in _CONNECT_FAILURES, f"unknown fake HTTP failure kind: {kind!r}"
    return _CONNECT_FAILURES[kind](url or "")


_MID_STREAM_FAILURES: dict[str, Callable[[], BaseException]] = {
    "connection": lambda: urllib3.exceptions.ProtocolError("fake: connection reset mid-stream"),
    "timeout": lambda: urllib3.exceptions.ReadTimeoutError(
        None, "fake://mid-stream", "fake: read timed out"
    ),
    "tls": lambda: urllib3.exceptions.SSLError("fake: tls failed mid-stream"),
    "decode": lambda: urllib3.exceptions.DecodeError("fake: could not decode mid-stream"),
}


def _build_response(
    request: requests.PreparedRequest, outcome: Mapping[str, Any]
) -> requests.Response:
    response = requests.Response()
    response.status_code = outcome.get("status", 200)
    header_dict = urllib3.HTTPHeaderDict()
    header_spec = outcome.get("headers", {})
    items = header_spec.items() if isinstance(header_spec, Mapping) else header_spec
    for name, value in items:
        header_dict.add(name, value)
    response.headers = requests.structures.CaseInsensitiveDict(header_dict)
    response.url = request.url or ""
    response.request = request
    body, mid_stream_failure = _body_and_failure(outcome)
    message = email.message.Message()
    for name, value in header_dict.items():
        message.add_header(name, value)
    response.raw = _FakeRaw(body, mid_stream_failure, message)
    _apply_response_cookies(response, header_dict.items())
    return response


def _apply_response_cookies(
    response: requests.Response, header_items: Iterable[tuple[str, str]]
) -> None:
    """Populate ``response.cookies`` from scripted ``set-cookie`` header values."""
    for name, value in header_items:
        if name.lower() != "set-cookie":
            continue
        parsed: http.cookies.SimpleCookie[str] = http.cookies.SimpleCookie()
        parsed.load(value)
        for morsel in parsed.values():
            response.cookies.set(morsel.key, morsel.value)


def _body_and_failure(outcome: Mapping[str, Any]) -> tuple[bytes, BaseException | None]:
    if "body_hex" in outcome:
        body = bytes.fromhex(outcome["body_hex"])
    elif "body" in outcome:
        body = outcome["body"].encode(outcome.get("charset", "utf-8"))
    else:
        body = b""
    kind = outcome.get("fail_mid_stream")
    failure = _MID_STREAM_FAILURES[kind]() if kind is not None else None
    return body, failure


class _OriginalResponse:
    """The ``http.client``-response shim ``http.cookiejar`` needs off ``.raw``.

    ``requests``' own redirect handling extracts a hop's ``Set-Cookie``
    straight from ``raw._original_response.msg`` (see
    ``requests.cookies.extract_cookies_to_jar``), bypassing the adapter
    entirely, so the fake must carry one too for that hand-off to work.
    """

    def __init__(self, msg: email.message.Message) -> None:
        self.msg = msg


class _FakeRaw:
    """Minimal urllib3-response surface that ``Response.iter_content`` needs."""

    def __init__(
        self, body: bytes, mid_stream_failure: BaseException | None, msg: email.message.Message
    ) -> None:
        self._body = body
        self._mid_stream_failure = mid_stream_failure
        self._original_response = _OriginalResponse(msg)

    def stream(self, chunk_size: int, decode_content: bool = True) -> Iterator[bytes]:
        del chunk_size, decode_content
        if self._body:
            yield self._body
        if self._mid_stream_failure is not None:
            raise self._mid_stream_failure

    def read(self, amt: int | None = None, decode_content: bool = False) -> bytes:
        del amt, decode_content
        return b""

    def close(self) -> None:
        pass

    def release_conn(self) -> None:
        pass
