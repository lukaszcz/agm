"""HTTP transport seam over ``requests``.

Builds and streams one request through a pooled session, classifies transport
failures into a small typed error, and applies the strict charset-decoding
rule the standard library's HTTP companion relies on.
"""

from __future__ import annotations

import http.cookiejar
import re
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import requests
import requests.auth
import requests.cookies
import requests.exceptions
import urllib3.exceptions

from agm.core import fs

TransportErrorKind = Literal["url", "connection", "timeout", "tls", "redirect", "request"]

# A charset value, quoted or not, captured whole (group 0): a named group
# would type as ``str | Any`` under strict mypy regardless of the pattern.
_CHARSET_PATTERN = re.compile(r"(?<=charset=)[^;]+", re.IGNORECASE)
_CHUNK_SIZE = 64 * 1024

# RFC 9110 tchar: the only characters a method token may contain.
_METHOD_TOKEN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")

# The same header validity rules ``requests.utils.check_header_validity`` applies.
_HEADER_NAME = re.compile(r"^[^:\s][^:\r\n]*$")
_HEADER_VALUE = re.compile(r"^\S[^\r\n]*$|^$")


class _NoNetrcAuth(requests.auth.AuthBase):
    """A present-but-inert session auth that blocks ``requests``' ``.netrc`` lookup.

    ``Session.prepare_request`` only consults ``~/.netrc`` when neither the
    request nor the session declares an auth; installing this as the
    session's auth defeats that check without adding anything to the
    request. An explicit request-level auth still overrides it.
    """

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        return r


class _Session(requests.Session):
    """A session that never re-derives credentials from ``~/.netrc`` on redirect."""

    def rebuild_auth(
        self, prepared_request: requests.PreparedRequest, response: requests.Response
    ) -> None:
        """Strip a leaked ``Authorization`` header across hosts; never consult ``.netrc``."""
        headers = prepared_request.headers
        previous_url = response.request.url if response.request is not None else None
        if (
            "Authorization" in headers
            and previous_url is not None
            and self.should_strip_auth(previous_url, prepared_request.url)
        ):
            del headers["Authorization"]


def open_session() -> requests.Session:
    """Build a pooled session for one interpreter run.

    Honours environment proxy variables and the CA-bundle variables
    (``requests``' own environment defaults, which is what the SRT sandbox
    injects) but never consults ``~/.netrc`` for credentials (see
    ``_Session``/``_NoNetrcAuth``), and never persists a cookie across calls:
    its cookie jar rejects every domain, and each call sends exactly the
    cookies :func:`perform` is given.
    """
    session = _Session()
    session.auth = _NoNetrcAuth()
    session.cookies.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
    return session


@dataclass(frozen=True, slots=True)
class Decode:
    """Accumulate the response body and decode it once."""


@dataclass(frozen=True, slots=True)
class Ignore:
    """Stream the response body and drop it."""


@dataclass(frozen=True, slots=True)
class Save:
    """Stream the response body to ``path``, overwriting it atomically."""

    path: Path


ReceiveSpec = Decode | Ignore | Save


class SaveFailure(Exception):
    """A ``Save`` destination that could not be written; wraps the ``OSError`` cause."""

    def __init__(self, path: Path, error: OSError) -> None:
        super().__init__(str(error))
        self.path = path
        self.error = error


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestSpec:
    """One request to perform, already unwrapped from AgL values.

    ``auth`` is either an already-rendered ``Authorization`` header value or
    a ``(user, password)`` basic-auth pair, which ``requests`` renders
    itself. ``params`` is appended to any query the url already carries.
    ``method`` is sent verbatim (never case-normalised), so an ``Other``
    method travels exactly as given. ``timeout_seconds`` has no default: the
    standard library's HTTP module owns the default inactivity timeout. It is
    positive, or ``None`` to disable the inactivity bound; the caller rejects
    a non-positive value before building a spec.
    """

    method: str
    url: str
    params: Sequence[tuple[str, str]] | Mapping[str, str] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes | None = None
    content_type: str | None = None
    auth: str | tuple[str, str] | None = None
    cookies: Mapping[str, str] = field(default_factory=dict)
    follow_redirects: bool = True
    verify_tls: bool = True
    timeout_seconds: float | None
    receive: ReceiveSpec = field(default_factory=Decode)


@dataclass(frozen=True, slots=True)
class StreamedResponse:
    """The result of one exchange, before the companion builds an AgL response."""

    status: int
    headers: dict[str, str]
    text: str
    body_bytes: int
    url: str
    cookies: dict[str, str]
    elapsed: float
    encoding: str


class TransportError(Exception):
    """A classified transport failure, raised by :func:`perform`."""

    def __init__(self, kind: TransportErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


class RedirectError(TransportError):
    """A ``redirect``-kind :class:`TransportError`, with the redirects actually followed."""

    def __init__(self, message: str, *, redirects: int) -> None:
        super().__init__("redirect", message)
        self.redirects = redirects


class DecodeFailure(Exception):
    """A response body that could not be decoded under the strict charset rule."""

    def __init__(self, *, status: int, headers: Mapping[str, str], encoding: str, url: str) -> None:
        super().__init__(f"cannot decode response body as {encoding!r}")
        self.status = status
        self.headers = dict(headers)
        self.encoding = encoding
        self.url = url


def resolve_charset(content_type: str | None) -> str:
    """The charset declared in a ``Content-Type`` header, else UTF-8.

    Never guesses statistically: an unlabelled body is always read as UTF-8.
    """
    if content_type is not None:
        match = _CHARSET_PATTERN.search(content_type)
        if match is not None:
            return match.group(0).strip().strip("\"'")
    return "utf-8"


def _response_headers(response: requests.Response) -> dict[str, str]:
    """Lowercase *response*'s header names; ``requests`` already joins repeats with ``", "``."""
    return {name.lower(): value for name, value in response.headers.items()}


def _validate_method(method: str) -> None:
    """Raise a ``request``-kind failure for a method that is not a single RFC 9110 token."""
    if not _METHOD_TOKEN.fullmatch(method):
        raise TransportError("request", f"invalid HTTP method token: {method!r}")


def _validate_headers(headers: Mapping[str, str]) -> None:
    """Raise a ``request``-kind failure naming only the header NAME for an invalid entry.

    Checked before any header reaches ``requests``, whose own validator
    raises an exception that quotes the value -- unsafe when the value is a
    credential such as a bearer token or session cookie.
    """
    for name, value in headers.items():
        if not _HEADER_NAME.fullmatch(name) or not _HEADER_VALUE.fullmatch(value):
            raise TransportError("request", f"invalid HTTP header: {name!r}")


def _cookie_jar_for(spec: RequestSpec) -> requests.cookies.RequestsCookieJar:
    """Cookies scoped to *spec*'s request host.

    ``requests.cookies.create_cookie`` defaults to an unscoped "supercookie"
    sent to every host; binding the domain here means a cross-host redirect
    drops these cookies while a same-host redirect keeps them.
    """
    host = urlsplit(spec.url).hostname or ""
    jar = requests.cookies.RequestsCookieJar()
    for name, value in spec.cookies.items():
        jar.set_cookie(requests.cookies.create_cookie(name, value, domain=host))
    return jar


def perform(session: requests.Session, spec: RequestSpec) -> StreamedResponse:
    """Send *spec* through *session* and stream its body per ``spec.receive``.

    Uses ``timeout=(t, t)``: both the connect and every subsequent read are
    bounded by the same inactivity budget, so a stalled transfer of any size
    fails as a timeout while a healthy transfer of any size completes.
    Redirects (uniform for every method, ``HEAD`` included) and TLS
    verification are ``requests``' own. The request is built and sent
    manually (``prepare_request``/``send``, reproducing what
    ``Session.request`` does) rather than through ``Session.request``, whose
    ``Request`` construction upper-cases the method; this seam sends it
    verbatim instead.
    """
    _validate_method(spec.method)
    headers = _headers_for(spec)
    _validate_headers(headers)
    timeout = None if spec.timeout_seconds is None else (spec.timeout_seconds, spec.timeout_seconds)
    started = time.monotonic()
    try:
        request = requests.Request(
            method=spec.method,
            url=spec.url,
            params=spec.params,
            headers=headers,
            data=spec.body,
            auth=spec.auth if isinstance(spec.auth, tuple) else None,
            cookies=_cookie_jar_for(spec),
        )
        prepared = session.prepare_request(request)
        prepared.method = spec.method
        settings = session.merge_environment_settings(prepared.url, {}, True, spec.verify_tls, None)
        with _insecure_warning_suppressed(spec.verify_tls):
            response = session.send(
                prepared,
                timeout=timeout,
                allow_redirects=spec.follow_redirects,
                **settings,
            )
            try:
                text, body_bytes, encoding = _consume(response, spec.receive)
            finally:
                response.close()
    except requests.exceptions.RequestException as exc:
        raise _classify(exc, session) from exc
    elapsed = time.monotonic() - started
    cookies = {
        name: value for name, value in response.cookies.get_dict().items() if value is not None
    }
    return StreamedResponse(
        status=response.status_code,
        headers=_response_headers(response),
        text=text,
        body_bytes=body_bytes,
        url=response.url,
        cookies=cookies,
        elapsed=elapsed,
        encoding=encoding,
    )


def _headers_for(spec: RequestSpec) -> dict[str, str]:
    """Merge the content type and a string-rendered auth into *spec*'s headers."""
    headers = dict(spec.headers)
    if spec.content_type is not None:
        headers["Content-Type"] = spec.content_type
    if isinstance(spec.auth, str):
        headers["Authorization"] = spec.auth
    return headers


def _consume(response: requests.Response, receive: ReceiveSpec) -> tuple[str, int, str]:
    """Stream *response*'s body per *receive*; returns (text, byte count, encoding)."""
    if isinstance(receive, Ignore):
        body_bytes = sum(len(chunk) for chunk in response.iter_content(chunk_size=_CHUNK_SIZE))
        return "", body_bytes, ""
    if isinstance(receive, Save):
        sizes: list[int] = []

        def chunks() -> Iterator[bytes]:
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                sizes.append(len(chunk))
                yield chunk

        try:
            fs.write_bytes_atomic(receive.path, chunks())
        except requests.exceptions.RequestException:
            # ``requests.exceptions.RequestException`` is itself an ``OSError``
            # subclass; a network failure raised while iterating *chunks* must
            # still reach :func:`perform`'s own classifier, not this catch.
            raise
        except OSError as exc:
            raise SaveFailure(receive.path, exc) from exc
        return "", sum(sizes), ""
    data = response.content
    encoding = resolve_charset(response.headers.get("Content-Type"))
    try:
        text = data.decode(encoding)
    except (LookupError, UnicodeDecodeError) as exc:
        raise DecodeFailure(
            status=response.status_code,
            headers=_response_headers(response),
            encoding=encoding,
            url=response.url,
        ) from exc
    return text, len(data), encoding


_CLASSIFIERS: tuple[tuple[type[requests.exceptions.RequestException], TransportErrorKind], ...] = (
    (requests.exceptions.MissingSchema, "url"),
    (requests.exceptions.InvalidSchema, "url"),
    (requests.exceptions.InvalidURL, "url"),
    (requests.exceptions.SSLError, "tls"),
    (requests.exceptions.Timeout, "timeout"),
)


def _classify(
    exc: requests.exceptions.RequestException, session: requests.Session
) -> TransportError:
    """Map a ``requests``/``urllib3`` failure onto one of the six transport kinds.

    ``requests`` wraps a read timeout raised mid-stream (while a body is
    being iterated) as a plain ``ConnectionError``, losing its timeout
    identity; recover it from the wrapped ``urllib3`` exception before
    falling back to the ordered classification below. ``InvalidHeader`` is
    classified without its own message, which quotes the offending value;
    ``_validate_headers`` catches the ordinary case earlier and names only
    the header, so this is just a safety net. ``_CLASSIFIERS`` lists only the
    exception types that need a kind other than the ``connection`` fallback:
    ``ConnectTimeout``/``ReadTimeout`` are ``Timeout`` subclasses and need no
    row of their own, and a bare ``ConnectionError`` or
    ``ChunkedEncodingError`` already falls through to ``connection``. Any
    ``RequestException`` matching none of them (an unlisted ``requests``
    failure) still classifies as ``connection``, the catch-all kind.
    """
    if isinstance(exc, requests.exceptions.TooManyRedirects):
        redirects = len(exc.response.history) if exc.response is not None else session.max_redirects
        return RedirectError(str(exc), redirects=redirects)
    if isinstance(exc, requests.exceptions.InvalidHeader):
        return TransportError("request", "invalid HTTP header")
    if isinstance(exc, requests.exceptions.ConnectionError) and isinstance(
        exc.__context__, urllib3.exceptions.ReadTimeoutError
    ):
        return TransportError("timeout", str(exc))
    for exc_type, kind in _CLASSIFIERS:
        if isinstance(exc, exc_type):
            return TransportError(kind, str(exc))
    return TransportError("connection", str(exc))


@contextmanager
def _insecure_warning_suppressed(verify_tls: bool) -> Iterator[None]:
    """Suppress urllib3's insecure-request warning for one call, when TLS is off."""
    if verify_tls:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
        yield
