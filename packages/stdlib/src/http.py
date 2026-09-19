"""HTTP requests for ``std/http``, over the pooled session in ``agm.core.http``."""

from __future__ import annotations

import base64
import time
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlencode

from agl import AglException, nominals, runtime
from agl import dict as agl_dict

from agm.agl.runtime.serialize import dumps_exact
from agm.core import http as core_http
from agm.core.fs import fs_error_message
from agm.core.parse import parse_timeout

Option = nominals.std.option.Option
Headers = nominals.std.http.Headers
Auth = nominals.std.http.Auth
Body = nominals.std.http.Body
Receive = nominals.std.http.Receive
Response = nominals.std.http.Response
HttpUrlError = nominals.std.http.HttpUrlError
HttpRequestError = nominals.std.http.HttpRequestError
HttpConnectionError = nominals.std.http.HttpConnectionError
HttpTimeoutError = nominals.std.http.HttpTimeoutError
HttpTlsError = nominals.std.http.HttpTlsError
HttpRedirectError = nominals.std.http.HttpRedirectError
HttpDecodeError = nominals.std.http.HttpDecodeError
TypeError = nominals.std.errors.TypeError
FsError = nominals.std.fs.FsError

# The pooled session's ``runtime.state`` key: one session per interpreter run.
_SESSION_KEY = "std/http/session"

_REDACTED = "<redacted>"
# Credential-bearing headers; the response side adds `set-cookie` (same secret
# a request `cookie` header carries). `Auth` needs no entry of its own here:
# it renders into `authorization`, already covered.
_REDACT_REQUEST_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie"})
_REDACT_RESPONSE_HEADERS = frozenset({"set-cookie"})

_RECEIVE_NAMES = {
    core_http.Decode: "decode",
    core_http.Ignore: "ignore",
    core_http.Save: "save",
}

_TRANSPORT_ERROR_NAMES = {
    "url": "HttpUrlError",
    "request": "HttpRequestError",
    "connection": "HttpConnectionError",
    "timeout": "HttpTimeoutError",
    "tls": "HttpTlsError",
    "redirect": "HttpRedirectError",
}


def _unwrap_headers(headers: object) -> dict[str, str]:
    """A plain, name-lowercased dict from a ``Headers`` argument.

    A raw ``Headers(...)`` construction keeps names verbatim (only the
    ``headers(...)`` constructor normalises them), so this is also the one
    place the auth-override check can rely on a folded name.
    """
    return {name.lower(): value for name, value in headers.entries.items()}


def _auth_header(auth: object) -> str | None:
    """Render *auth* as an ``Authorization`` header value, or ``None`` for ``NoAuth``."""
    if isinstance(auth, Auth.Basic):
        token = base64.b64encode(f"{auth.user}:{auth.password}".encode()).decode("ascii")
        return f"Basic {token}"
    if isinstance(auth, Auth.Bearer):
        return f"Bearer {auth.token}"
    return None


def _unwrap_body(body: object) -> tuple[bytes | None, str | None, str | None]:
    """Return (payload bytes, content type, verbatim text) for *body*.

    ``(None, None, None)`` for ``Empty``. A ``Json`` payload is rendered
    through the DSL's own exact-decimal JSON writer, never Python's
    ``json.dumps``, so an amount stays exact on the wire.
    """
    if isinstance(body, Body.Text):
        content = body.content
        return content.encode("utf-8"), getattr(body, "content-type"), content
    if isinstance(body, Body.Json):
        text = dumps_exact(body.value.value, indent=None)
        return text.encode("utf-8"), "application/json", text
    if isinstance(body, Body.Form):
        text = urlencode(dict(body.fields))
        return text.encode("utf-8"), "application/x-www-form-urlencoded", text
    return None, None, None


def _unwrap_receive(receive: object) -> core_http.ReceiveSpec:
    if isinstance(receive, Receive.Ignore):
        return core_http.Ignore()
    if isinstance(receive, Receive.Save):
        return core_http.Save(path=Path(receive.path))
    return core_http.Decode()


def _unwrap_timeout(timeout: object) -> tuple[float | None, str | None]:
    """Return (inactivity seconds, the given text) from a ``timeout: Option[text]`` argument.

    An unparseable duration raises ``TypeError``, exactly as ``exec`` does; so
    does a non-positive one, before any transport call is attempted.
    """
    if isinstance(timeout, Option.Some):
        text = timeout.value
        try:
            seconds = parse_timeout(text)
        except ValueError as exc:
            raise AglException(TypeError(message=f"invalid timeout: {exc}")) from exc
        if seconds <= 0:
            raise AglException(TypeError(message=f"invalid timeout: {text!r} is not positive"))
        return seconds, text
    return None, None


def _redact(values: dict[str, str], names: frozenset[str]) -> dict[str, str]:
    """*values* with every entry named in *names* replaced by the redaction marker."""
    return {name: (_REDACTED if name in names else value) for name, value in values.items()}


def _redact_all(values: dict[str, str]) -> dict[str, str]:
    """*values* with every value replaced by the redaction marker; cookies carry credentials too."""
    return dict.fromkeys(values, _REDACTED)


def _transport_exception(
    exc: core_http.TransportError, *, url: str, method: object, timeout_text: str | None
) -> AglException:
    """Map one classified transport failure onto its ``HttpError`` subtype."""
    base = {"url": url, "method": method, "message": exc.message}
    if isinstance(exc, core_http.RedirectError):
        return AglException(HttpRedirectError(redirects=exc.redirects, **base))
    if exc.kind == "url":
        return AglException(HttpUrlError(**base))
    if exc.kind == "request":
        return AglException(HttpRequestError(cause=exc.message, **base))
    if exc.kind == "connection":
        return AglException(HttpConnectionError(cause=exc.message, **base))
    if exc.kind == "timeout":
        # Empty when the inactivity timeout was disabled (``timeout = None``).
        timeout_field = timeout_text if timeout_text is not None else ""
        return AglException(HttpTimeoutError(timeout=timeout_field, **base))
    return AglException(HttpTlsError(cause=exc.message, **base))


def _trace_failure(error_type: str, message: str, elapsed: float) -> None:
    runtime.trace(
        "http_failure", {"error_type": error_type, "message": message, "elapsed": elapsed}
    )


def request(
    method_text: str,
    method: object,
    url: str,
    query: object,
    headers: object,
    body: object,
    auth: object,
    cookies: object,
    receive: object,
    timeout: object,
    follow_redirects: bool,
    verify_tls: bool,
) -> object:
    """Perform one HTTP request through the pooled session; see ``http.agl`` for the contract."""
    body_bytes, content_type, body_text = _unwrap_body(body)
    headers_dict = _unwrap_headers(headers)
    auth_header = _auth_header(auth)
    query_dict = dict(query)
    cookies_dict = dict(cookies)
    receive_spec = _unwrap_receive(receive)
    timeout_seconds, timeout_text = _unwrap_timeout(timeout)

    session = runtime.state(_SESSION_KEY, core_http.open_session, close=lambda s: s.close())

    traced_headers = dict(headers_dict)
    if content_type is not None:
        traced_headers["content-type"] = content_type
    if auth_header is not None:
        traced_headers["authorization"] = auth_header
    runtime.trace(
        "http_request",
        {
            "method": method_text,
            "url": url,
            "query": query_dict,
            "headers": _redact(traced_headers, _REDACT_REQUEST_HEADERS),
            "body": body_text,
            "timeout": timeout_text,
            "follow_redirects": follow_redirects,
            "verify_tls": verify_tls,
            "cookies": _redact_all(cookies_dict),
        },
    )

    spec = core_http.RequestSpec(
        method=method_text,
        url=url,
        params=query_dict,
        headers=headers_dict,
        body=body_bytes,
        content_type=content_type,
        auth=auth_header,
        cookies=cookies_dict,
        follow_redirects=follow_redirects,
        verify_tls=verify_tls,
        timeout_seconds=timeout_seconds,
        receive=receive_spec,
    )

    started = time.monotonic()
    try:
        streamed = core_http.perform(session, spec)
    except core_http.TransportError as exc:
        _trace_failure(_TRANSPORT_ERROR_NAMES[exc.kind], exc.message, time.monotonic() - started)
        raise _transport_exception(exc, url=url, method=method, timeout_text=timeout_text) from exc
    except core_http.DecodeFailure as exc:
        _trace_failure("HttpDecodeError", str(exc), time.monotonic() - started)
        raise AglException(
            HttpDecodeError(
                url=exc.url,
                method=method,
                message=str(exc),
                status=exc.status,
                headers=Headers(entries=agl_dict(exc.headers)),
                encoding=exc.encoding,
            )
        ) from exc
    except core_http.SaveFailure as exc:
        destination = str(exc.path)
        _trace_failure("FsError", str(exc.error), time.monotonic() - started)
        raise AglException(
            FsError(
                message=fs_error_message("write", destination), path=destination, operation="write"
            )
        ) from exc

    save_path = receive_spec.path if isinstance(receive_spec, core_http.Save) else None
    runtime.trace(
        "http_response",
        {
            "status": streamed.status,
            "url": streamed.url,
            "headers": _redact(streamed.headers, _REDACT_RESPONSE_HEADERS),
            "body": streamed.text if isinstance(receive_spec, core_http.Decode) else None,
            "body_bytes": streamed.body_bytes,
            "elapsed": streamed.elapsed,
            "receive": _RECEIVE_NAMES[type(receive_spec)],
            "save_path": str(save_path) if save_path is not None else None,
            "cookies": _redact_all(streamed.cookies),
        },
    )

    return Response(
        status=streamed.status,
        headers=Headers(entries=agl_dict(streamed.headers)),
        body=streamed.text,
        url=streamed.url,
        method=method,
        cookies=agl_dict(streamed.cookies),
        elapsed=Decimal(str(streamed.elapsed)),
    )


__all__ = ["request"]
