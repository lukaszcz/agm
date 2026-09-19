"""Static contracts for the ``std/http`` standard-library module.

These tests exercise the pure-AgL surface cheaply through the type-checker
rather than the full e2e harness, which covers behaviour.
"""

from __future__ import annotations

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.scope.symbols import AglScopeError
from tests.agl.module_graph import resolve_and_check_inline_entry


def test_data_layer_types_are_visible_through_the_import() -> None:
    resolve_and_check_inline_entry(
        "import std/http\n"
        "let _: http::Headers = http::headers({})\n"
        "let _: http::Method = http::Method::Get\n",
        HostCapabilities(),
    )


def test_method_render_and_other_are_visible_through_the_import() -> None:
    """A ``Method`` method requires a receiver statically typed as ``Method``,
    so a bare inline member widens through an annotated binding first."""
    resolve_and_check_inline_entry(
        "import std/http\n"
        "let get: http::Method = http::Method::Get\n"
        'let other: http::Method = http::Method::Other("PROPFIND")\n'
        "let _: text = get.render()\n"
        "let _: text = other.render()\n",
        HostCapabilities(),
    )


def test_every_http_error_subtype_type_checks_with_its_declared_fields() -> None:
    resolve_and_check_inline_entry(
        "import std/http\n"
        "let verb: http::Method = http::Method::Get\n"
        'let _ = http::HttpUrlError(url = "https://x", method = verb, message = "m")\n'
        'let _ = http::HttpRequestError(url = "https://x", method = verb,\n'
        '  cause = "bad token", message = "m")\n'
        'let _ = http::HttpConnectionError(url = "https://x", method = verb,\n'
        '  cause = "refused", message = "m")\n'
        'let _ = http::HttpTimeoutError(url = "https://x", method = verb,\n'
        '  timeout = "30s", message = "m")\n'
        'let _ = http::HttpTlsError(url = "https://x", method = verb,\n'
        '  cause = "bad cert", message = "m")\n'
        'let _ = http::HttpRedirectError(url = "https://x", method = verb,\n'
        '  redirects = 5, message = "m")\n'
        'let _ = http::HttpDecodeError(url = "https://x", method = verb,\n'
        '  status = 200, headers = http::headers({}), encoding = "latin-1", message = "m")\n'
        'let response = http::Response(200, http::headers({}), "", "https://x", verb, {}, 0.0)\n'
        'let _ = http::HttpStatusError(url = "https://x", method = verb,\n'
        '  response = response, message = "m")\n',
        HostCapabilities(),
    )


def test_timeout_is_a_param_and_settable_across_the_import_boundary() -> None:
    resolve_and_check_inline_entry(
        "import std/http\nhttp::timeout := None\nlet _: Option[text] = http::timeout\n",
        HostCapabilities(),
    )


def test_lower_name_is_hidden_behind_headers_internals() -> None:
    """``lower-name`` lives in the ``HeadersInternals`` scope, not the public
    surface, so an importer cannot reach it as ``http::lower-name``."""
    with pytest.raises(AglScopeError):
        resolve_and_check_inline_entry(
            'import std/http\nlet _ = http::lower-name("X")\n', HostCapabilities()
        )
