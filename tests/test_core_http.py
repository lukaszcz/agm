"""Behavior tests for the HTTP transport seam (agm.core.http), never touching a real network."""

from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
import requests.exceptions
import urllib3.exceptions

from agm.core import http
from tests._http_helpers import FakeHttp, fake_session


def test_open_session_honours_environment_proxy_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    session, adapter = fake_session([{"status": 200, "body": ""}])

    http.perform(
        session, http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert adapter.proxies_sent[0] == {"https": "http://proxy.example:8080"}
    adapter.assert_complete()


def test_perform_sends_method_url_and_headers() -> None:
    session, adapter = fake_session(
        [
            {
                "expect": {
                    "method": "GET",
                    "url": "https://example.org/things",
                    "headers": {"X-Trace": "abc"},
                },
                "status": 200,
                "body": "ok",
            }
        ]
    )

    result = http.perform(
        session,
        http.RequestSpec(
            method="GET",
            url="https://example.org/things",
            headers={"X-Trace": "abc"},
            timeout_seconds=5.0,
        ),
    )

    assert result.status == 200
    assert result.text == "ok"
    adapter.assert_complete()


def test_perform_accepts_a_plain_requests_session() -> None:
    session = requests.Session()
    adapter = FakeHttp([{"status": 200, "body": ""}])
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    result = http.perform(
        session, http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert result.status == 200
    adapter.assert_complete()


def test_perform_appends_params_to_an_existing_query() -> None:
    session, adapter = fake_session([{"status": 200, "body": ""}])

    http.perform(
        session,
        http.RequestSpec(
            method="GET",
            url="https://example.org/x?a=1",
            params={"b": "2"},
            timeout_seconds=5.0,
        ),
    )

    query = parse_qs(urlparse(adapter.sent[0].url).query)
    assert query == {"a": ["1"], "b": ["2"]}
    adapter.assert_complete()


def test_perform_sends_body_bytes_and_content_type() -> None:
    session, adapter = fake_session([{"expect": {"body": '{"a": 1}'}, "status": 200, "body": ""}])

    http.perform(
        session,
        http.RequestSpec(
            method="POST",
            url="https://example.org/x",
            body=b'{"a": 1}',
            content_type="application/json",
            timeout_seconds=5.0,
        ),
    )

    assert adapter.sent[0].headers["Content-Type"] == "application/json"
    adapter.assert_complete()


def test_perform_sends_no_body_and_no_content_type_by_default() -> None:
    session, adapter = fake_session([{"status": 200, "body": ""}])

    http.perform(
        session, http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert adapter.sent[0].body is None
    assert "Content-Type" not in adapter.sent[0].headers
    adapter.assert_complete()


def test_perform_renders_a_string_auth_as_the_authorization_header() -> None:
    session, adapter = fake_session([{"status": 200, "body": ""}])

    http.perform(
        session,
        http.RequestSpec(
            method="GET", url="https://example.org/x", auth="Bearer tok", timeout_seconds=5.0
        ),
    )

    assert adapter.sent[0].headers["Authorization"] == "Bearer tok"
    adapter.assert_complete()


def test_perform_renders_a_basic_auth_tuple() -> None:
    session, adapter = fake_session([{"status": 200, "body": ""}])

    http.perform(
        session,
        http.RequestSpec(
            method="GET", url="https://example.org/x", auth=("user", "pw"), timeout_seconds=5.0
        ),
    )

    assert adapter.sent[0].headers["Authorization"].startswith("Basic ")
    adapter.assert_complete()


def test_perform_sends_exactly_the_given_cookies() -> None:
    session, adapter = fake_session([{"status": 200, "body": ""}])

    http.perform(
        session,
        http.RequestSpec(
            method="GET", url="https://example.org/x", cookies={"a": "1"}, timeout_seconds=5.0
        ),
    )

    assert adapter.sent[0].headers.get("Cookie") == "a=1"
    adapter.assert_complete()


def test_perform_drops_caller_cookies_on_a_cross_host_redirect() -> None:
    session, adapter = fake_session(
        [
            {"status": 302, "headers": {"Location": "https://other.example/final"}},
            {"status": 200, "body": ""},
        ]
    )

    http.perform(
        session,
        http.RequestSpec(
            method="GET",
            url="https://example.org/start",
            cookies={"a": "1"},
            timeout_seconds=5.0,
        ),
    )

    assert adapter.sent[0].headers.get("Cookie") == "a=1"
    assert "Cookie" not in adapter.sent[1].headers
    adapter.assert_complete()


def test_perform_drops_caller_cookies_on_a_subdomain_redirect() -> None:
    session, adapter = fake_session(
        [
            {"status": 302, "headers": {"Location": "https://sub.example.org/final"}},
            {"status": 200, "body": ""},
        ]
    )

    http.perform(
        session,
        http.RequestSpec(
            method="GET",
            url="https://example.org/start",
            cookies={"secret": "value"},
            timeout_seconds=5.0,
        ),
    )

    assert adapter.sent[0].headers.get("Cookie") == "secret=value"
    assert "Cookie" not in adapter.sent[1].headers
    adapter.assert_complete()


def test_perform_never_restores_caller_cookies_after_a_cross_host_redirect() -> None:
    session, adapter = fake_session(
        [
            {"status": 302, "headers": {"Location": "https://other.example/one"}},
            {"status": 302, "headers": {"Location": "https://other.example/two"}},
            {"status": 200, "body": ""},
        ]
    )

    http.perform(
        session,
        http.RequestSpec(
            method="GET",
            url="https://example.org/start",
            cookies={"secret": "value"},
            timeout_seconds=5.0,
        ),
    )

    assert adapter.sent[0].headers.get("Cookie") == "secret=value"
    assert "Cookie" not in adapter.sent[1].headers
    assert "Cookie" not in adapter.sent[2].headers
    adapter.assert_complete()


@pytest.mark.parametrize("url", ["http://localhost/start", "http://[::1]/start"])
def test_perform_sends_cookies_to_local_and_ipv6_hosts(url: str) -> None:
    session, adapter = fake_session([{"status": 200, "body": ""}])

    http.perform(
        session,
        http.RequestSpec(method="GET", url=url, cookies={"a": "1"}, timeout_seconds=5.0),
    )

    assert adapter.sent[0].headers.get("Cookie") == "a=1"
    adapter.assert_complete()


def test_perform_keeps_caller_cookies_on_a_same_host_redirect() -> None:
    session, adapter = fake_session(
        [
            {"status": 302, "headers": {"Location": "https://example.org/final"}},
            {"status": 200, "body": ""},
        ]
    )

    http.perform(
        session,
        http.RequestSpec(
            method="GET",
            url="https://example.org/start",
            cookies={"a": "1"},
            timeout_seconds=5.0,
        ),
    )

    assert adapter.sent[0].headers.get("Cookie") == "a=1"
    assert adapter.sent[1].headers.get("Cookie") == "a=1"
    adapter.assert_complete()


@pytest.mark.parametrize(
    ("cookies", "expected_cookie"),
    [({}, "header=value"), ({"caller": "value"}, "header=value; caller=value")],
)
def test_perform_merges_lowercase_cookie_headers_and_keeps_them_on_same_host_redirect(
    cookies: dict[str, str], expected_cookie: str
) -> None:
    session, adapter = fake_session(
        [
            {"status": 302, "headers": {"Location": "https://example.org/final"}},
            {"status": 200, "body": ""},
        ]
    )

    http.perform(
        session,
        http.RequestSpec(
            method="GET",
            url="https://example.org/start",
            headers={"cookie": "header=value"},
            cookies=cookies,
            timeout_seconds=5.0,
        ),
    )

    assert adapter.sent[0].headers.get("Cookie") == expected_cookie
    assert adapter.sent[1].headers.get("Cookie") == expected_cookie
    adapter.assert_complete()


def test_perform_never_retains_a_cookie_across_calls() -> None:
    session, adapter = fake_session(
        [
            {"status": 200, "body": "", "headers": {"Set-Cookie": "session=abc"}},
            {"status": 200, "body": ""},
        ]
    )

    http.perform(
        session, http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0)
    )
    http.perform(
        session, http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert "Cookie" not in adapter.sent[1].headers
    adapter.assert_complete()


def test_perform_forwards_a_cookie_set_by_an_intermediate_redirect_hop() -> None:
    session, adapter = fake_session(
        [
            {
                "status": 302,
                "headers": {
                    "Location": "https://example.org/final",
                    "Set-Cookie": "session=abc",
                },
            },
            {"status": 200, "body": ""},
        ]
    )

    http.perform(
        session,
        http.RequestSpec(method="GET", url="https://example.org/start", timeout_seconds=5.0),
    )

    assert adapter.sent[1].headers.get("Cookie") == "session=abc"
    adapter.assert_complete()


def test_perform_keeps_caller_and_intermediate_cookies_on_a_same_host_redirect() -> None:
    session, adapter = fake_session(
        [
            {
                "status": 302,
                "headers": {
                    "Location": "https://example.org/final",
                    "Set-Cookie": "session=abc",
                },
            },
            {"status": 200, "body": ""},
        ]
    )

    http.perform(
        session,
        http.RequestSpec(
            method="GET",
            url="https://example.org/start",
            cookies={"caller": "value"},
            timeout_seconds=5.0,
        ),
    )

    assert adapter.sent[1].headers.get("Cookie") == "session=abc; caller=value"
    adapter.assert_complete()


def test_redirect_without_a_source_request_is_ignored() -> None:
    session = http.open_session()
    assert isinstance(session, http._Session)
    prepared = requests.Request(method="GET", url="https://example.org/final").prepare()
    response = requests.Response()

    session.rebuild_auth(prepared, response)

    assert "Cookie" not in prepared.headers


def test_perform_strips_the_authorization_header_on_a_cross_host_redirect() -> None:
    session, adapter = fake_session(
        [
            {"status": 302, "headers": {"Location": "https://other.example/final"}},
            {"status": 200, "body": ""},
        ]
    )

    http.perform(
        session,
        http.RequestSpec(
            method="GET",
            url="https://example.org/start",
            auth="Bearer tok",
            timeout_seconds=5.0,
        ),
    )

    assert adapter.sent[0].headers.get("Authorization") == "Bearer tok"
    assert "Authorization" not in adapter.sent[1].headers
    adapter.assert_complete()


def test_perform_never_sends_netrc_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    netrc_path = tmp_path / "netrc"
    netrc_path.write_text("machine example.org login user password secret\n", encoding="utf-8")
    monkeypatch.setenv("NETRC", str(netrc_path))
    session, adapter = fake_session([{"status": 200, "body": ""}])

    http.perform(
        session, http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert "Authorization" not in adapter.sent[0].headers
    adapter.assert_complete()


def test_perform_never_sends_netrc_credentials_on_a_redirect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    netrc_path = tmp_path / "netrc"
    netrc_path.write_text("machine example.org login user password secret\n", encoding="utf-8")
    monkeypatch.setenv("NETRC", str(netrc_path))
    session, adapter = fake_session(
        [
            {"status": 302, "headers": {"Location": "https://example.org/final"}},
            {"status": 200, "body": ""},
        ]
    )

    http.perform(
        session,
        http.RequestSpec(method="GET", url="https://example.org/start", timeout_seconds=5.0),
    )

    assert "Authorization" not in adapter.sent[0].headers
    assert "Authorization" not in adapter.sent[1].headers
    adapter.assert_complete()


def test_perform_reports_cookies_set_by_the_final_response() -> None:
    session, adapter = fake_session(
        [{"status": 200, "body": "", "headers": {"Set-Cookie": "session=abc"}}]
    )

    result = http.perform(
        session, http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert result.cookies == {"session": "abc"}
    adapter.assert_complete()


def test_perform_decodes_utf8_by_default() -> None:
    session, adapter = fake_session([{"status": 200, "body": "café"}])

    result = http.perform(
        session, http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert result.text == "café"
    assert result.encoding == "utf-8"
    adapter.assert_complete()


def test_perform_decodes_the_declared_charset() -> None:
    session, adapter = fake_session(
        [
            {
                "status": 200,
                "headers": {"Content-Type": "text/plain; charset=iso-8859-1"},
                "body": "café",
                "charset": "iso-8859-1",
            }
        ]
    )

    result = http.perform(
        session, http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert result.text == "café"
    assert result.encoding == "iso-8859-1"
    adapter.assert_complete()


@pytest.mark.parametrize(
    "content_type",
    (
        'text/plain; note="x;charset=ascii"; charset=utf-8',
        'text/plain; note="x\\";charset=ascii"; charset=utf-8',
    ),
)
def test_perform_ignores_charset_text_inside_a_quoted_parameter(content_type: str) -> None:
    session, adapter = fake_session(
        [
            {
                "status": 200,
                "headers": {"Content-Type": content_type},
                "body": "café",
            }
        ]
    )

    result = http.perform(
        session, http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert result.text == "café"
    assert result.encoding == "utf-8"
    adapter.assert_complete()


def test_perform_ignores_a_parameter_whose_name_only_ends_in_charset() -> None:
    session, adapter = fake_session(
        [
            {
                "status": 200,
                "headers": {"Content-Type": "text/plain; xcharset=iso-8859-1"},
                "body": "café",
            }
        ]
    )

    result = http.perform(
        session, http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert result.text == "café"
    assert result.encoding == "utf-8"
    adapter.assert_complete()


def test_perform_raises_decode_failure_for_an_undecodable_body() -> None:
    session, adapter = fake_session(
        [
            {
                "status": 200,
                "headers": {"Content-Type": "text/plain; charset=ascii"},
                "body_hex": "fffe",
            }
        ]
    )

    with pytest.raises(http.DecodeFailure) as excinfo:
        http.perform(
            session,
            http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0),
        )

    assert excinfo.value.status == 200
    assert excinfo.value.encoding == "ascii"
    assert excinfo.value.headers["content-type"] == "text/plain; charset=ascii"
    assert excinfo.value.url == "https://example.org/x"
    adapter.assert_complete()


def test_perform_ignore_streams_and_drops_the_body() -> None:
    session, adapter = fake_session([{"status": 200, "body_hex": b"some content".hex()}])

    result = http.perform(
        session,
        http.RequestSpec(
            method="GET", url="https://example.org/x", receive=http.Ignore(), timeout_seconds=5.0
        ),
    )

    assert result.text == ""
    assert result.body_bytes == len(b"some content")
    adapter.assert_complete()


def test_perform_ignore_propagates_a_mid_stream_failure() -> None:
    session, adapter = fake_session(
        [{"status": 200, "body_hex": b"partial".hex(), "fail_mid_stream": "connection"}]
    )

    with pytest.raises(http.TransportError) as excinfo:
        http.perform(
            session,
            http.RequestSpec(
                method="GET",
                url="https://example.org/x",
                receive=http.Ignore(),
                timeout_seconds=5.0,
            ),
        )

    assert excinfo.value.kind == "connection"
    adapter.assert_complete()


def test_perform_save_streams_the_body_to_disk_and_overwrites(tmp_path: Path) -> None:
    destination = tmp_path / "out.bin"
    destination.write_bytes(b"stale")
    session, adapter = fake_session([{"status": 200, "body_hex": b"hello".hex()}])

    result = http.perform(
        session,
        http.RequestSpec(
            method="GET",
            url="https://example.org/x",
            receive=http.Save(destination),
            timeout_seconds=5.0,
        ),
    )

    assert destination.read_bytes() == b"hello"
    assert result.text == ""
    assert result.body_bytes == 5
    assert [child.name for child in tmp_path.iterdir()] == ["out.bin"]
    adapter.assert_complete()


def test_perform_save_raises_a_save_failure_wrapping_the_os_error(tmp_path: Path) -> None:
    destination = tmp_path / "missing-dir" / "out.bin"
    session, adapter = fake_session([{"status": 200, "body_hex": b"hello".hex()}])

    with pytest.raises(http.SaveFailure) as excinfo:
        http.perform(
            session,
            http.RequestSpec(
                method="GET",
                url="https://example.org/x",
                receive=http.Save(destination),
                timeout_seconds=5.0,
            ),
        )

    assert excinfo.value.path == destination
    assert isinstance(excinfo.value.error, OSError)
    adapter.assert_complete()


def test_perform_save_leaves_the_previous_file_on_a_mid_stream_failure(tmp_path: Path) -> None:
    destination = tmp_path / "out.bin"
    destination.write_bytes(b"previous")
    session, adapter = fake_session(
        [{"status": 200, "body_hex": b"partial".hex(), "fail_mid_stream": "connection"}]
    )

    with pytest.raises(http.TransportError) as excinfo:
        http.perform(
            session,
            http.RequestSpec(
                method="GET",
                url="https://example.org/x",
                receive=http.Save(destination),
                timeout_seconds=5.0,
            ),
        )

    assert excinfo.value.kind == "connection"
    assert destination.read_bytes() == b"previous"
    assert [child.name for child in tmp_path.iterdir()] == ["out.bin"]
    adapter.assert_complete()


def test_perform_head_request_has_an_empty_body() -> None:
    # A server may still send a body on a HEAD response; the transport must
    # force it empty and never read it, regardless of what was sent.
    session, adapter = fake_session([{"status": 200, "body": "unexpected"}])

    result = http.perform(
        session, http.RequestSpec(method="HEAD", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert result.text == ""
    assert result.body_bytes == 0
    adapter.assert_complete()


def test_perform_follows_redirects_to_the_final_url() -> None:
    session, adapter = fake_session(
        [
            {"status": 302, "headers": {"Location": "https://example.org/final"}},
            {"status": 200, "body": "done"},
        ]
    )

    result = http.perform(
        session,
        http.RequestSpec(method="GET", url="https://example.org/start", timeout_seconds=5.0),
    )

    assert result.url == "https://example.org/final"
    assert result.text == "done"
    adapter.assert_complete()


def test_perform_reports_the_method_of_the_final_redirected_request() -> None:
    session, adapter = fake_session(
        [
            {"status": 303, "headers": {"Location": "https://example.org/final"}},
            {"status": 200, "body": "done", "expect": {"method": "GET"}},
        ]
    )

    result = http.perform(
        session,
        http.RequestSpec(method="POST", url="https://example.org/start", timeout_seconds=5.0),
    )

    assert result.method == "GET"
    adapter.assert_complete()


def test_perform_follows_redirects_for_head_with_an_empty_body() -> None:
    session, adapter = fake_session(
        [
            {"status": 302, "headers": {"Location": "https://example.org/final"}},
            {"status": 200, "body": "unexpected"},
        ]
    )

    result = http.perform(
        session,
        http.RequestSpec(method="HEAD", url="https://example.org/start", timeout_seconds=5.0),
    )

    assert result.url == "https://example.org/final"
    assert result.text == ""
    assert adapter.sent[1].method == "HEAD"
    adapter.assert_complete()


def test_perform_does_not_follow_redirects_when_disabled() -> None:
    session, adapter = fake_session(
        [{"status": 302, "headers": {"Location": "https://example.org/final"}}]
    )

    result = http.perform(
        session,
        http.RequestSpec(
            method="GET",
            url="https://example.org/start",
            follow_redirects=False,
            timeout_seconds=5.0,
        ),
    )

    assert result.status == 302
    adapter.assert_complete()


def test_perform_sends_an_inactivity_timeout_tuple() -> None:
    session, adapter = fake_session([{"status": 200, "body": ""}])

    http.perform(
        session,
        http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0),
    )

    assert adapter.timeouts == [(5.0, 5.0)]
    adapter.assert_complete()


def test_perform_disables_the_timeout_when_none() -> None:
    session, adapter = fake_session([{"status": 200, "body": ""}])

    http.perform(
        session,
        http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=None),
    )

    assert adapter.timeouts == [None]
    adapter.assert_complete()


def test_perform_sends_verify_false_to_the_transport() -> None:
    session, adapter = fake_session([{"status": 200, "body": ""}])

    http.perform(
        session,
        http.RequestSpec(
            method="GET", url="https://example.org/x", verify_tls=False, timeout_seconds=5.0
        ),
    )

    assert adapter.verifies == [False]
    adapter.assert_complete()


def test_perform_streams_the_response() -> None:
    session, adapter = fake_session([{"status": 200, "body": ""}])

    http.perform(
        session, http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert adapter.streams == [True]
    adapter.assert_complete()


def test_perform_reports_a_non_negative_elapsed_time() -> None:
    session, adapter = fake_session([{"status": 200, "body": ""}])

    result = http.perform(
        session, http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert result.elapsed >= 0
    adapter.assert_complete()


@pytest.mark.parametrize("kind", ["url", "connection", "timeout", "tls", "redirect"])
def test_perform_classifies_every_connect_time_failure_kind(kind: str) -> None:
    session, adapter = fake_session([{"fail": kind}])

    with pytest.raises(http.TransportError) as excinfo:
        http.perform(
            session,
            http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0),
        )

    assert excinfo.value.kind == kind
    adapter.assert_complete()


def test_perform_classifies_too_many_redirects_with_the_redirect_count() -> None:
    session, adapter = fake_session([{"fail": "redirect"}])

    with pytest.raises(http.TransportError) as excinfo:
        http.perform(
            session,
            http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0),
        )

    assert excinfo.value.kind == "redirect"
    assert excinfo.value.redirects == session.max_redirects
    adapter.assert_complete()


def test_perform_classifies_too_many_redirects_with_a_scripted_redirect_count() -> None:
    session, adapter = fake_session([{"fail": "redirect", "redirects": 2}])

    with pytest.raises(http.TransportError) as excinfo:
        http.perform(
            session,
            http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0),
        )

    assert excinfo.value.kind == "redirect"
    assert excinfo.value.redirects == 2
    adapter.assert_complete()


def test_perform_classifies_a_mid_stream_read_timeout_as_timeout() -> None:
    session, adapter = fake_session(
        [{"status": 200, "body_hex": b"partial".hex(), "fail_mid_stream": "timeout"}]
    )

    with pytest.raises(http.TransportError) as excinfo:
        http.perform(
            session,
            http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0),
        )

    assert excinfo.value.kind == "timeout"
    adapter.assert_complete()


def test_perform_classifies_a_mid_stream_tls_failure_as_tls() -> None:
    session, adapter = fake_session(
        [{"status": 200, "body_hex": b"partial".hex(), "fail_mid_stream": "tls"}]
    )

    with pytest.raises(http.TransportError) as excinfo:
        http.perform(
            session,
            http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0),
        )

    assert excinfo.value.kind == "tls"
    adapter.assert_complete()


def test_perform_classifies_a_mid_stream_connection_failure_as_connection() -> None:
    session, adapter = fake_session(
        [{"status": 200, "body_hex": b"partial".hex(), "fail_mid_stream": "connection"}]
    )

    with pytest.raises(http.TransportError) as excinfo:
        http.perform(
            session,
            http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0),
        )

    assert excinfo.value.kind == "connection"
    adapter.assert_complete()


def test_perform_classifies_a_mid_stream_decode_error_as_the_fallback_connection_kind() -> None:
    """``urllib3.DecodeError`` becomes ``ContentDecodingError``, which has no dedicated kind."""
    session, adapter = fake_session(
        [{"status": 200, "body_hex": b"partial".hex(), "fail_mid_stream": "decode"}]
    )

    with pytest.raises(http.TransportError) as excinfo:
        http.perform(
            session,
            http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0),
        )

    assert excinfo.value.kind == "connection"
    adapter.assert_complete()


def test_perform_sends_an_other_method_verbatim() -> None:
    session, adapter = fake_session([{"expect": {"method": "purge"}, "status": 200, "body": ""}])

    http.perform(
        session, http.RequestSpec(method="purge", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert adapter.sent[0].method == "purge"
    adapter.assert_complete()


def test_perform_classifies_an_invalid_method_token_as_request() -> None:
    session, adapter = fake_session([])

    with pytest.raises(http.TransportError) as excinfo:
        http.perform(
            session,
            http.RequestSpec(method="GET \n", url="https://example.org/x", timeout_seconds=5.0),
        )

    assert excinfo.value.kind == "request"
    adapter.assert_complete()


def test_perform_classifies_an_invalid_header_as_request() -> None:
    session, adapter = fake_session([])

    with pytest.raises(http.TransportError) as excinfo:
        http.perform(
            session,
            http.RequestSpec(
                method="GET",
                url="https://example.org/x",
                headers={"X-Bad": "a\nb"},
                timeout_seconds=5.0,
            ),
        )

    assert excinfo.value.kind == "request"
    adapter.assert_complete()


@pytest.mark.parametrize(
    ("name", "value"),
    [("X-Snowman-☃", "ok"), ("X-Test", "snowman: ☃")],
)
def test_perform_classifies_a_header_the_wire_cannot_encode_as_request(
    name: str, value: str
) -> None:
    session, adapter = fake_session([])

    with pytest.raises(http.TransportError) as excinfo:
        http.perform(
            session,
            http.RequestSpec(
                method="GET",
                url="https://example.org/x",
                headers={name: value},
                timeout_seconds=5.0,
            ),
        )

    assert excinfo.value.kind == "request"
    assert value not in excinfo.value.message
    adapter.assert_complete()


def test_classify_treats_an_invalid_header_as_a_request_failure_without_its_message() -> None:
    """``_validate_headers`` now catches every header ``requests`` itself would reject, so this
    exercises ``_classify``'s own safety net directly rather than through ``perform``."""
    exc = requests.exceptions.InvalidHeader("Invalid header value b'Bearer secret\\n'")

    result = http._classify(exc, requests.Session())

    assert result.kind == "request"
    assert "secret" not in result.message


def test_perform_names_only_the_header_for_an_invalid_header_value() -> None:
    """A credential with a control character must never reach the failure message."""
    session, adapter = fake_session([])

    with pytest.raises(http.TransportError) as excinfo:
        http.perform(
            session,
            http.RequestSpec(
                method="GET",
                url="https://example.org/x",
                headers={"X-Token": "Bearer secret\n"},
                timeout_seconds=5.0,
            ),
        )

    assert excinfo.value.kind == "request"
    assert "secret" not in excinfo.value.message
    assert "x-token" in excinfo.value.message.lower()
    adapter.assert_complete()


def test_perform_names_only_the_header_for_an_invalid_rendered_auth_value() -> None:
    """A ``spec.auth`` string is merged into the request headers before sending."""
    session, adapter = fake_session([])

    with pytest.raises(http.TransportError) as excinfo:
        http.perform(
            session,
            http.RequestSpec(
                method="GET",
                url="https://example.org/x",
                auth="Bearer secret\n",
                timeout_seconds=5.0,
            ),
        )

    assert excinfo.value.kind == "request"
    assert "secret" not in excinfo.value.message
    adapter.assert_complete()


@pytest.mark.parametrize("cookies", [{"bad\nname": "value"}, {"session": "secret\r\nvalue"}])
def test_perform_classifies_an_invalid_cookie_header_as_a_request_failure(
    cookies: dict[str, str],
) -> None:
    session, adapter = fake_session([])

    with pytest.raises(http.TransportError) as excinfo:
        http.perform(
            session,
            http.RequestSpec(
                method="GET", url="https://example.org/x", cookies=cookies, timeout_seconds=5.0
            ),
        )

    assert excinfo.value.kind == "request"
    assert "secret" not in excinfo.value.message
    adapter.assert_complete()


def test_perform_suppresses_the_insecure_warning_only_for_that_call() -> None:
    session, adapter = fake_session([{"status": 200, "body": ""}])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        http.perform(
            session,
            http.RequestSpec(
                method="GET", url="https://example.org/x", verify_tls=False, timeout_seconds=5.0
            ),
        )
        assert not any(
            issubclass(w.category, urllib3.exceptions.InsecureRequestWarning) for w in caught
        )
        warnings.warn(urllib3.exceptions.InsecureRequestWarning("issued right after the call"))
        assert any(
            issubclass(w.category, urllib3.exceptions.InsecureRequestWarning) for w in caught
        )
    adapter.assert_complete()


def test_perform_emits_no_insecure_warning_when_verify_tls_is_true() -> None:
    session, adapter = fake_session([{"status": 200, "body": ""}])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        http.perform(
            session,
            http.RequestSpec(
                method="GET", url="https://example.org/x", verify_tls=True, timeout_seconds=5.0
            ),
        )
        assert not any(
            issubclass(w.category, urllib3.exceptions.InsecureRequestWarning) for w in caught
        )
    adapter.assert_complete()


@pytest.mark.parametrize(
    ("content_type", "expected"),
    [
        (None, "utf-8"),
        ("text/plain", "utf-8"),
        ("text/plain; charset=iso-8859-1", "iso-8859-1"),
        ('text/plain; charset="UTF-16"', "UTF-16"),
        ("text/plain; CHARSET=UTF-16", "UTF-16"),
        ("text/plain; charset='utf-16'", "utf-16"),
        ("text/plain; xcharset=iso-8859-1", "utf-8"),
        # A present but empty declaration is not a fallback to UTF-8: it is
        # returned empty and fails to decode.
        ("text/plain; charset", ""),
        ('text/plain; charset=""', ""),
        # An RFC 2231 extended parameter carries its own charset and language.
        ("text/plain; charset*=us-ascii''utf-16", "utf-16"),
    ],
)
def test_resolve_charset(content_type: str | None, expected: str) -> None:
    assert http.resolve_charset(content_type) == expected


@pytest.mark.parametrize(
    ("charset", "body_hex"),
    [
        # "+2AA-": valid UTF-7 for U+D800. Forbidden by the Encoding Standard,
        # and Python would otherwise decode it straight into a lone surrogate.
        ("utf-7", "2b3241412d"),
        # UTF-32LE "a": forbidden by the Encoding Standard.
        ("utf-32", "61000000"),
        # A Python-only codec, never a wire charset.
        ("unicode_escape", "61"),
        # Historically escaped as an uncaught ``UnicodeError``; now rejected up front.
        ("undefined", "61"),
        # ``codecs.lookup`` raises ``ValueError`` rather than ``LookupError`` for this label.
        ("bad\0codec", "61"),
        # A present but empty declaration names no codec, so it decodes nothing.
        ("", "61"),
    ],
)
def test_perform_rejects_a_charset_outside_the_text_encoding_allowlist(
    charset: str, body_hex: str
) -> None:
    session, adapter = fake_session(
        [
            {
                "status": 200,
                "headers": {"Content-Type": f"text/plain; charset={charset}"},
                "body_hex": body_hex,
            }
        ]
    )

    with pytest.raises(http.DecodeFailure) as excinfo:
        http.perform(
            session,
            http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0),
        )

    assert excinfo.value.encoding == charset
    adapter.assert_complete()


@pytest.mark.parametrize(
    ("charset", "body_hex", "expected"),
    [
        ("iso-8859-1", "63e9", "cé"),
        ("shift_jis", "82a0", "あ"),
        ("utf-16", "fffe4100", "A"),
        # Encoding Standard labels whose Python codec is not the windows-125x
        # the standard maps them to, nor the label's own spelling.
        ("iso-8859-9", "e7", "ç"),
        ("latin5", "fe", "ş"),
        ("iso-8859-11", "a1", "ก"),
        ("tis-620", "a1", "ก"),
        ("windows-31j", "82a0", "あ"),
        ("cp949", "b0a1", "가"),
        ("uhc", "b0a1", "가"),
    ],
)
def test_perform_decodes_every_allowlisted_charset_sampled(
    charset: str, body_hex: str, expected: str
) -> None:
    session, adapter = fake_session(
        [
            {
                "status": 200,
                "headers": {"Content-Type": f"text/plain; charset={charset}"},
                "body_hex": body_hex,
            }
        ]
    )

    result = http.perform(
        session, http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert result.text == expected
    assert result.encoding == charset
    adapter.assert_complete()


def test_iso_8859_1_decodes_high_bytes_as_python_latin1_not_windows_1252() -> None:
    """Byte 0x80 is undefined in windows-1252's C1 remapping but is U+0080 in
    plain Latin-1: the allowlisted label decodes with Python's same-name codec,
    never the Encoding Standard's windows-1252 remapping."""
    session, adapter = fake_session(
        [
            {
                "status": 200,
                "headers": {"Content-Type": "text/plain; charset=iso-8859-1"},
                "body_hex": "80",
            }
        ]
    )

    result = http.perform(
        session, http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0)
    )

    assert result.text == "\u0080"


def test_no_allowlisted_charset_can_decode_to_a_surrogate() -> None:
    """A cheap sample, not an exhaustive sweep: every single byte, plus a spread
    of two-byte sequences, decoded leniently under every allowlisted codec.
    """
    samples = [bytes([b]) for b in range(256)]
    lead_bytes = (0x00, 0x41, 0x80, 0xA0, 0xC0, 0xE0, 0xFF)
    samples += [bytes((a, b)) for a in lead_bytes for b in range(256)]
    for name in http._TEXT_CHARSETS:
        for sample in samples:
            decoded = sample.decode(name, errors="ignore")
            assert not any(0xD800 <= ord(ch) <= 0xDFFF for ch in decoded), (name, sample)


def test_perform_save_digests_the_body_it_wrote(tmp_path: Path) -> None:
    destination = tmp_path / "out.bin"
    content = b"archive bytes"
    session, adapter = fake_session([{"status": 200, "body_hex": content.hex()}])

    result = http.perform(
        session,
        http.RequestSpec(
            method="GET",
            url="https://example.org/x",
            receive=http.Save(destination, sha256=True),
            timeout_seconds=5.0,
        ),
    )

    assert destination.read_bytes() == content
    assert result.body_sha256 == hashlib.sha256(content).hexdigest()
    adapter.assert_complete()


def test_perform_runs_on_headers_once_before_streaming_the_body(tmp_path: Path) -> None:
    destination = tmp_path / "out.bin"
    session, adapter = fake_session([{"status": 200, "body_hex": b"hello".hex()}])
    body_written_when_called: list[bool] = []

    http.perform(
        session,
        http.RequestSpec(
            method="GET",
            url="https://example.org/x",
            receive=http.Save(destination),
            timeout_seconds=5.0,
        ),
        on_headers=lambda: body_written_when_called.append(destination.exists()),
    )

    assert body_written_when_called == [False]
    assert destination.read_bytes() == b"hello"
    adapter.assert_complete()


def test_perform_runs_on_headers_only_after_the_last_redirect_hop() -> None:
    session, adapter = fake_session(
        [
            {"status": 302, "headers": {"location": "https://example.org/final"}},
            {"status": 200, "body": "arrived"},
        ]
    )
    requests_sent_when_called: list[int] = []

    result = http.perform(
        session,
        http.RequestSpec(method="GET", url="https://example.org/x", timeout_seconds=5.0),
        on_headers=lambda: requests_sent_when_called.append(len(adapter.sent)),
    )

    assert requests_sent_when_called == [2]
    assert result.text == "arrived"
    adapter.assert_complete()


def test_perform_save_abandons_a_body_past_its_size_limit(tmp_path: Path) -> None:
    destination = tmp_path / "out.bin"
    destination.write_bytes(b"previous")
    session, adapter = fake_session([{"status": 200, "body_hex": b"far too much".hex()}])

    with pytest.raises(http.SizeLimitExceeded) as excinfo:
        http.perform(
            session,
            http.RequestSpec(
                method="GET",
                url="https://example.org/x",
                receive=http.Save(destination, max_bytes=4),
                timeout_seconds=5.0,
            ),
        )

    assert excinfo.value.path == destination
    assert excinfo.value.limit == 4
    assert destination.read_bytes() == b"previous"
    assert [child.name for child in tmp_path.iterdir()] == ["out.bin"]
    adapter.assert_complete()


def test_perform_save_accepts_a_body_at_its_size_limit(tmp_path: Path) -> None:
    destination = tmp_path / "out.bin"
    session, adapter = fake_session([{"status": 200, "body_hex": b"hello".hex()}])

    result = http.perform(
        session,
        http.RequestSpec(
            method="GET",
            url="https://example.org/x",
            receive=http.Save(destination, max_bytes=5),
            timeout_seconds=5.0,
        ),
    )

    assert destination.read_bytes() == b"hello"
    assert result.body_bytes == 5
    assert result.body_sha256 == ""
    adapter.assert_complete()
