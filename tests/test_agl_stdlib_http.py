"""Behavior and static contracts for the ``std/http`` standard-library module.

Static contracts exercise the pure-AgL surface cheaply through the
type-checker. Behavior tests run real programs through the pipeline against
``FakeHttp`` (never a real network) to cover ``request``'s companion:
value mapping, error classification, the pooled session, and trace records.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.capabilities import HostCapabilities
from agm.agl.pipeline import RunResult
from agm.agl.scope.symbols import AglScopeError
from tests._agl_helpers import run_inline_command
from tests._http_helpers import FakeHttp, fake_session, install
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


# ---------------------------------------------------------------------------
# Behavior: the ``request`` extern, method wrappers, and error mapping
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcomes: list[dict[str, object]],
    source: str,
    **kwargs: object,
) -> tuple[RunResult, FakeHttp]:
    adapter = install(monkeypatch, outcomes)
    result = run_inline_command(
        PipelineDriver(), source, entry_path=tmp_path / "entry.agl", **kwargs
    )
    return result, adapter


def test_every_method_wrapper_and_other_sends_its_verb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [
        {"status": 200, "body": "", "expect": {"method": "GET"}},
        {"status": 200, "body": "", "expect": {"method": "POST"}},
        {"status": 200, "body": "", "expect": {"method": "PUT"}},
        {"status": 200, "body": "", "expect": {"method": "PATCH"}},
        {"status": 200, "body": "", "expect": {"method": "DELETE"}},
        {"status": 200, "body": "", "expect": {"method": "HEAD"}},
        {"status": 200, "body": "", "expect": {"method": "OPTIONS"}},
        {"status": 200, "body": "", "expect": {"method": "PROPFIND"}},
    ]
    source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y")
  let _ = http::post("https://x/y")
  let _ = http::put("https://x/y")
  let _ = http::patch("https://x/y")
  let _ = http::delete("https://x/y")
  let _ = http::head("https://x/y")
  let _ = http::request(http::Method::Options, "https://x/y")
  let _ = http::request(http::Method::Other("PROPFIND"), "https://x/y")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_body_variants_render_content_and_content_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [
        {"status": 200, "body": "", "expect": {"method": "POST"}},
        {
            "status": 200,
            "body": "",
            "expect": {
                "headers": {"content-type": "text/plain"},
                "body": "hi",
            },
        },
        {
            "status": 200,
            "body": "",
            "expect": {
                "headers": {"content-type": "application/json"},
                "body": '{"a": 1}',
            },
        },
        {
            "status": 200,
            "body": "",
            "expect": {
                "headers": {"content-type": "application/x-www-form-urlencoded"},
                "body": "a=1",
            },
        },
    ]
    source = """import std/http
program def main() -> unit =
  let _ = http::post("https://x/y", body = http::Body::Empty)
  let _ = http::post("https://x/y", body = http::Body::Text("hi", "text/plain"))
  let _ = http::post("https://x/y", body = http::Body::Json({"a": 1}))
  let _ = http::post("https://x/y", body = http::Body::Form({"a": "1"}))
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_auth_variants_render_authorization_and_override_a_supplied_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [
        {"status": 200, "body": "", "expect": {"headers": {"authorization": "Basic dXNlcjpwdw=="}}},
        {"status": 200, "body": "", "expect": {"headers": {"authorization": "Bearer tok"}}},
        {"status": 200, "body": "", "expect": {"headers": {"authorization": "Bearer tok2"}}},
    ]
    source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y", auth = http::Auth::Basic("user", "pw"))
  let _ = http::get("https://x/y", auth = http::Auth::Bearer("tok"))
  let _ = http::get("https://x/y",
    headers = http::headers({"Authorization": "stale"}), auth = http::Auth::Bearer("tok2"))
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_auth_override_wins_over_a_raw_non_lowercased_authorization_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A raw ``Headers({...})`` record keeps names verbatim; the override must still win."""
    outcomes = [
        {"status": 200, "body": "", "expect": {"headers": {"authorization": "Bearer tok3"}}}
    ]
    source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y",
    headers = http::Headers({"AUTHORIZATION": "stale"}), auth = http::Auth::Bearer("tok3"))
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_raw_headers_lowercase_and_collapse_names_on_unwrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two raw entries differing only by case fold to one on unwrap; the later entry wins."""
    outcomes = [{"status": 200, "body": "", "expect": {"headers": {"x-custom": "second"}}}]
    source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y",
    headers = http::Headers({"X-Custom": "first", "x-custom": "second"}))
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_plain_headers_are_sent_verbatim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    outcomes = [{"status": 200, "body": "", "expect": {"headers": {"x-trace": "abc"}}}]
    source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y", headers = http::headers({"X-Trace": "abc"}))
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_query_is_appended_to_the_urls_own_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [{"status": 200, "body": "", "expect": {"url": "https://x/y?a=1&b=2"}}]
    source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y?a=1", query = {"b": "2"})
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_cookies_are_sent_as_given(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A single-label host defeats ``http.cookiejar``'s domain matching, so a real one is used."""
    outcomes = [{"status": 200, "body": "", "expect": {"headers": {"cookie": "a=1"}}}]
    source = """import std/http
program def main() -> unit =
  let _ = http::get("https://example.org/y", cookies = {"a": "1"})
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_receive_ignore_drops_the_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outcomes = [{"status": 200, "body": "dropped"}]
    source = """import std/http
program def main() -> unit =
  let r = http::get("https://x/y", receive = http::Receive::Ignore)
  print(r.body)
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()
    assert capsys.readouterr().out == "\n"


def test_download_saves_the_body_to_the_destination_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "downloaded.bin"
    outcomes = [{"status": 200, "body": "payload"}]
    source = f"""import std/http
program def main() -> unit =
  let r = http::download("https://x/y", "{destination}")
  print(r.status)
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()
    assert destination.read_text() == "payload"
    assert capsys.readouterr().out == "200\n"


def test_save_write_failure_raises_fs_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ``Save`` destination whose parent does not exist raises ``std/fs::FsError``."""
    destination = tmp_path / "missing" / "out.bin"
    outcomes = [{"status": 200, "body": "payload"}]
    source = f"""import std/http
program def main() -> unit =
  let _ = http::download("https://x/y", "{destination}")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "FsError"
    assert result.error.fields["operation"] == "write"
    assert result.error.fields["path"] == str(destination)
    adapter.assert_complete()


def test_response_fields_reflect_the_final_exchange(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outcomes = [
        {
            "status": 201,
            "body": "hi",
            "headers": {"X-Custom": "yes", "Set-Cookie": "session=abc"},
        }
    ]
    source = """import std/http
program def main() -> unit =
  let r = http::get("https://x/y")
  print(r.status)
  print(r.url)
  print(r.method.render())
  print(r.headers.get("x-custom"))
  print(r.cookies)
  print(r.elapsed >= 0.0)
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()
    assert capsys.readouterr().out == '201\nhttps://x/y\nGET\nyes\n{"session": "abc"}\ntrue\n'


def test_module_default_timeout_is_forwarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [{"status": 200, "body": "", "expect": {"timeout": (30.0, 30.0)}}]
    source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_timeout_override_via_the_module_var_reaches_the_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [{"status": 200, "body": "", "expect": {"timeout": (2.0, 2.0)}}]
    source = """import std/http
program def main() -> unit =
  http::timeout := Some("2s")
  let _ = http::get("https://x/y")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_timeout_text_is_forwarded_and_invalid_text_raises_type_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [{"status": 200, "body": "", "expect": {"timeout": (5.0, 5.0)}}]
    ok_source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y", timeout = Some("5s"))
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, ok_source)
    assert result.ok, result.error
    adapter.assert_complete()

    bad_source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y", timeout = Some("not-a-duration"))
  ()
"""
    result, _adapter = _run(monkeypatch, tmp_path, [{"status": 200, "body": ""}], bad_source)
    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "TypeError"


@pytest.mark.parametrize("timeout_text", ["0", "0s", "0.0"])
def test_non_positive_timeout_raises_type_error_before_any_transport_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, timeout_text: str
) -> None:
    source = f"""import std/http
program def main() -> unit =
  let _ = http::get("https://x/y", timeout = Some("{timeout_text}"))
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, [], source)
    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "TypeError"
    assert len(adapter.sent) == 0


def test_timeout_none_disables_the_inactivity_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [{"status": 200, "body": "", "expect": {"timeout": None}}]
    source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y", timeout = None)
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


@pytest.mark.parametrize(
    ("fail_kind", "type_name", "field_name"),
    [
        ("url", "HttpUrlError", None),
        ("connection", "HttpConnectionError", "cause"),
        ("timeout", "HttpTimeoutError", "timeout"),
        ("tls", "HttpTlsError", "cause"),
        ("redirect", "HttpRedirectError", "redirects"),
    ],
)
def test_every_transport_error_kind_maps_to_its_http_error_subtype(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_kind: str,
    type_name: str,
    field_name: str | None,
) -> None:
    outcomes = [{"fail": fail_kind}]
    source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y", timeout = Some("5s"))
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == type_name
    assert result.error.fields["url"] == "https://x/y"
    assert result.error.fields["method"] == {"$case": "Get"}
    if field_name == "cause":
        cause = result.error.fields["cause"]
        assert isinstance(cause, str) and cause
    elif field_name == "timeout":
        assert result.error.fields["timeout"] == "5s"
    elif field_name == "redirects":
        redirects = result.error.fields["redirects"]
        assert isinstance(redirects, int) and redirects > 0
    adapter.assert_complete()


def test_request_error_kind_maps_via_an_invalid_method_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An invalid method token is rejected by the transport as a ``request``-kind failure."""
    outcomes = [{"status": 200, "body": ""}]
    source = """import std/http
program def main() -> unit =
  let _ = http::request(http::Method::Other("BAD METHOD"), "https://x/y")
  ()
"""
    result, _adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "HttpRequestError"


def test_a_credential_with_a_control_character_never_leaks_into_the_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bearer token with a trailing newline is an invalid header value; the failure
    must name only the header, never echo the credential, in either the exception or
    the trace."""
    log_path = tmp_path / "trace.jsonl"
    source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y", auth = http::Auth::Bearer("SECRET\\n"))
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, [], source, log_file=log_path)
    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "HttpRequestError"
    assert "SECRET" not in str(result.error.fields)
    assert len(adapter.sent) == 0

    records = _load_jsonl(log_path)
    failure_recs = [r for r in records if r.get("kind") == "http_failure"]
    assert failure_recs
    assert "SECRET" not in str(failure_recs[0]["message"])


def test_decode_failure_raises_http_decode_error_with_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [
        {"status": 200, "headers": {"Content-Type": "text/plain; charset=ascii"}, "body_hex": "ff"}
    ]
    source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "HttpDecodeError"
    assert result.error.fields["status"] == 200
    assert result.error.fields["encoding"] == "ascii"
    headers = result.error.fields["headers"]
    assert isinstance(headers, dict)
    entries = headers["entries"]
    assert isinstance(entries, dict)
    assert entries["content-type"] == "text/plain; charset=ascii"
    adapter.assert_complete()


def test_session_is_created_once_per_run_and_closed_at_the_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [{"status": 200, "body": "a"}, {"status": 200, "body": "b"}]
    session, adapter = fake_session(outcomes)
    open_calls: list[None] = []

    def fake_open() -> object:
        open_calls.append(None)
        return session

    monkeypatch.setattr("agm.core.http.open_session", fake_open)
    closed: list[None] = []
    original_close = session.close

    def spy_close() -> None:
        closed.append(None)
        original_close()

    monkeypatch.setattr(session, "close", spy_close)

    source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/a")
  let _ = http::get("https://x/b")
  ()
"""
    result = run_inline_command(PipelineDriver(), source, entry_path=tmp_path / "entry.agl")
    assert result.ok, result.error
    adapter.assert_complete()
    assert len(open_calls) == 1
    assert len(closed) == 1


def test_trace_records_are_shaped_with_credential_redaction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_path = tmp_path / "trace.jsonl"
    outcomes = [
        {
            "status": 200,
            "headers": {"set-cookie": "session=topsecret"},
            "body": "ok",
        }
    ]
    source = """import std/http
program def main() -> unit =
  let _ = http::post("https://x/y", query = {"q": "1"},
    headers = http::headers({"Proxy-Authorization": "Basic zzz", "Cookie": "a=b"}),
    auth = http::Auth::Bearer("tok"), cookies = {"session": "req-secret"},
    body = http::Body::Text("hi", "text/plain"))
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source, log_file=log_path)
    assert result.ok, result.error
    adapter.assert_complete()

    records = _load_jsonl(log_path)
    request_recs = [r for r in records if r.get("kind") == "http_request"]
    response_recs = [r for r in records if r.get("kind") == "http_response"]
    assert request_recs and response_recs

    request_rec = request_recs[0]
    assert request_rec["headers"]["proxy-authorization"] == "<redacted>"
    assert request_rec["headers"]["cookie"] == "<redacted>"
    assert request_rec["headers"]["authorization"] == "<redacted>"
    assert request_rec["headers"]["content-type"] == "text/plain"
    assert request_rec["cookies"]["session"] == "<redacted>"
    assert request_rec["method"] == "POST"
    assert request_rec["url"] == "https://x/y"
    assert request_rec["query"] == {"q": "1"}
    assert request_rec["body"] == "hi"
    assert request_rec["timeout"] == "30s"
    assert request_rec["follow_redirects"] is True
    assert request_rec["verify_tls"] is True

    response_rec = response_recs[0]
    assert response_rec["headers"]["set-cookie"] == "<redacted>"
    assert response_rec["cookies"]["session"] == "<redacted>"
    assert response_rec["body"] == "ok"
    assert response_rec["status"] == 200
    assert response_rec["url"] == "https://x/y?q=1"


def test_trace_records_receive_kind_and_save_path_for_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_path = tmp_path / "trace.jsonl"
    destination = tmp_path / "out.bin"
    outcomes = [{"status": 200, "body": "payload"}]
    source = f"""import std/http
program def main() -> unit =
  let _ = http::download("https://x/y", "{destination}")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source, log_file=log_path)
    assert result.ok, result.error
    adapter.assert_complete()

    records = _load_jsonl(log_path)
    response_recs = [r for r in records if r.get("kind") == "http_response"]
    assert response_recs
    assert response_recs[0]["receive"] == "save"
    assert response_recs[0]["save_path"] == str(destination)
    assert response_recs[0]["body"] is None
    assert response_recs[0]["body_bytes"] == len("payload")


def test_trace_records_receive_kind_for_ignore(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_path = tmp_path / "trace.jsonl"
    outcomes = [{"status": 200, "body": "dropped"}]
    source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y", receive = http::Receive::Ignore)
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source, log_file=log_path)
    assert result.ok, result.error
    adapter.assert_complete()

    records = _load_jsonl(log_path)
    response_recs = [r for r in records if r.get("kind") == "http_response"]
    assert response_recs
    assert response_recs[0]["receive"] == "ignore"
    assert response_recs[0]["save_path"] is None
    assert response_recs[0]["body"] is None
    assert response_recs[0]["body_bytes"] == len("dropped")


def test_trace_records_a_failure_verbatim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log_path = tmp_path / "trace.jsonl"
    outcomes = [{"fail": "connection"}]
    source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source, log_file=log_path)
    assert not result.ok
    adapter.assert_complete()

    records = _load_jsonl(log_path)
    failure_recs = [r for r in records if r.get("kind") == "http_failure"]
    assert failure_recs
    assert failure_recs[0]["error_type"] == "HttpConnectionError"
    message = failure_recs[0]["message"]
    assert isinstance(message, str) and message
    elapsed = failure_recs[0]["elapsed"]
    assert isinstance(elapsed, int | float) and elapsed >= 0


def test_try_twins_return_ok_on_success_and_err_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outcomes = [{"fail": "connection"}, {"status": 200, "body": "hi"}]
    source = """import std/http
program def main() -> unit =
  let failure = http::try-get("https://x/y")
  print(failure.is-err())
  let success = http::try-get("https://x/y")
  print(success.is-ok())
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()
    assert capsys.readouterr().out == "true\ntrue\n"


@pytest.mark.parametrize(
    ("method_expr", "outcome", "type_name"),
    [
        ("http::Method::Get", {"fail": "url"}, "HttpUrlError"),
        ('http::Method::Other("BAD METHOD")', {"status": 200, "body": ""}, "HttpRequestError"),
        ("http::Method::Get", {"fail": "connection"}, "HttpConnectionError"),
        ("http::Method::Get", {"fail": "timeout"}, "HttpTimeoutError"),
        ("http::Method::Get", {"fail": "tls"}, "HttpTlsError"),
        ("http::Method::Get", {"fail": "redirect"}, "HttpRedirectError"),
        ("http::Method::Get", {"status": 200, "body_hex": "ff"}, "HttpDecodeError"),
    ],
)
def test_try_request_maps_every_error_kind_to_err_with_the_right_subtype(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    method_expr: str,
    outcome: dict[str, object],
    type_name: str,
) -> None:
    # An invalid method token never reaches the transport, so no outcome is scripted.
    outcomes = [] if type_name == "HttpRequestError" else [outcome]
    source = f"""import std/http
program def main() -> unit =
  let r = http::try-request({method_expr}, "https://x/y")
  print(r.is-err())
  print(r)
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()
    output = capsys.readouterr().out
    assert output.startswith("true\n")
    assert f"{type_name}(" in output


# ---------------------------------------------------------------------------
# Behavior: the data-only ``Client``
# ---------------------------------------------------------------------------


def test_client_type_and_constructor_are_visible_through_the_import() -> None:
    resolve_and_check_inline_entry(
        "import std/http\nlet _: http::Client = http::client()\n", HostCapabilities()
    )


def test_client_normalises_headers_at_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A raw ``Headers({...})`` keeps names verbatim; the client normalises on construction."""
    source = """import std/http
program def main() -> unit =
  let api = http::client(headers = http::Headers({"X-A": "1"}))
  print(api.headers.contains("x-a"))
"""
    result, adapter = _run(monkeypatch, tmp_path, [], source)
    assert result.ok, result.error
    adapter.assert_complete()
    assert capsys.readouterr().out == "true\n"


def test_client_joins_a_relative_target_onto_its_base_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [{"status": 200, "body": "", "expect": {"url": "https://api.example.org/repos/x"}}]
    source = """import std/http
program def main() -> unit =
  let api = http::client(base-url = Some("https://api.example.org"))
  let _ = api.get("/repos/x")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_client_joins_onto_a_base_url_with_a_trailing_slash_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A base path with a trailing ``/`` keeps its full path per RFC 3986 join rules."""
    outcomes = [
        {"status": 200, "body": "", "expect": {"url": "https://api.example.org/v1/repos/x"}}
    ]
    source = """import std/http
program def main() -> unit =
  let api = http::client(base-url = Some("https://api.example.org/v1/"))
  let _ = api.get("repos/x")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_client_resolves_a_query_only_reference_against_the_base_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A reference that is only a query keeps the base's path, per RFC 3986 merge rules."""
    outcomes = [
        {"status": 200, "body": "", "expect": {"url": "https://api.example.org/v1/?page=2"}}
    ]
    source = """import std/http
program def main() -> unit =
  let api = http::client(base-url = Some("https://api.example.org/v1/"))
  let _ = api.get("?page=2")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_client_uses_an_absolute_target_as_given_over_its_base_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [{"status": 200, "body": "", "expect": {"url": "https://other.example/direct"}}]
    source = """import std/http
program def main() -> unit =
  let api = http::client(base-url = Some("https://api.example.org"))
  let _ = api.get("https://other.example/direct")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_client_without_a_base_url_uses_the_target_as_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [{"status": 200, "body": "", "expect": {"url": "https://x/y"}}]
    source = """import std/http
program def main() -> unit =
  let api = http::client()
  let _ = api.get("https://x/y")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_client_merges_its_headers_with_call_headers_winning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [
        {
            "status": 200,
            "body": "",
            "expect": {"headers": {"x-common": "call", "x-client-only": "yes"}},
        }
    ]
    source = """import std/http
program def main() -> unit =
  let api = http::client(
    headers = http::headers({"X-Common": "client", "X-Client-Only": "yes"}))
  let _ = api.get("https://x/y", headers = http::headers({"X-Common": "call"}))
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_client_policy_fields_apply_to_every_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """auth, cookies, timeout, follow-redirects, and verify-tls all come from the client.

    The target host is a real multi-label domain: a single-label host such as
    ``x`` defeats ``http.cookiejar``'s domain matching, so no ``cookie`` header
    would be sent regardless of policy (see ``test_cookies_are_sent_as_given``).
    """
    outcomes = [
        {
            "status": 302,
            "headers": {"location": "https://example.org/other"},
            "body": "",
            "expect": {
                "headers": {"authorization": "Bearer tok", "cookie": "a=1"},
                "timeout": (9.0, 9.0),
                "verify": False,
            },
        }
    ]
    source = """import std/http
program def main() -> unit =
  let api = http::client(auth = http::Auth::Bearer("tok"), cookies = {"a": "1"},
    timeout = Some("9s"), follow-redirects = false, verify-tls = false)
  let r = api.get("https://example.org/y")
  print(r.status)
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()
    # follow-redirects = false means the 302 itself comes back, unfollowed.
    assert capsys.readouterr().out == "302\n"


def test_with_override_changes_client_policy_for_one_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [{"status": 200, "body": "", "expect": {"timeout": (120.0, 120.0)}}]
    source = """import std/http
program def main() -> unit =
  let api = http::client()
  let _ = (api with timeout = Some("2m")).get("https://x/y")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_client_captures_the_module_timeout_at_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A later ``http::timeout`` write must not reach a client built before it."""
    outcomes = [{"status": 200, "body": "", "expect": {"timeout": (5.0, 5.0)}}]
    source = """import std/http
program def main() -> unit =
  http::timeout := Some("5s")
  let api = http::client()
  http::timeout := Some("2s")
  let _ = api.get("https://x/y")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_client_timeout_none_at_construction_means_no_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``client(timeout = None)`` disables the inactivity bound regardless of the
    module setting in effect at construction."""
    outcomes = [{"status": 200, "body": "", "expect": {"timeout": None}}]
    source = """import std/http
program def main() -> unit =
  http::timeout := Some("5s")
  let api = http::client(timeout = None)
  let _ = api.get("https://x/y")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()


def test_client_every_verb_and_download_send_their_method(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = [
        {"status": 200, "body": "", "expect": {"method": "GET"}},
        {"status": 200, "body": "", "expect": {"method": "POST"}},
        {"status": 200, "body": "", "expect": {"method": "PUT"}},
        {"status": 200, "body": "", "expect": {"method": "PATCH"}},
        {"status": 200, "body": "", "expect": {"method": "DELETE"}},
        {"status": 200, "body": "", "expect": {"method": "HEAD"}},
        {"status": 200, "body": "", "expect": {"method": "OPTIONS"}},
        {"status": 200, "body": "", "expect": {"method": "PROPFIND"}},
        {"status": 200, "body": "payload", "expect": {"method": "GET"}},
    ]
    destination = tmp_path / "downloaded.bin"
    source = f"""import std/http
program def main() -> unit =
  let api = http::client()
  let _ = api.get("https://x/y")
  let _ = api.post("https://x/y")
  let _ = api.put("https://x/y")
  let _ = api.patch("https://x/y")
  let _ = api.delete("https://x/y")
  let _ = api.head("https://x/y")
  let _ = api.request(http::Method::Options, "https://x/y")
  let _ = api.request(http::Method::Other("PROPFIND"), "https://x/y")
  let _ = api.download("https://x/y", "{destination}")
  ()
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()
    assert destination.read_text() == "payload"


def test_client_try_twins_delegate_to_the_free_try_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outcomes = [
        {"fail": "connection", "expect": {"method": "GET"}},
        {"status": 200, "body": "", "expect": {"method": "GET"}},
        {"status": 200, "body": "", "expect": {"method": "POST"}},
        {"status": 200, "body": "", "expect": {"method": "PUT"}},
        {"status": 200, "body": "", "expect": {"method": "PATCH"}},
        {"status": 200, "body": "", "expect": {"method": "DELETE"}},
        {"status": 200, "body": "", "expect": {"method": "HEAD"}},
        {
            "status": 200,
            "body": "",
            "expect": {
                "url": "https://api.example.org/v1/widgets",
                "headers": {
                    "x-common": "call",
                    "x-client-only": "yes",
                    "authorization": "Bearer tok",
                    "cookie": "a=1",
                },
                "timeout": (9.0, 9.0),
                "verify": False,
            },
        },
    ]
    source = """import std/http
program def main() -> unit =
  let api = http::client()
  let failure = api.try-request(http::Method::Get, "https://x/y")
  print(failure.is-err())
  print(api.try-get("https://x/y").is-ok())
  print(api.try-post("https://x/y").is-ok())
  print(api.try-put("https://x/y").is-ok())
  print(api.try-patch("https://x/y").is-ok())
  print(api.try-delete("https://x/y").is-ok())
  print(api.try-head("https://x/y").is-ok())
  let full = http::client(base-url = Some("https://api.example.org/v1/"),
    headers = http::headers({"X-Common": "client", "X-Client-Only": "yes"}),
    auth = http::Auth::Bearer("tok"), cookies = {"a": "1"}, timeout = Some("9s"),
    follow-redirects = false, verify-tls = false)
  print(full.try-get("widgets", headers = http::headers({"X-Common": "call"})).is-ok())
"""
    result, adapter = _run(monkeypatch, tmp_path, outcomes, source)
    assert result.ok, result.error
    adapter.assert_complete()
    assert capsys.readouterr().out == "true\n" * 8
