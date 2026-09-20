"""Companion contracts for the ``std/url`` standard-library module."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor, NominalKind
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.boundary import AglException, decode_boundary_value
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.semantics.values import ArrayValue, IntValue, RecordValue, TextValue
from tests._agl_helpers import option_nominal_descriptors

_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "packages" / "stdlib"
_URL_MODULE = ModuleId(("std", "url"))
_URL = NominalId(9_900_001)
_URL_PARSE_ERROR = NominalId(9_900_002)
_OPTION = NominalId(9_900_003)
_OPTION_NONE = NominalId(9_900_004)
_OPTION_SOME = NominalId(9_900_005)
_PAIR = NominalId(9_900_006)


class _UrlCompanion(Protocol):
    def parse(self, value: str) -> object: ...

    def render(self, value: object) -> str: ...

    def join(self, base: str, reference: str) -> str: ...

    def encode(self, value: str) -> str: ...

    def decode(self, value: str) -> str: ...

    def encode_query(self, fields: object) -> str: ...

    def parse_query(self, value: str) -> object: ...


def _url_companion() -> tuple[_UrlCompanion, ExternRegistry]:
    """Load ``std/url`` through the same extern boundary as production.

    Returns the registry alongside the companion so a test can build a
    hand-crafted record with its synthesized ``Url``/``Option``/``Pair``
    classes, the same classes the companion itself constructs values with.
    """
    registry = ExternRegistry()
    registry.set_nominals(
        {
            _URL: NominalDescriptor(
                nominal=_URL,
                module_id=_URL_MODULE,
                scope_path=(),
                declared_name="Url",
                kind=NominalKind.RECORD,
                fields=("scheme", "host", "port", "path", "query", "fragment"),
            ),
            _URL_PARSE_ERROR: NominalDescriptor(
                nominal=_URL_PARSE_ERROR,
                module_id=_URL_MODULE,
                scope_path=(),
                declared_name="UrlParseError",
                kind=NominalKind.EXCEPTION,
                fields=("message", "raw"),
            ),
            **option_nominal_descriptors(_OPTION, _OPTION_NONE, _OPTION_SOME),
            _PAIR: NominalDescriptor(
                nominal=_PAIR,
                module_id=ModuleId(("std", "pair")),
                scope_path=(),
                declared_name="Pair",
                kind=NominalKind.RECORD,
                fields=("first", "second"),
            ),
        },
    )
    module: ModuleType = registry.load_companion(_URL_MODULE, _STDLIB_ROOT / "src" / "url.py")
    return cast(_UrlCompanion, module), registry


def _pair(first: str, second: str) -> RecordValue:
    return RecordValue(_PAIR, {"first": TextValue(first), "second": TextValue(second)})


def _none_value(registry: ExternRegistry) -> object:
    return registry._nominal_classes[_OPTION_NONE]()


def _some_value(registry: ExternRegistry, value: object) -> object:
    return registry._nominal_classes[_OPTION_SOME](value=value)


def _url_record(registry: ExternRegistry, **fields: object) -> object:
    return registry._nominal_classes[_URL](**fields)


def _pair_value(registry: ExternRegistry, first: str, second: str) -> object:
    return registry._nominal_classes[_PAIR](first=first, second=second)


def _parse_error_raw(companion: _UrlCompanion, value: str) -> str:
    """Parse *value*, returning the caught ``UrlParseError``'s ``raw`` field."""
    with pytest.raises(AglException) as exc_info:
        companion.parse(value)
    assert exc_info.value.value.nominal == _URL_PARSE_ERROR
    raw = exc_info.value.value.fields["raw"]
    assert isinstance(raw, TextValue)
    return raw.value


def test_parse_extracts_every_component_of_a_full_url() -> None:
    companion, _ = _url_companion()

    parsed = decode_boundary_value(companion.parse("https://example.org:8443/a/b?x=1#top"))

    assert parsed == RecordValue(
        _URL,
        {
            "scheme": TextValue("https"),
            "host": TextValue("example.org"),
            "port": RecordValue(_OPTION_SOME, {"value": IntValue(8443)}),
            "path": TextValue("/a/b"),
            "query": ArrayValue([_pair("x", "1")]),
            "fragment": RecordValue(_OPTION_SOME, {"value": TextValue("top")}),
        },
    )
    assert companion.render(companion.parse("https://example.org:8443/a/b?x=1#top")) == (
        "https://example.org:8443/a/b?x=1#top"
    )


def test_parse_leaves_absent_port_and_fragment_as_none() -> None:
    companion, _ = _url_companion()

    parsed = decode_boundary_value(companion.parse("https://example.org"))

    assert parsed == RecordValue(
        _URL,
        {
            "scheme": TextValue("https"),
            "host": TextValue("example.org"),
            "port": RecordValue(_OPTION_NONE, {}),
            "path": TextValue(""),
            "query": ArrayValue([]),
            "fragment": RecordValue(_OPTION_NONE, {}),
        },
    )
    assert companion.render(companion.parse("https://example.org")) == "https://example.org"


def test_render_reproduces_an_empty_but_present_fragment() -> None:
    companion, _ = _url_companion()

    assert companion.render(companion.parse("https://example.org/a#")) == ("https://example.org/a#")


def test_parse_preserves_repeated_query_keys_order_and_blank_values() -> None:
    companion, _ = _url_companion()

    parsed = decode_boundary_value(companion.parse("https://example.org/a?x=1&x=2&y="))

    assert parsed.fields["query"] == ArrayValue([_pair("x", "1"), _pair("x", "2"), _pair("y", "")])
    assert companion.render(companion.parse("https://example.org/a?x=1&x=2&y=")) == (
        "https://example.org/a?x=1&x=2&y="
    )


@pytest.mark.parametrize("value", ("relative/path", "//example.org/path"))
def test_parse_rejects_a_url_with_no_scheme(value: str) -> None:
    companion, _ = _url_companion()
    assert _parse_error_raw(companion, value) == value


@pytest.mark.parametrize("value", ("mailto:foo@bar.com", "urn:example"))
def test_parse_rejects_a_url_with_no_authority_marker(value: str) -> None:
    companion, _ = _url_companion()
    assert _parse_error_raw(companion, value) == value


@pytest.mark.parametrize("value", ("https:///path", "https:///"))
def test_parse_rejects_an_empty_host_outside_the_file_scheme(value: str) -> None:
    companion, _ = _url_companion()
    assert _parse_error_raw(companion, value) == value


def test_parse_accepts_an_empty_host_for_the_file_scheme_and_round_trips() -> None:
    companion, _ = _url_companion()

    parsed = decode_boundary_value(companion.parse("file:///tmp/x"))

    assert parsed == RecordValue(
        _URL,
        {
            "scheme": TextValue("file"),
            "host": TextValue(""),
            "port": RecordValue(_OPTION_NONE, {}),
            "path": TextValue("/tmp/x"),
            "query": ArrayValue([]),
            "fragment": RecordValue(_OPTION_NONE, {}),
        },
    )
    assert companion.render(companion.parse("file:///tmp/x")) == "file:///tmp/x"


@pytest.mark.parametrize(
    "value",
    (
        "https://user@example.org/",
        "https://:pass@example.org/",
        "https://user:pass@example.org/",
    ),
)
def test_parse_rejects_userinfo(value: str) -> None:
    companion, _ = _url_companion()
    assert _parse_error_raw(companion, value) == value


@pytest.mark.parametrize("value", ("https://example.org:notaport/", "https://example.org:99999/"))
def test_parse_rejects_an_invalid_port(value: str) -> None:
    companion, _ = _url_companion()
    assert _parse_error_raw(companion, value) == value


@pytest.mark.parametrize(
    "value",
    (
        "https://exa\tmple.org/a",
        "https://exa mple.org/a",
        "https://example.org/a\nb",
    ),
)
def test_parse_rejects_control_characters_and_spaces(value: str) -> None:
    companion, _ = _url_companion()
    assert _parse_error_raw(companion, value) == value


@pytest.mark.parametrize("value", ("https://[::1/a", "https://[zz]/", "https://[127.0.0.1]/"))
def test_parse_rejects_a_malformed_bracketed_host(value: str) -> None:
    companion, _ = _url_companion()
    assert _parse_error_raw(companion, value) == value


@pytest.mark.parametrize(
    "value", ("https://exa<mple.org/", "https://exa{mple.org/", "https://exa|mple.org/")
)
def test_parse_rejects_a_host_outside_the_reg_name_character_class(value: str) -> None:
    companion, _ = _url_companion()
    assert _parse_error_raw(companion, value) == value


@pytest.mark.parametrize(
    "value",
    (
        "https://example.org/%zz",
        "https://example.org/%",
        "https://example.org/a%F",
        "https://example.org/?a=%zz",
        "https://example.org/a#%zz",
        "https://example.org/%c3%28",
        "https://example.org/?a=%c3%28",
        "https://example.org/a#%c3%28",
    ),
)
def test_parse_rejects_malformed_percent_encoding_anywhere(value: str) -> None:
    companion, _ = _url_companion()
    assert _parse_error_raw(companion, value) == value


def test_parse_and_render_round_trip_an_ipv6_host_with_and_without_a_port() -> None:
    companion, _ = _url_companion()

    with_port = decode_boundary_value(companion.parse("https://[::1]:8080/a"))
    assert with_port.fields["host"] == TextValue("::1")
    assert with_port.fields["port"] == RecordValue(_OPTION_SOME, {"value": IntValue(8080)})
    assert companion.render(companion.parse("https://[::1]:8080/a")) == "https://[::1]:8080/a"

    without_port = decode_boundary_value(companion.parse("https://[::1]/a"))
    assert without_port.fields["host"] == TextValue("::1")
    assert without_port.fields["port"] == RecordValue(_OPTION_NONE, {})
    assert companion.render(companion.parse("https://[::1]/a")) == "https://[::1]/a"


def test_parse_rejects_a_host_with_a_non_utf8_percent_encoded_octet() -> None:
    companion, _ = _url_companion()
    value = "http://x%FF.org/"
    assert _parse_error_raw(companion, value) == value


def test_parse_rejects_an_ipv6_zone_id_whose_percent_delimiter_is_not_encoded() -> None:
    companion, _ = _url_companion()
    value = "http://[fe80::1%eth0]/"
    assert _parse_error_raw(companion, value) == value


def test_parse_and_render_round_trip_an_ipv6_zone_id_using_the_rfc_6874_percent_form() -> None:
    companion, _ = _url_companion()

    parsed = decode_boundary_value(companion.parse("http://[::1%25lo]/"))

    assert parsed.fields["host"] == TextValue("::1%25lo")
    assert companion.render(companion.parse("http://[::1%25lo]/")) == "http://[::1%25lo]/"


@pytest.mark.parametrize(
    "value",
    (
        "https://x/?",
        "https://x/?a=%20&b=+",
        "https://x/?flag",
        "https://x/?next=/a:b",
        "https://x/?a=1&&b=2",
        "https://x/%7e",
        "https://x:/a",
        "https://x:080/a",
        "HTTPS://EXAMPLE.ORG/A",
        "https://[::1]:8080/a",
        "https://[::1]/a",
        "http://x/é",
        'http://x/"',
        "http://x/<>",
        "http://x/a[b]",
        "http://x/a\\b",
        "http://x/p?q#f#g",
    ),
)
def test_parse_and_render_round_trip_at_the_record_level_for_edge_inputs(value: str) -> None:
    """``parse(u.render()) == u`` for every ``Url`` ``parse`` returns; the text
    matches the input only when it was already canonical (e.g. the trailing
    ``?`` above is dropped, not reproduced)."""
    companion, _ = _url_companion()

    parsed = decode_boundary_value(companion.parse(value))
    reparsed = decode_boundary_value(companion.parse(companion.render(companion.parse(value))))

    assert parsed == reparsed


def test_render_prefixes_a_bare_non_empty_path_with_a_slash() -> None:
    companion, registry = _url_companion()
    record = _url_record(
        registry,
        scheme="https",
        host="x",
        port=_none_value(registry),
        path="a",
        query=[],
        fragment=_none_value(registry),
    )

    assert companion.render(record) == "https://x/a"


def test_render_brackets_an_ipv6_host_and_places_the_port_after_the_bracket() -> None:
    companion, registry = _url_companion()
    record = _url_record(
        registry,
        scheme="https",
        host="::1",
        port=_some_value(registry, 8080),
        path="/a",
        query=[],
        fragment=_none_value(registry),
    )

    assert companion.render(record) == "https://[::1]:8080/a"


def test_render_quotes_reserved_characters_in_a_hand_built_path_and_fragment() -> None:
    companion, registry = _url_companion()
    record = _url_record(
        registry,
        scheme="https",
        host="x",
        port=_none_value(registry),
        path="a?b#c d",
        query=[],
        fragment=_some_value(registry, "e/f?g h"),
    )

    rendered = companion.render(record)

    assert rendered == "https://x/a%3Fb%23c%20d#e/f?g%20h"
    # The rendered path/fragment survive a fresh parse without splitting early.
    reparsed = decode_boundary_value(companion.parse(rendered))
    assert reparsed.fields["path"] == TextValue("/a%3Fb%23c%20d")


def test_render_keeps_an_existing_percent_escape_intact() -> None:
    companion, registry = _url_companion()
    record = _url_record(
        registry,
        scheme="https",
        host="x",
        port=_none_value(registry),
        path="/%7e",
        query=[],
        fragment=_none_value(registry),
    )

    assert companion.render(record) == "https://x/%7e"


def test_render_rejects_a_hand_built_record_with_an_empty_host_outside_the_file_scheme() -> None:
    companion, registry = _url_companion()
    record = _url_record(
        registry,
        scheme="https",
        host="",
        port=_none_value(registry),
        path="/",
        query=[],
        fragment=_none_value(registry),
    )

    with pytest.raises(AglException) as exc_info:
        companion.render(record)
    assert exc_info.value.value.nominal == _URL_PARSE_ERROR


def test_render_accepts_a_hand_built_record_with_an_empty_host_for_the_file_scheme() -> None:
    companion, registry = _url_companion()
    record = _url_record(
        registry,
        scheme="file",
        host="",
        port=_none_value(registry),
        path="/tmp/x",
        query=[],
        fragment=_none_value(registry),
    )

    assert companion.render(record) == "file:///tmp/x"


@pytest.mark.parametrize("port", (-1, 99999))
def test_render_rejects_a_hand_built_record_with_a_port_out_of_range(port: int) -> None:
    companion, registry = _url_companion()
    record = _url_record(
        registry,
        scheme="https",
        host="x",
        port=_some_value(registry, port),
        path="/",
        query=[],
        fragment=_none_value(registry),
    )

    with pytest.raises(AglException) as exc_info:
        companion.render(record)
    assert exc_info.value.value.nominal == _URL_PARSE_ERROR


def test_render_rejects_a_hand_built_record_with_an_invalid_host() -> None:
    companion, registry = _url_companion()
    record = _url_record(
        registry,
        scheme="https",
        host="a b",
        port=_none_value(registry),
        path="/",
        query=[],
        fragment=_none_value(registry),
    )

    with pytest.raises(AglException) as exc_info:
        companion.render(record)
    assert exc_info.value.value.nominal == _URL_PARSE_ERROR


def test_render_rejects_a_hand_built_record_with_a_host_that_looks_like_a_bad_ipv6_literal() -> (
    None
):
    companion, registry = _url_companion()
    record = _url_record(
        registry,
        scheme="https",
        host="a:b",
        port=_none_value(registry),
        path="/",
        query=[],
        fragment=_none_value(registry),
    )

    with pytest.raises(AglException) as exc_info:
        companion.render(record)
    assert exc_info.value.value.nominal == _URL_PARSE_ERROR


@pytest.mark.parametrize("scheme", ("HTTPS", "1https", "ht tps"))
def test_render_rejects_a_hand_built_record_with_a_non_canonical_or_invalid_scheme(
    scheme: str,
) -> None:
    companion, registry = _url_companion()
    record = _url_record(
        registry,
        scheme=scheme,
        host="x",
        port=_none_value(registry),
        path="/",
        query=[],
        fragment=_none_value(registry),
    )

    with pytest.raises(AglException) as exc_info:
        companion.render(record)
    assert exc_info.value.value.nominal == _URL_PARSE_ERROR


@pytest.mark.parametrize(
    ("reference", "expected"),
    (
        # RFC 3986 5.4.1, normal examples.
        ("g:h", "g:h"),
        ("g", "http://a/b/c/g"),
        ("./g", "http://a/b/c/g"),
        ("g/", "http://a/b/c/g/"),
        ("/g", "http://a/g"),
        ("//g", "http://g"),
        ("?y", "http://a/b/c/d;p?y"),
        ("g?y", "http://a/b/c/g?y"),
        ("#s", "http://a/b/c/d;p?q#s"),
        ("g#s", "http://a/b/c/g#s"),
        ("g?y#s", "http://a/b/c/g?y#s"),
        (";x", "http://a/b/c/;x"),
        ("g;x", "http://a/b/c/g;x"),
        ("g;x?y#s", "http://a/b/c/g;x?y#s"),
        ("", "http://a/b/c/d;p?q"),
        (".", "http://a/b/c/"),
        ("./", "http://a/b/c/"),
        ("..", "http://a/b/"),
        ("../", "http://a/b/"),
        ("../g", "http://a/b/g"),
        ("../..", "http://a/"),
        ("../../", "http://a/"),
        ("../../g", "http://a/g"),
        # RFC 3986 5.4.2, abnormal examples.
        ("../../../g", "http://a/g"),
        ("../../../../g", "http://a/g"),
        ("/./g", "http://a/g"),
        ("/../g", "http://a/g"),
        ("g.", "http://a/b/c/g."),
        (".g", "http://a/b/c/.g"),
        ("g..", "http://a/b/c/g.."),
        ("..g", "http://a/b/c/..g"),
        ("./../g", "http://a/b/g"),
        ("./g/.", "http://a/b/c/g/"),
        ("g/./h", "http://a/b/c/g/h"),
        ("g/../h", "http://a/b/c/h"),
        ("g;x=1/./y", "http://a/b/c/g;x=1/y"),
        ("g;x=1/../y", "http://a/b/c/y"),
        ("g?y/./x", "http://a/b/c/g?y/./x"),
        ("g?y/../x", "http://a/b/c/g?y/../x"),
        ("g#s/./x", "http://a/b/c/g#s/./x"),
        ("g#s/../x", "http://a/b/c/g#s/../x"),
        # A scheme-relative reference resolves its own opaque path.
        ("g:./p", "g:p"),
        ("g:../p", "g:p"),
        ("g:.", "g:"),
        ("g:..", "g:"),
        # RFC 3986 5.4.2's strict-parser vector: a reference whose scheme
        # matches the base's own is still resolved as scheme-relative, never
        # folded onto the base's authority the way pre-RFC parsers did.
        ("http:g", "http:g"),
    ),
)
def test_join_follows_rfc_3986_reference_resolution(reference: str, expected: str) -> None:
    companion, _ = _url_companion()

    assert companion.join("http://a/b/c/d;p?q", reference) == expected


def test_join_works_for_a_non_http_scheme() -> None:
    companion, _ = _url_companion()

    assert companion.join("foo://a/b", "c") == "foo://a/c"


def test_join_merges_onto_an_authority_with_an_empty_base_path() -> None:
    companion, _ = _url_companion()

    assert companion.join("https://x", "y") == "https://x/y"


def test_join_raises_when_the_base_does_not_parse() -> None:
    companion, _ = _url_companion()

    with pytest.raises(AglException) as exc_info:
        companion.join("relative/path", "g")
    assert exc_info.value.value.nominal == _URL_PARSE_ERROR


@pytest.mark.parametrize("reference", ("g h", "%zz", "g\x00"))
def test_join_rejects_a_reference_with_a_textual_defect(reference: str) -> None:
    companion, _ = _url_companion()

    with pytest.raises(AglException) as exc_info:
        companion.join("http://a/b/c/d;p?q", reference)
    assert exc_info.value.value.nominal == _URL_PARSE_ERROR
    assert exc_info.value.value.fields["raw"] == TextValue(reference)


@pytest.mark.parametrize("reference", ("//u@h/x", "//[zz]/", "//h:99999/", "//"))
def test_join_rejects_a_resolved_result_with_a_bad_authority(reference: str) -> None:
    companion, _ = _url_companion()

    with pytest.raises(AglException) as exc_info:
        companion.join("http://a/b/c/d;p?q", reference)
    assert exc_info.value.value.nominal == _URL_PARSE_ERROR


def test_encode_and_decode_round_trip_reserved_characters() -> None:
    companion, _ = _url_companion()
    raw = "a b/c?d#e"

    encoded = companion.encode(raw)

    assert " " not in encoded and "/" not in encoded
    assert companion.decode(encoded) == raw


@pytest.mark.parametrize("value", ("%zz", "%", "a%F", "%c3%28"))
def test_decode_raises_on_a_malformed_escape_or_invalid_utf8(value: str) -> None:
    companion, _ = _url_companion()

    with pytest.raises(AglException) as exc_info:
        companion.decode(value)
    assert exc_info.value.value.nominal == _URL_PARSE_ERROR
    assert exc_info.value.value.fields["raw"] == TextValue(value)


def test_encode_query_and_parse_query_round_trip_repeats_and_blanks() -> None:
    companion, _ = _url_companion()
    query = "x=1&x=2&y="

    fields = companion.parse_query(query)

    assert decode_boundary_value(fields) == ArrayValue(
        [_pair("x", "1"), _pair("x", "2"), _pair("y", "")]
    )
    assert companion.encode_query(fields) == query


def test_encode_query_and_parse_query_handle_no_fields() -> None:
    companion, _ = _url_companion()

    assert companion.encode_query([]) == ""
    assert decode_boundary_value(companion.parse_query("")) == ArrayValue([])


def test_parse_query_treats_a_bare_key_as_an_empty_value() -> None:
    companion, _ = _url_companion()

    assert decode_boundary_value(companion.parse_query("flag")) == ArrayValue([_pair("flag", "")])


def test_parse_query_ignores_an_empty_segment_between_ampersands() -> None:
    companion, _ = _url_companion()

    assert decode_boundary_value(companion.parse_query("a=1&&b=2")) == ArrayValue(
        [_pair("a", "1"), _pair("b", "2")]
    )


@pytest.mark.parametrize("value", ("%zz", "a=%c3%28"))
def test_parse_query_raises_on_a_malformed_escape_or_invalid_utf8(value: str) -> None:
    companion, _ = _url_companion()

    with pytest.raises(AglException) as exc_info:
        companion.parse_query(value)
    assert exc_info.value.value.nominal == _URL_PARSE_ERROR
    assert exc_info.value.value.fields["raw"] == TextValue(value)
