"""URL parsing, rendering, joining, and percent-/query-encoding for ``std/url``."""

from __future__ import annotations

import ipaddress
import re
import string
from typing import NoReturn
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

from agl import AglException, array, nominals, option_none, option_some

Option = nominals.std.option.Option
Pair = nominals.std.pair.Pair
Url = nominals.std.url.Url
UrlParseError = nominals.std.url.UrlParseError

# ASCII control characters (including DEL) and space: ``urlsplit`` silently
# strips some of these from its input, so they are rejected up front instead.
_CONTROL_OR_SPACE_RE = re.compile(r"[\x00-\x20\x7f]")
# A ``%`` not followed by two hex digits.
_MALFORMED_PERCENT_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_HEX_PAIR_RE = re.compile(r"[0-9A-Fa-f]{2}")
# RFC 3986 reg-name: unreserved / pct-encoded / sub-delims.
_REG_NAME_RE = re.compile(r"(?:[A-Za-z0-9\-._~!$&'()*+,;=]|%[0-9A-Fa-f]{2})+\Z")
# RFC 6874 ZoneID: 1*( unreserved / pct-encoded ), after its "%25" introducer.
_ZONE_ID_RE = re.compile(r"(?:[A-Za-z0-9\-._~]|%[0-9A-Fa-f]{2})+\Z")

_UNRESERVED = string.ascii_letters + string.digits + "-._~"
_SUB_DELIMS = "!$&'()*+,;="
_PCHAR_SAFE = _UNRESERVED + _SUB_DELIMS + ":@"
_PATH_SAFE = _PCHAR_SAFE + "/"
_FRAGMENT_SAFE = _PATH_SAFE + "?"

# A URI-reference's optional scheme, up to and including its ":".
_SCHEME_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*:")
# RFC 3986 scheme, canonical (lowercase) form only.
_LOWERCASE_SCHEME_RE = re.compile(r"[a-z][a-z0-9+.\-]*\Z")


def _parse_error(raw: str, reason: str) -> NoReturn:
    raise AglException(UrlParseError(message=f"Could not parse URL: {reason}.", raw=raw))


def _pairs(values: object) -> list[tuple[str, str]]:
    return [(pair.first, pair.second) for pair in values]


def _validate_host(host: str) -> None:
    """Raise ``ValueError`` unless *host* is a valid reg-name or IP-literal for a URI.

    Shared by ``parse`` (where a host containing ``:`` already came from a
    bracketed IP-literal ``urlsplit`` split out) and ``render`` (where a
    hand-built record's ``host`` field is unvalidated text), so a host
    containing ``:`` is independently checked as an IPv6 address here too
    (a colon rules out IPv4, so a successful parse is always IPv6).
    ``ipaddress`` accepts a bare RFC 4007 zone id (``fe80::1%eth0``) as
    readily as a percent-encoded one, since it has no notion of URI syntax,
    so an optional zone id is checked here against RFC 6874: its
    introducing ``%`` must itself be percent-encoded (``%25``), never bare.
    A reg-name host is checked against the RFC 3986 grammar and for valid
    percent-encoding (a percent-encoded octet must decode as UTF-8).
    """
    if ":" in host:
        try:
            ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError(f"invalid host {host!r}") from exc
        _, sep, zone = host.partition("%")
        if sep and not (zone.startswith("25") and _ZONE_ID_RE.fullmatch(zone[2:])):
            raise ValueError(f"invalid zone id in host {host!r}")
        return
    if not _REG_NAME_RE.fullmatch(host):
        raise ValueError(f"invalid host {host!r}")
    _require_valid_percent_encoding(host)


def _require_valid_percent_encoding(component: str) -> None:
    """Raise ``ValueError`` on a malformed ``%`` escape or invalid UTF-8 in *component*."""
    if _MALFORMED_PERCENT_RE.search(component):
        raise ValueError("malformed percent-encoding")
    try:
        unquote(component, errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid UTF-8 after percent-decoding") from exc


def _decode_pairs(query: str) -> list[tuple[str, str]]:
    """Decode a query string into key/value pairs; see :func:`_require_valid_percent_encoding`."""
    _require_valid_percent_encoding(query)
    return parse_qsl(query, keep_blank_values=True, errors="strict")


def parse(value: str) -> object:
    """Parse *value* into its components; see ``url.agl`` for the exact rules."""
    if _CONTROL_OR_SPACE_RE.search(value):
        _parse_error(value, "contains a control character or space")
    try:
        parts = urlsplit(value)
    except ValueError as exc:
        _parse_error(value, str(exc))
    if not parts.scheme:
        _parse_error(value, "missing scheme")
    if value[len(parts.scheme) : len(parts.scheme) + 3] != "://":
        _parse_error(value, "missing authority")
    if parts.username is not None or parts.password is not None:
        _parse_error(value, "URL carries userinfo, which Url cannot represent")
    try:
        host = parts.hostname or ""
        port = parts.port
    except ValueError as exc:
        _parse_error(value, str(exc))
    if not host and parts.scheme != "file":
        _parse_error(value, "missing host")
    try:
        if host:
            _validate_host(host)
        _require_valid_percent_encoding(parts.path)
        query_pairs = _decode_pairs(parts.query)
        if "#" in value:
            _require_valid_percent_encoding(parts.fragment)
    except ValueError as exc:
        _parse_error(value, str(exc))
    return Url(
        scheme=parts.scheme,
        host=host,
        port=option_some(port) if port is not None else option_none(),
        path=_quote_component(parts.path, _PATH_SAFE),
        query=array([Pair(first=key, second=val) for key, val in query_pairs]),
        fragment=(
            option_some(_quote_component(parts.fragment, _FRAGMENT_SAFE))
            if "#" in value
            else option_none()
        ),
    )


def _quote_component(value: str, safe: str) -> str:
    """Percent-encode *value* outside *safe*, keeping any existing ``%XX`` escape intact."""
    pieces: list[str] = []
    index, length = 0, len(value)
    while index < length:
        char = value[index]
        if char == "%" and _HEX_PAIR_RE.match(value, index + 1):
            pieces.append(value[index : index + 3])
            index += 3
        elif char in safe:
            pieces.append(char)
            index += 1
        else:
            pieces.append(quote(char, safe=""))
            index += 1
    return "".join(pieces)


def render(self: object) -> str:
    """Reconstruct canonical URL text from *self*'s components; see ``url.agl``.

    A hand-built record's ``scheme``/``host``/``port`` are not otherwise
    checked, so they are validated here through the same rules ``parse``
    applies, raising ``UrlParseError(raw=<the offending field>)`` rather than
    emitting text that would not itself parse or would parse back to a
    different record.
    """
    if not _LOWERCASE_SCHEME_RE.fullmatch(self.scheme):
        _parse_error(self.scheme, "invalid or non-canonical scheme")
    if not self.host and self.scheme != "file":
        _parse_error(self.host, "missing host")
    if self.host:
        try:
            _validate_host(self.host)
        except ValueError as exc:
            _parse_error(self.host, str(exc))
    host = f"[{self.host}]" if ":" in self.host else self.host
    if isinstance(self.port, Option.Some):
        if not 0 <= self.port.value <= 65535:
            _parse_error(str(self.port.value), "port out of range 0-65535")
        host = f"{host}:{self.port.value}"
    path = self.path
    if path and not path.startswith("/"):
        path = "/" + path
    rendered = f"{self.scheme}://{host}{_quote_component(path, _PATH_SAFE)}"
    query = urlencode(_pairs(self.query))
    if query:
        rendered += f"?{query}"
    if isinstance(self.fragment, Option.Some):
        fragment = self.fragment.value
        rendered += f"#{_quote_component(fragment, _FRAGMENT_SAFE)}"
    return rendered


def _split_reference(value: str) -> tuple[str | None, str | None, str, str | None, str | None]:
    """Split a URI-reference into (scheme, authority, path, query, fragment).

    Each optional component is ``None`` when absent, distinct from
    present-but-empty (RFC 3986 appendix B).
    """
    rest, sep, fragment_part = value.partition("#")
    fragment = fragment_part if sep else None
    rest, sep, query_part = rest.partition("?")
    query = query_part if sep else None
    scheme = None
    match = _SCHEME_RE.match(rest)
    if match is not None:
        scheme = match.group(0)[:-1]
        rest = rest[match.end() :]
    authority = None
    if rest.startswith("//"):
        rest = rest[2:]
        slash = rest.find("/")
        end = slash if slash != -1 else len(rest)
        authority, rest = rest[:end], rest[end:]
    return scheme, authority, rest, query, fragment


def _remove_dot_segments(path: str) -> str:
    """RFC 3986 section 5.2.4: collapse ``.``/``..`` segments in *path*."""
    output = ""
    while path:
        if path.startswith("../"):
            path = path[3:]
        elif path.startswith("./"):
            path = path[2:]
        elif path.startswith("/./"):
            path = "/" + path[3:]
        elif path == "/.":
            path = "/"
        elif path.startswith("/../"):
            path = "/" + path[4:]
            output = output[: output.rfind("/")] if "/" in output else ""
        elif path == "/..":
            path = "/"
            output = output[: output.rfind("/")] if "/" in output else ""
        elif path in (".", ".."):
            path = ""
        else:
            body = path[1:] if path.startswith("/") else path
            slash = body.find("/")
            end = slash if slash != -1 else len(body)
            segment = (path[:1] if path.startswith("/") else "") + body[:end]
            path = body[end:]
            output += segment
    return output


def _merge_paths(base_path: str, ref_path: str) -> str:
    """RFC 3986 section 5.3: merge a relative-path reference onto *base_path*.

    ``join``'s base always parses, so it always has a defined authority: an
    empty base path merges directly onto it. A non-empty base path always
    starts with ``/`` (see :func:`_split_reference`), so it always has a
    rightmost segment to replace.
    """
    if base_path == "":
        return "/" + ref_path
    return base_path[: base_path.rfind("/") + 1] + ref_path


def _recompose(
    scheme: str | None, authority: str | None, path: str, query: str | None, fragment: str | None
) -> str:
    """RFC 3986 section 5.3: recompose the resolved components into URI text."""
    result = f"{scheme}:" if scheme is not None else ""
    if authority is not None:
        result += f"//{authority}"
    result += path
    if query is not None:
        result += f"?{query}"
    if fragment is not None:
        result += f"#{fragment}"
    return result


def _validate_reference(reference: str) -> None:
    """Raise ``UrlParseError(raw=reference)`` on the same textual defects ``parse`` rejects."""
    if _CONTROL_OR_SPACE_RE.search(reference):
        _parse_error(reference, "contains a control character or space")
    try:
        _require_valid_percent_encoding(reference)
    except ValueError as exc:
        _parse_error(reference, str(exc))


def join(base: str, reference: str) -> str:
    """Resolve *reference* against *base* per RFC 3986 section 5.2, for any scheme.

    A resolved result carrying an authority is re-validated through
    :func:`parse` (a network-path reference such as ``//u@h/x`` can smuggle
    in userinfo, a bad host, or a bad port); a result with no authority
    (e.g. ``g:h``) has no such component to check and stays as resolved.
    """
    parse(base)
    _validate_reference(reference)
    b_scheme, b_authority, b_path, b_query, _ = _split_reference(base)
    r_scheme, r_authority, r_path, r_query, r_fragment = _split_reference(reference)

    if r_scheme is not None:
        scheme, authority = r_scheme, r_authority
        path, query = _remove_dot_segments(r_path), r_query
    else:
        scheme = b_scheme
        if r_authority is not None:
            authority = r_authority
            path, query = _remove_dot_segments(r_path), r_query
        else:
            authority = b_authority
            if r_path == "":
                path = b_path
                query = r_query if r_query is not None else b_query
            else:
                if r_path.startswith("/"):
                    path = _remove_dot_segments(r_path)
                else:
                    path = _remove_dot_segments(_merge_paths(b_path, r_path))
                query = r_query
    resolved = _recompose(scheme, authority, path, query, r_fragment)
    if authority is not None:
        parse(resolved)
    return resolved


def encode(value: str) -> str:
    """Percent-encode every reserved character in *value*."""
    return quote(value, safe="")


def decode(value: str) -> str:
    """Percent-decode *value*.

    Raises ``UrlParseError(raw=value)`` on a malformed escape or invalid UTF-8.
    """
    try:
        _require_valid_percent_encoding(value)
    except ValueError:
        _parse_error(value, "malformed percent-encoding")
    return unquote(value, errors="strict")


def encode_query(fields: object) -> str:
    """Render ordered *fields* as a form-encoded query string."""
    return urlencode(_pairs(fields))


def parse_query(value: str) -> object:
    """Parse a query string into ordered pairs.

    Raises ``UrlParseError(raw=value)`` on a malformed escape or invalid UTF-8;
    see :func:`decode`.
    """
    try:
        pairs = _decode_pairs(value)
    except ValueError:
        _parse_error(value, "malformed percent-encoding")
    return array([Pair(first=key, second=val) for key, val in pairs])


__all__ = ["decode", "encode", "encode_query", "join", "parse", "parse_query", "render"]
