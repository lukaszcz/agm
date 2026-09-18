"""Minimal type stub for ``requests.models``.

Only ``Request``, ``PreparedRequest``, and ``Response`` are typed, covering
exactly what ``agm.core.http`` (the transport seam) and ``agm.packages.fetch``
(the archive fetcher, structurally, through its own ``Response`` protocol)
use. The real ``Response.raw`` is untyped (``Any``); ``_RawStream`` below is
this stub's own concrete stand-in, typed without ``Any``.
"""

from collections.abc import Iterator, Mapping, MutableMapping

from requests.cookies import RequestsCookieJar

class Request:
    def __init__(
        self,
        *,
        method: str | None = ...,
        url: str | None = ...,
        params: object = ...,
        headers: Mapping[str, str] | None = ...,
        data: object = ...,
        auth: tuple[str, str] | None = ...,
        cookies: RequestsCookieJar | Mapping[str, str] | None = ...,
    ) -> None: ...

class PreparedRequest:
    method: str
    url: str
    headers: MutableMapping[str, str]
    body: bytes | str | None

class _RawStream:
    def read1(self, _amt: int, _decode_content: bool, /) -> bytes: ...

class Response:
    status_code: int
    headers: Mapping[str, str]
    url: str
    cookies: RequestsCookieJar
    history: list[Response]
    request: PreparedRequest | None
    raw: _RawStream
    @property
    def content(self) -> bytes: ...
    def iter_content(self, chunk_size: int | None = ...) -> Iterator[bytes]: ...
    def close(self) -> None: ...
    def raise_for_status(self) -> None: ...
    def __enter__(self) -> Response: ...
    def __exit__(self, *args: object) -> None: ...
