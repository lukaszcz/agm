"""Minimal type stubs for ``requests``.

Only the pieces used by ``agm.core.http`` (the transport seam) and
``agm.packages.fetch`` (the archive fetcher) are typed: ``Session``,
``Request``/``PreparedRequest``/``Response`` (see ``requests.models``), and
the auth base class (see ``requests.auth``). Kept free of ``Any`` throughout,
unlike the real package's own inline types, so an isinstance check or a bare
class reference against ``requests.exceptions`` (see that stub) never taints
its expression type.
"""

from requests.auth import AuthBase
from requests.cookies import RequestsCookieJar
from requests.models import PreparedRequest as PreparedRequest
from requests.models import Request as Request
from requests.models import Response as Response
from typing import TypedDict

class _EnvironmentSettings(TypedDict):
    proxies: dict[str, str]
    stream: bool
    verify: bool | str
    cert: str | tuple[str, str] | None

class Session:
    auth: AuthBase | tuple[str, str] | None
    cookies: RequestsCookieJar
    max_redirects: int
    trust_env: bool
    def __init__(self) -> None: ...
    def prepare_request(self, request: Request) -> PreparedRequest: ...
    def merge_environment_settings(
        self,
        url: str,
        proxies: dict[str, str],
        stream: bool | None,
        verify: bool | str | None,
        cert: str | tuple[str, str] | None,
    ) -> _EnvironmentSettings: ...
    def send(
        self,
        request: PreparedRequest,
        *,
        timeout: float | tuple[float, float] | None = ...,
        allow_redirects: bool = ...,
        proxies: dict[str, str] = ...,
        stream: bool = ...,
        verify: bool | str = ...,
        cert: str | tuple[str, str] | None = ...,
    ) -> Response: ...
    def should_strip_auth(self, old_url: str, new_url: str) -> bool: ...
    def rebuild_auth(self, prepared_request: PreparedRequest, response: Response) -> None: ...
    def get(self, url: str, *, stream: bool, timeout: float) -> Response: ...
    def close(self) -> None: ...
