"""Minimal type stub for ``requests.cookies``.

Only ``RequestsCookieJar`` (the pieces ``agm.core.http`` reads and writes)
and ``create_cookie`` are typed.
"""

from http.cookiejar import Cookie, CookieJar, CookiePolicy

class RequestsCookieJar(CookieJar):
    def get_dict(self) -> dict[str, str | None]: ...
    def set_policy(self, policy: CookiePolicy) -> None: ...
    def set(self, name: str, value: str) -> Cookie | None: ...

def create_cookie(name: str, value: str, *, domain: str = ...) -> Cookie: ...
