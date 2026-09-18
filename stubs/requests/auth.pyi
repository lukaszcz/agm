"""Minimal type stub for ``requests.auth``.

Only ``AuthBase`` is typed: the base class ``agm.core.http`` subclasses to
install a session auth that blocks ``.netrc`` lookup without adding a header.
"""

from requests.models import PreparedRequest

class AuthBase:
    def __call__(self, r: PreparedRequest) -> PreparedRequest: ...
