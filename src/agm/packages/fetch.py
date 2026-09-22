"""Streaming URL fetch boundary for package archives."""

from __future__ import annotations

import signal
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from agm.core import http as core_http
from agm.packages.errors import FetchError as FetchError
from agm.packages.manifest import parse_sha256

# Bounds one connect or read of the transport, and how long the archive's first
# byte may take to arrive.  A healthy transfer of any size keeps going; the
# archive size limit, not elapsed time, is what bounds a download that never
# stalls.
_FETCH_TIMEOUT_SECONDS = 30.0
MAX_ARCHIVE_DOWNLOAD_SIZE = 128 * 1024 * 1024


def fetch_archive(
    *,
    requirement: str,
    url: str,
    expected_hash: str,
    handoff: Callable[[Path], None],
    scratch_dir: Path | None = None,
) -> None:
    """Fetch, hash-verify, and hand an archive to the installer.

    Transport, streaming, digesting and the size cap are the shared HTTP seam's
    (:mod:`agm.core.http`); this boundary adds the temporary archive, the digest
    comparison, and the ``FetchError`` vocabulary.  The temporary archive exists
    only for the handoff callback.  Keeping archive extraction outside this seam
    keeps the archive format independent of URL transport and its verification
    contract.

    A started transfer is bounded by inactivity rather than by total elapsed
    time, so a large healthy download completes while a stalled connection
    still fails fast.
    """

    expected_digest = _expected_digest(requirement, expected_hash)
    directory = Path(tempfile.gettempdir()) if scratch_dir is None else scratch_dir
    archive = directory / f"agm-fetch-{uuid4().hex}.agmpkg"
    primary_failure = False
    try:
        with _connect_alarm(requirement, _FETCH_TIMEOUT_SECONDS) as connected:
            digest = _download(requirement, url, archive, on_headers=connected)
        if digest != expected_digest:
            raise FetchError(f"hash mismatch for {requirement}")
        handoff(archive)
    except BaseException:
        primary_failure = True
        raise
    finally:
        _discard_archive(archive, requirement, primary_failure=primary_failure)


def _download(requirement: str, url: str, archive: Path, *, on_headers: Callable[[], None]) -> str:
    """Stream *url* into *archive*; returns the SHA-256 hex digest of what arrived."""

    session = core_http.open_session()
    spec = core_http.RequestSpec(
        method="GET",
        url=url,
        timeout_seconds=_FETCH_TIMEOUT_SECONDS,
        receive=core_http.Save(archive, max_bytes=MAX_ARCHIVE_DOWNLOAD_SIZE, sha256=True),
    )
    try:
        response = core_http.perform(session, spec, on_headers=on_headers)
    except core_http.TransportError as exc:
        if exc.kind == "timeout":
            raise FetchError(f"fetch timed out for {requirement}") from exc
        raise FetchError(f"fetch failed for {requirement}: {exc}") from exc
    except core_http.SizeLimitExceeded as exc:
        raise FetchError(
            f"fetch failed for {requirement}: package archive exceeds the download size limit"
        ) from exc
    except core_http.SaveFailure as exc:
        raise FetchError(f"fetch failed for {requirement}: {exc.error}") from exc
    finally:
        session.close()
    if response.status >= 400:
        raise FetchError(f"fetch failed for {requirement}: HTTP status {response.status}")
    return response.body_sha256


def _discard_archive(archive: Path, requirement: str, *, primary_failure: bool) -> None:
    """Remove a transfer's temporary archive, if one was ever written.

    A cleanup failure is reported only when nothing else already failed: the
    transfer's own error is what the caller needs to see.
    """
    try:
        archive.unlink(missing_ok=True)
    except OSError as exc:
        if not primary_failure:
            raise FetchError(f"fetch failed for {requirement}: cleanup failed: {exc}") from exc


@contextmanager
def _connect_alarm(requirement: str, budget: float) -> Iterator[Callable[[], None]]:
    """Fail the fetch if reaching the archive's first byte takes longer than *budget*.

    Name resolution, connect and each redirect hop can block outside any timed
    socket operation, so the transport's own timeout does not bound them;
    ``SIGALRM`` does.  Streaming the body needs no such cover -- every read of
    it is a timed socket read -- so the yielded callable drops the alarm as
    soon as the response headers arrive, leaving a slow but live download
    bounded only by read inactivity and the size cap.  Any alarm the process
    already had is restored, minus the time this fetch took.
    """

    def expire(_signum: int, _frame: object) -> None:
        raise FetchError(f"fetch timed out for {requirement}")

    def connected() -> None:
        signal.setitimer(signal.ITIMER_REAL, 0)

    started = time.monotonic()
    previous_handler = signal.signal(signal.SIGALRM, expire)
    previous_remaining, previous_interval = signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        yield connected
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_remaining > 0:
            elapsed = time.monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL, max(previous_remaining - elapsed, 1e-6), previous_interval
            )


def _expected_digest(requirement: str, expected_hash: str) -> str:
    """Extract a SHA-256 hex digest from a manifest hash declaration."""

    digest = parse_sha256(expected_hash)
    if digest is None:
        raise FetchError(f"hash mismatch for {requirement}")
    return digest
