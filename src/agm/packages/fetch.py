"""Streaming URL fetch boundary for package archives."""

from __future__ import annotations

import hashlib
import signal
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests

from agm.packages.errors import FetchError as FetchError
from agm.packages.manifest import parse_sha256

# Bounds one connect or read of the transport, and how long a started transfer
# may make no progress.  A healthy transfer of any size keeps going; the archive
# size limit, not elapsed time, is what bounds a download that never stalls.
_FETCH_TIMEOUT_SECONDS = 30.0
_CHUNK_SIZE = 1024 * 1024
MAX_ARCHIVE_DOWNLOAD_SIZE = 128 * 1024 * 1024


class Response(Protocol):
    """The small response surface used by the fetch seam."""

    def __enter__(self) -> Response: ...

    def __exit__(self, *_: object) -> None: ...

    def raise_for_status(self) -> None: ...

    def iter_content(self, _chunk_size: int) -> Iterator[bytes]: ...


class Session(Protocol):
    """The small requests-session surface used by the fetch seam."""

    def get(self, url: str, *, stream: bool, timeout: float) -> Response: ...


def fetch_archive(
    *,
    requirement: str,
    url: str,
    expected_hash: str,
    handoff: Callable[[Path], None],
    session: Session | None = None,
    scratch_dir: Path | None = None,
) -> None:
    """Fetch, hash-verify, and hand an archive to the installer.

    The temporary archive exists only for the handoff callback.  Keeping archive
    extraction outside this seam keeps the archive format independent of URL
    transport and its verification contract.

    A started transfer is bounded by inactivity rather than by total elapsed
    time, so a large healthy download completes while a stalled connection
    still fails fast.
    """

    expected_digest = _expected_digest(requirement, expected_hash)
    client: Session = requests.Session() if session is None else session
    archive: Path | None = None
    primary_failure = False
    stall = _StallDeadline(requirement, _FETCH_TIMEOUT_SECONDS)
    try:
        try:
            with tempfile.NamedTemporaryFile(
                dir=scratch_dir, suffix=".agmpkg", delete=False
            ) as file:
                archive = Path(file.name)
                digest = hashlib.sha256()
                downloaded_size = 0
                try:
                    with _wall_clock_deadline(stall):
                        with client.get(
                            url, stream=True, timeout=_FETCH_TIMEOUT_SECONDS
                        ) as response:
                            response.raise_for_status()
                            stall.check()
                            for chunk in response.iter_content(_CHUNK_SIZE):
                                stall.check()
                                if chunk:
                                    downloaded_size += len(chunk)
                                    if downloaded_size > MAX_ARCHIVE_DOWNLOAD_SIZE:
                                        raise FetchError(
                                            f"fetch failed for {requirement}: "
                                            "package archive exceeds the download size limit"
                                        )
                                    file.write(chunk)
                                    digest.update(chunk)
                                    stall.progressed()
                except FetchError:
                    raise
                except Exception as exc:
                    raise FetchError(f"fetch failed for {requirement}: {exc}") from exc
        except OSError as exc:
            raise FetchError(f"fetch failed for {requirement}: {exc}") from exc
        if digest.hexdigest() != expected_digest:
            raise FetchError(f"hash mismatch for {requirement}")
        handoff(archive)
    except BaseException:
        primary_failure = True
        raise
    finally:
        _discard_archive(archive, requirement, primary_failure=primary_failure)


def _discard_archive(archive: Path | None, requirement: str, *, primary_failure: bool) -> None:
    """Remove a transfer's temporary archive, if one was ever created.

    A cleanup failure is reported only when nothing else already failed: the
    transfer's own error is what the caller needs to see.
    """
    if archive is None:
        return
    try:
        archive.unlink(missing_ok=True)
    except OSError as exc:
        if not primary_failure:
            raise FetchError(f"fetch failed for {requirement}: cleanup failed: {exc}") from exc


@dataclass(slots=True)
class _StallDeadline:
    """The inactivity budget of one archive transfer.

    The deadline bounds how long the transfer may go without delivering data,
    not how long it may run in total, so an archive of any permitted size can
    be downloaded as long as it keeps arriving.
    """

    requirement: str
    budget: float
    deadline: float = 0.0

    def __post_init__(self) -> None:
        self.deadline = time.monotonic() + self.budget

    def check(self) -> None:
        """Raise once no data has arrived for a whole budget."""

        if time.monotonic() >= self.deadline:
            raise FetchError(f"fetch timed out for {self.requirement}")

    def progressed(self) -> None:
        """Restart the budget, and its alarm, after data actually arrived."""

        self.deadline = time.monotonic() + self.budget
        signal.setitimer(signal.ITIMER_REAL, self.budget)


@contextmanager
def _wall_clock_deadline(stall: _StallDeadline) -> Iterator[None]:
    """Interrupt a transport operation blocked past its inactivity deadline."""

    started = time.monotonic()
    remaining = stall.deadline - started
    if remaining <= 0:
        raise FetchError(f"fetch timed out for {stall.requirement}")

    def raise_timeout(_signum: int, _frame: object) -> None:
        raise FetchError(f"fetch timed out for {stall.requirement}")

    previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
    previous_timer = (0.0, 0.0)
    try:
        previous_timer = signal.setitimer(signal.ITIMER_REAL, remaining)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        previous_remaining, previous_interval = previous_timer
        if previous_remaining > 0:
            elapsed = time.monotonic() - started
            restored_remaining = max(previous_remaining - elapsed, 1e-6)
            signal.setitimer(signal.ITIMER_REAL, restored_remaining, previous_interval)


def _expected_digest(requirement: str, expected_hash: str) -> str:
    """Extract a SHA-256 hex digest from a manifest hash declaration."""

    digest = parse_sha256(expected_hash)
    if digest is None:
        raise FetchError(f"hash mismatch for {requirement}")
    return digest
