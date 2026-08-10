"""Streaming URL fetch boundary for package archives."""

from __future__ import annotations

import hashlib
import signal
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

import requests

_FETCH_TIMEOUT_SECONDS = 30.0
_CHUNK_SIZE = 1024 * 1024
MAX_ARCHIVE_DOWNLOAD_SIZE = 128 * 1024 * 1024
_SHA256_PREFIXES = ("sha256=", "sha256:", "sha256-")


class FetchError(ValueError):
    """Raised when a package archive cannot be fetched or verified."""


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
    extraction outside this seam lets M6 add the archive format without changing
    URL transport or its verification contract.
    """

    expected_digest = _expected_digest(requirement, expected_hash)
    client: Session = requests.Session() if session is None else session
    archive: Path | None = None
    primary_failure = False
    deadline = time.monotonic() + _FETCH_TIMEOUT_SECONDS
    try:
        try:
            with tempfile.NamedTemporaryFile(
                dir=scratch_dir, suffix=".agmpkg", delete=False
            ) as file:
                archive = Path(file.name)
                digest = hashlib.sha256()
                downloaded_size = 0
                try:
                    with _wall_clock_deadline(requirement, deadline):
                        with client.get(
                            url, stream=True, timeout=_FETCH_TIMEOUT_SECONDS
                        ) as response:
                            response.raise_for_status()
                            _check_timeout(requirement, deadline)
                            for chunk in response.iter_content(_CHUNK_SIZE):
                                _check_timeout(requirement, deadline)
                                if chunk:
                                    downloaded_size += len(chunk)
                                    if downloaded_size > MAX_ARCHIVE_DOWNLOAD_SIZE:
                                        raise FetchError(
                                            f"fetch failed for {requirement}: "
                                            "package archive exceeds the download size limit"
                                        )
                                    file.write(chunk)
                                    digest.update(chunk)
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
        if archive is not None:
            try:
                archive.unlink(missing_ok=True)
            except OSError as exc:
                if not primary_failure:
                    error = FetchError(f"fetch failed for {requirement}: cleanup failed: {exc}")
                    raise error from exc


@contextmanager
def _wall_clock_deadline(requirement: str, deadline: float) -> Iterator[None]:
    """Interrupt a blocked transport operation at its absolute deadline."""

    started = time.monotonic()
    remaining = deadline - started
    if remaining <= 0:
        raise FetchError(f"fetch timed out for {requirement}")

    def raise_timeout(_signum: int, _frame: object) -> None:
        raise FetchError(f"fetch timed out for {requirement}")

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


def _check_timeout(requirement: str, deadline: float) -> None:
    """Raise when the archive transfer has exceeded its total deadline."""

    if time.monotonic() >= deadline:
        raise FetchError(f"fetch timed out for {requirement}")


def _expected_digest(requirement: str, expected_hash: str) -> str:
    """Extract a SHA-256 hex digest from a manifest hash declaration."""

    prefix = next(
        (candidate for candidate in _SHA256_PREFIXES if expected_hash.startswith(candidate)), None
    )
    if prefix is None:
        raise FetchError(f"hash mismatch for {requirement}")
    digest = expected_hash[len(prefix) :]
    if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        raise FetchError(f"hash mismatch for {requirement}")
    return digest.lower()
