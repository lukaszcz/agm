"""Tests for the package URL-fetch boundary; all transport is mocked."""

from __future__ import annotations

import hashlib
import signal
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import agm.packages.fetch as package_fetch
from agm.packages.fetch import FetchError, fetch_archive


class _Response:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        error: Exception | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.stream_error = stream_error
        self.status_checked = False

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def raise_for_status(self) -> None:
        self.status_checked = True
        if self.error is not None:
            raise self.error

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        assert chunk_size > 0
        yield from self.chunks
        if self.stream_error is not None:
            raise self.stream_error


class _BlockingResponse(_Response):
    def __init__(self) -> None:
        super().__init__([])
        self.exited = False

    def __exit__(self, *_: object) -> None:
        self.exited = True

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        del chunk_size
        time.sleep(0.2)
        raise AssertionError("the fetch deadline did not interrupt the blocked body")
        yield b""


class _Session:
    def __init__(
        self, response: _Response | None = None, *, get_error: Exception | None = None
    ) -> None:
        self.response = response
        self.get_error = get_error
        self.calls: list[tuple[str, bool, float]] = []

    def get(self, url: str, *, stream: bool, timeout: float) -> _Response:
        self.calls.append((url, stream, timeout))
        if self.get_error is not None:
            raise self.get_error
        assert self.response is not None
        return self.response


def test_fetch_streams_with_explicit_timeout_hashes_and_hands_off_archive(tmp_path: Path) -> None:
    content = b"archive bytes"
    response = _Response([content[:4], b"", content[4:]])
    session = _Session(response)
    handed_off: list[bytes] = []

    fetch_archive(
        requirement="tools >= 1.0",
        url="https://example.test/tools.agmpkg",
        expected_hash="sha256=" + hashlib.sha256(content).hexdigest(),
        handoff=lambda path: handed_off.append(path.read_bytes()),
        session=session,
        scratch_dir=tmp_path,
    )

    assert session.calls == [("https://example.test/tools.agmpkg", True, 30.0)]
    assert response.status_checked
    assert handed_off == [content]
    assert not tuple(tmp_path.iterdir())


def test_fetch_preserves_an_existing_process_alarm(tmp_path: Path) -> None:
    content = b"archive bytes"
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def existing_handler(_signum: int, _frame: object) -> None:
        return None

    signal.signal(signal.SIGALRM, existing_handler)
    signal.setitimer(signal.ITIMER_REAL, 10.0)
    try:
        fetch_archive(
            requirement="tools >= 1.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + hashlib.sha256(content).hexdigest(),
            handoff=lambda _: None,
            session=_Session(_Response([content])),
            scratch_dir=tmp_path,
        )

        restored_remaining, restored_interval = signal.getitimer(signal.ITIMER_REAL)
        assert signal.getsignal(signal.SIGALRM) is existing_handler
        assert 0 < restored_remaining <= 10.0
        assert restored_interval == 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def test_fetch_deadline_interrupts_a_blocked_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(package_fetch, "_FETCH_TIMEOUT_SECONDS", 0.01)

    class BlockingSession(_Session):
        def get(self, url: str, *, stream: bool, timeout: float) -> _Response:
            del url, stream, timeout
            time.sleep(0.2)
            raise AssertionError("the fetch deadline did not interrupt the blocked connect")

    with pytest.raises(FetchError, match=r"fetch timed out.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            session=BlockingSession(),
            scratch_dir=tmp_path,
        )

    assert not tuple(tmp_path.iterdir())


def test_fetch_deadline_interrupts_a_blocked_body_and_closes_the_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(package_fetch, "_FETCH_TIMEOUT_SECONDS", 0.01)
    response = _BlockingResponse()

    with pytest.raises(FetchError, match=r"fetch timed out.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            session=_Session(response),
            scratch_dir=tmp_path,
        )

    assert response.exited
    assert not tuple(tmp_path.iterdir())


def test_fetch_times_out_if_the_deadline_expires_before_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(package_fetch, "_FETCH_TIMEOUT_SECONDS", 1.0)
    monotonic_values = iter((0.0, 1.0))
    monkeypatch.setattr(
        package_fetch,
        "time",
        SimpleNamespace(monotonic=monotonic_values.__next__),
        raising=False,
    )
    session = _Session(_Response([b"unused"]))

    with pytest.raises(FetchError, match=r"fetch timed out.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            session=session,
            scratch_dir=tmp_path,
        )

    assert session.calls == []
    assert not tuple(tmp_path.iterdir())


def test_fetch_stops_a_trickling_response_at_the_total_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"archive bytes"
    monkeypatch.setattr(package_fetch, "_FETCH_TIMEOUT_SECONDS", 1.0)
    monotonic_values = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(
        package_fetch,
        "time",
        SimpleNamespace(monotonic=monotonic_values.__next__),
        raising=False,
    )

    with pytest.raises(FetchError, match=r"fetch timed out.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + hashlib.sha256(content).hexdigest(),
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            session=_Session(_Response([content])),
            scratch_dir=tmp_path,
        )

    assert not tuple(tmp_path.iterdir())


def test_fetch_reports_hash_mismatch_and_never_hands_off(tmp_path: Path) -> None:
    session = _Session(_Response([b"wrong bytes"]))

    with pytest.raises(FetchError, match=r"hash mismatch.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            session=session,
            scratch_dir=tmp_path,
        )

    assert not tuple(tmp_path.iterdir())


def test_fetch_wraps_transport_failures_with_the_requirement(tmp_path: Path) -> None:
    session = _Session(_Response([], error=OSError("offline")))

    with pytest.raises(FetchError, match=r"fetch failed.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: None,
            session=session,
            scratch_dir=tmp_path,
        )


def test_fetch_wraps_get_connection_failure_with_normalized_requirement(tmp_path: Path) -> None:
    error = requests.ConnectionError("offline")
    session = _Session(get_error=error)

    with pytest.raises(FetchError) as raised:
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            session=session,
            scratch_dir=tmp_path,
        )

    assert str(raised.value) == "fetch failed for tools >= 1.0.0: offline"
    assert raised.value.__cause__ is error
    assert session.calls == [("https://example.test/tools.agmpkg", True, 30.0)]
    assert not tuple(tmp_path.iterdir())


def test_fetch_wraps_stream_failure_with_normalized_requirement(tmp_path: Path) -> None:
    error = requests.ConnectionError("connection interrupted")
    response = _Response([b"partial archive"], stream_error=error)

    with pytest.raises(FetchError) as raised:
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            session=_Session(response),
            scratch_dir=tmp_path,
        )

    assert str(raised.value) == "fetch failed for tools >= 1.0.0: connection interrupted"
    assert raised.value.__cause__ is error
    assert response.status_checked
    assert not tuple(tmp_path.iterdir())


def test_fetch_preserves_primary_failure_when_archive_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport_error = requests.ConnectionError("offline")
    cleanup_error = OSError("cleanup denied")
    original_unlink = Path.unlink

    def fail_archive_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        if path.suffix == ".agmpkg":
            raise cleanup_error
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_archive_cleanup)

    with pytest.raises(FetchError) as raised:
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            session=_Session(get_error=transport_error),
            scratch_dir=tmp_path,
        )

    assert str(raised.value) == "fetch failed for tools >= 1.0.0: offline"
    assert raised.value.__cause__ is transport_error


def test_fetch_wraps_archive_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"archive bytes"
    cleanup_error = OSError("cleanup denied")
    original_unlink = Path.unlink

    def fail_archive_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        if path.suffix == ".agmpkg":
            raise cleanup_error
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_archive_cleanup)

    with pytest.raises(FetchError, match=r"fetch failed.*tools >= 1\.0\.0") as raised:
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + hashlib.sha256(content).hexdigest(),
            handoff=lambda _: None,
            session=_Session(_Response([content])),
            scratch_dir=tmp_path,
        )

    assert str(raised.value) == "fetch failed for tools >= 1.0.0: cleanup failed: cleanup denied"
    assert raised.value.__cause__ is cleanup_error


def test_fetch_reports_scratch_creation_failure(tmp_path: Path) -> None:
    content = b"archive bytes"
    session = _Session(_Response([content]))

    with pytest.raises(FetchError, match=r"fetch failed.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + hashlib.sha256(content).hexdigest(),
            handoff=lambda _: None,
            session=session,
            scratch_dir=tmp_path / "missing",
        )


def test_fetch_rejects_an_archive_larger_than_the_download_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(package_fetch, "MAX_ARCHIVE_DOWNLOAD_SIZE", 4)
    content = b"oversized"

    with pytest.raises(FetchError, match=r"fetch failed.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + hashlib.sha256(content).hexdigest(),
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            session=_Session(_Response([content])),
            scratch_dir=tmp_path,
        )

    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("expected_hash", ["md5=bad", "sha256=short"])
def test_fetch_rejects_non_sha256_hash_before_transport(tmp_path: Path, expected_hash: str) -> None:
    session = _Session(_Response([b"unused"]))

    with pytest.raises(FetchError, match=r"hash mismatch.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash=expected_hash,
            handoff=lambda _: None,
            session=session,
            scratch_dir=tmp_path,
        )

    assert session.calls == []
