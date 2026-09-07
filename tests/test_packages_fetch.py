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


class _Clock:
    """A deterministic monotonic clock advanced by the fake transport."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _PacedResponse(_Response):
    """Delivers each chunk after advancing the injected clock by its gap."""

    def __init__(self, clock: _Clock, paced_chunks: list[tuple[float, bytes]]) -> None:
        super().__init__([chunk for _, chunk in paced_chunks])
        self.clock = clock
        self.paced_chunks = paced_chunks

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        assert chunk_size > 0
        for gap, chunk in self.paced_chunks:
            self.clock.advance(gap)
            yield chunk


def _use_clock(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    monkeypatch.setattr(package_fetch, "time", SimpleNamespace(monotonic=clock.monotonic))


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


def test_fetch_stops_a_response_that_stalls_before_its_first_chunk(
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


def test_fetch_completes_a_steady_transfer_longer_than_the_stall_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A healthy download is bounded by inactivity, not by total elapsed time."""

    monkeypatch.setattr(package_fetch, "_FETCH_TIMEOUT_SECONDS", 5.0)
    clock = _Clock()
    _use_clock(monkeypatch, clock)
    paced_chunks = [(2.0, b"chunk%d;" % index) for index in range(10)]
    content = b"".join(chunk for _, chunk in paced_chunks)
    handed_off: list[bytes] = []

    fetch_archive(
        requirement="tools >= 1.0.0",
        url="https://example.test/tools.agmpkg",
        expected_hash="sha256=" + hashlib.sha256(content).hexdigest(),
        handoff=lambda path: handed_off.append(path.read_bytes()),
        session=_Session(_PacedResponse(clock, paced_chunks)),
        scratch_dir=tmp_path,
    )

    assert clock.now > 5.0
    assert handed_off == [content]
    assert not tuple(tmp_path.iterdir())


def test_fetch_stops_a_transfer_that_stalls_mid_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(package_fetch, "_FETCH_TIMEOUT_SECONDS", 5.0)
    clock = _Clock()
    _use_clock(monkeypatch, clock)
    paced_chunks = [(2.0, b"first"), (2.0, b"second"), (9.0, b"stalled")]
    content = b"".join(chunk for _, chunk in paced_chunks)

    with pytest.raises(FetchError, match=r"fetch timed out.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + hashlib.sha256(content).hexdigest(),
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            session=_Session(_PacedResponse(clock, paced_chunks)),
            scratch_dir=tmp_path,
        )

    assert not tuple(tmp_path.iterdir())


def test_fetch_size_limit_bounds_a_steady_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The size limit remains the backstop once inactivity replaces a total budget."""

    monkeypatch.setattr(package_fetch, "_FETCH_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(package_fetch, "MAX_ARCHIVE_DOWNLOAD_SIZE", 8)
    clock = _Clock()
    _use_clock(monkeypatch, clock)
    paced_chunks = [(1.0, b"chunk"), (1.0, b"chunk"), (1.0, b"chunk")]
    content = b"".join(chunk for _, chunk in paced_chunks)

    with pytest.raises(FetchError, match=r"fetch failed.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + hashlib.sha256(content).hexdigest(),
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            session=_Session(_PacedResponse(clock, paced_chunks)),
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

    assert "tools >= 1.0.0" in str(raised.value)
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

    assert "tools >= 1.0.0" in str(raised.value)
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

    assert "tools >= 1.0.0" in str(raised.value)
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

    assert "tools >= 1.0.0" in str(raised.value)
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


@pytest.mark.parametrize(
    "content", [b"", b"abc", b"abcd"], ids=["empty", "below-limit", "at-limit"]
)
def test_fetch_accepts_bytes_up_to_the_exact_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    monkeypatch.setattr(package_fetch, "MAX_ARCHIVE_DOWNLOAD_SIZE", 4)
    received: list[bytes] = []
    fetch_archive(
        requirement="tools >= 1.0.0",
        url="https://example.test/tools.agmpkg",
        expected_hash="sha256=" + hashlib.sha256(content).hexdigest(),
        handoff=lambda path: received.append(path.read_bytes()),
        session=_Session(_Response([content[:2], b"", content[2:]])),
        scratch_dir=tmp_path,
    )
    assert received == [content]
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("chunks", [[b"abcde"], [b"ab", b"", b"cde"]], ids=["one-chunk", "split"])
def test_fetch_rejects_one_byte_over_the_limit_before_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunks: list[bytes]
) -> None:
    monkeypatch.setattr(package_fetch, "MAX_ARCHIVE_DOWNLOAD_SIZE", 4)
    with pytest.raises(FetchError):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + hashlib.sha256(b"".join(chunks)).hexdigest(),
            handoff=lambda _: pytest.fail("oversized content reached the installer"),
            session=_Session(_Response(chunks)),
            scratch_dir=tmp_path,
        )
    assert not tuple(tmp_path.iterdir())


def test_empty_chunks_do_not_extend_the_inactivity_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(package_fetch, "_FETCH_TIMEOUT_SECONDS", 5.0)
    clock = _Clock()
    _use_clock(monkeypatch, clock)
    with pytest.raises(FetchError):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + hashlib.sha256(b"data").hexdigest(),
            handoff=lambda _: pytest.fail("stalled content reached the installer"),
            session=_Session(_PacedResponse(clock, [(2, b""), (2, b""), (2, b"data")])),
            scratch_dir=tmp_path,
        )
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("error", [ValueError("invalid archive"), KeyboardInterrupt()])
def test_installer_failure_removes_the_download_and_preserves_the_original_error(
    tmp_path: Path, error: BaseException
) -> None:
    content = b"archive"
    received: list[bytes] = []

    def install(path: Path) -> None:
        received.append(path.read_bytes())
        raise error

    with pytest.raises(type(error)) as raised:
        fetch_archive(
            requirement="tools >= 1.0.0",
            url="https://example.test/tools.agmpkg",
            expected_hash="sha256=" + hashlib.sha256(content).hexdigest(),
            handoff=install,
            session=_Session(_Response([content])),
            scratch_dir=tmp_path,
        )

    assert raised.value is error
    assert received == [content]
    assert not tuple(tmp_path.iterdir())
