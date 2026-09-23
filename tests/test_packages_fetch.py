"""Tests for the package URL-fetch boundary; all transport is mocked."""

from __future__ import annotations

import hashlib
import signal
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
import requests
import requests.adapters

import agm.packages.fetch as package_fetch
from agm.core import http as core_http
from agm.packages.fetch import FetchError, fetch_archive
from tests._http_helpers import install

_URL = "https://example.test/tools.agmpkg"


def _sha256(content: bytes) -> str:
    return "sha256=" + hashlib.sha256(content).hexdigest()


def _body(content: bytes) -> dict[str, object]:
    """One scripted outcome delivering *content* as the archive body."""
    return {"status": 200, "body_hex": content.hex()}


def _mount(monkeypatch: pytest.MonkeyPatch, adapter: requests.adapters.BaseAdapter) -> None:
    """Answer every fetch through *adapter* instead of the network."""
    session = core_http.open_session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    monkeypatch.setattr("agm.core.http.open_session", lambda: session)


class _BlockedAdapter(requests.adapters.BaseAdapter):
    """A transport that blocks far past any fetch budget."""

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: object = None,
        verify: object = True,
        cert: object = None,
        proxies: dict[str, str] | None = None,
    ) -> requests.Response:
        del request, stream, timeout, verify, cert, proxies
        time.sleep(30.0)
        raise AssertionError("the stall alarm did not interrupt the blocked transport")

    def close(self) -> None:
        pass


class _PacedRaw:
    """A urllib3-response surface that hands out *chunks*, pausing before each."""

    def __init__(self, chunks: Sequence[bytes], gap: float) -> None:
        self._chunks = chunks
        self._gap = gap

    def stream(self, chunk_size: int, decode_content: bool = True) -> Iterator[bytes]:
        del chunk_size, decode_content
        for chunk in self._chunks:
            time.sleep(self._gap)
            yield chunk

    def close(self) -> None:
        pass

    def release_conn(self) -> None:
        pass


class _PacedAdapter(requests.adapters.BaseAdapter):
    """A transport whose body arrives in steady chunks spread over real time."""

    def __init__(self, chunks: Sequence[bytes], gap: float) -> None:
        super().__init__()
        self._chunks = chunks
        self._gap = gap

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: object = None,
        verify: object = True,
        cert: object = None,
        proxies: dict[str, str] | None = None,
    ) -> requests.Response:
        del stream, timeout, verify, cert, proxies
        response = requests.Response()
        response.status_code = 200
        response.url = request.url or ""
        response.request = request
        response.raw = _PacedRaw(self._chunks, self._gap)
        return response

    def close(self) -> None:
        pass


def test_fetch_streams_hashes_and_hands_off_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"archive bytes"
    adapter = install(monkeypatch, [_body(content)])
    handed_off: list[bytes] = []

    fetch_archive(
        requirement="tools >= 1.0",
        url=_URL,
        expected_hash=_sha256(content),
        handoff=lambda path: handed_off.append(path.read_bytes()),
        scratch_dir=tmp_path,
    )

    assert handed_off == [content]
    assert adapter.sent[0].method == "GET"
    assert adapter.sent[0].url == _URL
    assert adapter.streams == [True]
    assert adapter.timeouts == [(30.0, 30.0)]
    assert not tuple(tmp_path.iterdir())
    adapter.assert_complete()


def test_fetch_reports_hash_mismatch_and_never_hands_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, [_body(b"wrong bytes")])

    with pytest.raises(FetchError, match=r"hash mismatch.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url=_URL,
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            scratch_dir=tmp_path,
        )

    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("expected_hash", ["md5=bad", "sha256=short"])
def test_fetch_rejects_a_non_sha256_hash_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expected_hash: str
) -> None:
    adapter = install(monkeypatch, [])

    with pytest.raises(FetchError, match=r"hash mismatch.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url=_URL,
            expected_hash=expected_hash,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            scratch_dir=tmp_path,
        )

    assert adapter.sent == []


def test_fetch_reports_an_error_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, [{"status": 404, "body": "no such archive"}])

    with pytest.raises(FetchError, match=r"fetch failed.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url=_URL,
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            scratch_dir=tmp_path,
        )

    assert not tuple(tmp_path.iterdir())


def test_fetch_wraps_a_connect_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, [{"fail": "connection"}])

    with pytest.raises(FetchError, match=r"fetch failed.*tools >= 1\.0\.0") as raised:
        fetch_archive(
            requirement="tools >= 1.0.0",
            url=_URL,
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            scratch_dir=tmp_path,
        )

    assert isinstance(raised.value.__cause__, core_http.TransportError)
    assert not tuple(tmp_path.iterdir())


def test_fetch_reports_a_transport_timeout_as_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, [{"fail": "timeout"}])

    with pytest.raises(FetchError, match=r"fetch timed out.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url=_URL,
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            scratch_dir=tmp_path,
        )

    assert not tuple(tmp_path.iterdir())


def test_fetch_wraps_a_mid_stream_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(
        monkeypatch,
        [{"status": 200, "body_hex": b"partial".hex(), "fail_mid_stream": "connection"}],
    )

    with pytest.raises(FetchError, match=r"fetch failed.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url=_URL,
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            scratch_dir=tmp_path,
        )

    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize(
    "content", [b"", b"abc", b"abcd"], ids=["empty", "below-limit", "at-limit"]
)
def test_fetch_accepts_bytes_up_to_the_exact_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    monkeypatch.setattr(package_fetch, "MAX_ARCHIVE_DOWNLOAD_SIZE", 4)
    install(monkeypatch, [_body(content)])
    received: list[bytes] = []

    fetch_archive(
        requirement="tools >= 1.0.0",
        url=_URL,
        expected_hash=_sha256(content),
        handoff=lambda path: received.append(path.read_bytes()),
        scratch_dir=tmp_path,
    )

    assert received == [content]
    assert not tuple(tmp_path.iterdir())


def test_fetch_rejects_an_archive_larger_than_the_download_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(package_fetch, "MAX_ARCHIVE_DOWNLOAD_SIZE", 4)
    content = b"oversized"
    install(monkeypatch, [_body(content)])

    with pytest.raises(FetchError, match=r"fetch failed.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url=_URL,
            expected_hash=_sha256(content),
            handoff=lambda _: pytest.fail("oversized content reached the installer"),
            scratch_dir=tmp_path,
        )

    assert not tuple(tmp_path.iterdir())


def test_fetch_reports_a_scratch_directory_that_does_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"archive bytes"
    install(monkeypatch, [_body(content)])

    with pytest.raises(FetchError, match=r"fetch failed.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url=_URL,
            expected_hash=_sha256(content),
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            scratch_dir=tmp_path / "missing",
        )


def test_fetch_preserves_primary_failure_when_archive_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cleanup_error = OSError("cleanup denied")
    original_unlink = Path.unlink
    install(monkeypatch, [{"fail": "connection"}])

    def fail_archive_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        if path.suffix == ".agmpkg":
            raise cleanup_error
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_archive_cleanup)

    with pytest.raises(FetchError) as raised:
        fetch_archive(
            requirement="tools >= 1.0.0",
            url=_URL,
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            scratch_dir=tmp_path,
        )

    assert "tools >= 1.0.0" in str(raised.value)
    assert isinstance(raised.value.__cause__, core_http.TransportError)


def test_fetch_wraps_archive_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"archive bytes"
    cleanup_error = OSError("cleanup denied")
    original_unlink = Path.unlink
    install(monkeypatch, [_body(content)])

    def fail_archive_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        if path.suffix == ".agmpkg":
            raise cleanup_error
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_archive_cleanup)

    with pytest.raises(FetchError, match=r"fetch failed.*tools >= 1\.0\.0") as raised:
        fetch_archive(
            requirement="tools >= 1.0.0",
            url=_URL,
            expected_hash=_sha256(content),
            handoff=lambda _: None,
            scratch_dir=tmp_path,
        )

    assert raised.value.__cause__ is cleanup_error


@pytest.mark.parametrize("error", [ValueError("invalid archive"), KeyboardInterrupt()])
def test_installer_failure_removes_the_download_and_preserves_the_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    content = b"archive"
    install(monkeypatch, [_body(content)])
    received: list[bytes] = []

    def installer(path: Path) -> None:
        received.append(path.read_bytes())
        raise error

    with pytest.raises(type(error)) as raised:
        fetch_archive(
            requirement="tools >= 1.0.0",
            url=_URL,
            expected_hash=_sha256(content),
            handoff=installer,
            scratch_dir=tmp_path,
        )

    assert raised.value is error
    assert received == [content]
    assert not tuple(tmp_path.iterdir())


def test_fetch_preserves_an_existing_process_alarm(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fetch with no scratch directory of its own still leaves the process alarm intact."""
    content = b"archive bytes"
    install(monkeypatch, [_body(content)])
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def existing_handler(_signum: int, _frame: object) -> None:
        return None

    signal.signal(signal.SIGALRM, existing_handler)
    signal.setitimer(signal.ITIMER_REAL, 10.0)
    try:
        fetch_archive(
            requirement="tools >= 1.0",
            url=_URL,
            expected_hash=_sha256(content),
            handoff=lambda _: None,
        )

        restored_remaining, restored_interval = signal.getitimer(signal.ITIMER_REAL)
        assert signal.getsignal(signal.SIGALRM) is existing_handler
        assert 0 < restored_remaining <= 10.0
        assert restored_interval == 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def test_fetch_times_out_when_the_transport_blocks_past_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transport that blocks where no socket timeout reaches is still preempted."""
    monkeypatch.setattr(package_fetch, "_FETCH_TIMEOUT_SECONDS", 0.05)
    _mount(monkeypatch, _BlockedAdapter())

    with pytest.raises(FetchError, match=r"fetch timed out.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url=_URL,
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            scratch_dir=tmp_path,
        )

    assert not tuple(tmp_path.iterdir())


def test_a_body_trickling_slower_than_the_budget_still_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the archive starts arriving, only read inactivity and the size cap bound it.

    Every gap here is longer than the whole fetch budget, so a download that
    creeps along -- a congested link delivering a trickle -- runs to the end
    instead of being preempted.
    """
    chunks = [b"trickle%d;" % index for index in range(3)]
    content = b"".join(chunks)
    monkeypatch.setattr(package_fetch, "_FETCH_TIMEOUT_SECONDS", 0.05)
    _mount(monkeypatch, _PacedAdapter(chunks, 0.1))
    handed_off: list[bytes] = []
    started = time.monotonic()

    fetch_archive(
        requirement="tools >= 1.0.0",
        url=_URL,
        expected_hash=_sha256(content),
        handoff=lambda path: handed_off.append(path.read_bytes()),
        scratch_dir=tmp_path,
    )

    assert time.monotonic() - started > 0.05
    assert handed_off == [content]
    assert not tuple(tmp_path.iterdir())


def test_fetch_reports_a_body_that_stalls_mid_stream_as_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A body that stops arriving is preempted by the transport's own read timeout."""
    install(
        monkeypatch,
        [{"status": 200, "body_hex": b"partial".hex(), "fail_mid_stream": "timeout"}],
    )

    with pytest.raises(FetchError, match=r"fetch timed out.*tools >= 1\.0\.0"):
        fetch_archive(
            requirement="tools >= 1.0.0",
            url=_URL,
            expected_hash="sha256=" + "0" * 64,
            handoff=lambda _: pytest.fail("archive handoff must not run"),
            scratch_dir=tmp_path,
        )

    assert not tuple(tmp_path.iterdir())
