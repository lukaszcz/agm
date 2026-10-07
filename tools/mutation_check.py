"""Optional semantic mutation gate; imports are changed in memory, never on disk."""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import Field, asdict, dataclass, field, fields
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import CodeType, ModuleType
from typing import Literal, TypedDict, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("mutations.json")
MARKER = "_agm_mutation_marker"


@dataclass(frozen=True)
class Mutation:
    id: str
    path: str
    module: str
    function: str
    line: int
    column: int
    source_sha256: str
    before: str
    after: str
    mode: Literal["eval", "exec", "keyword"]
    reason: str
    tests: list[str]


class Failure(TypedDict):
    nodeid: str
    phase: str
    exception: str | None
    detail: str


@dataclass
class Evidence:
    injected: bool = False
    reached: bool = False
    reached_tests: list[str] = field(default_factory=list)
    calls: int = 0
    failures: list[Failure] = field(default_factory=list)
    collection_errors: list[str] = field(default_factory=list)
    exitstatus: int | None = None


class Result(TypedDict):
    id: str
    status: str
    reason: str
    returncode: int | None
    command: list[str]
    log: str
    evidence: Evidence | None


def read_json(path: Path) -> object:
    return cast(object, json.loads(path.read_text()))


def evidence_dict(evidence: Evidence) -> dict[str, object]:
    return cast(dict[str, object], asdict(evidence))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=evidence_dict) + "\n")


def decode_record[T](record: Callable[..., T], value: object) -> T:
    """Read declared dataclass fields, ignoring extra manifest metadata."""
    data = cast(dict[str, object], value)
    record_fields = cast(tuple[Field[object], ...], fields(cast(type, record)))
    return record(**{item.name: data[item.name] for item in record_fields})


def read_manifest() -> list[Mutation]:
    return [decode_record(Mutation, value) for value in cast(list[object], read_json(MANIFEST))]


def snippet(text: str, mode: str) -> ast.AST:
    if mode == "keyword":
        call = ast.parse(f"f({text})", mode="eval").body
        assert isinstance(call, ast.Call)
        return call.keywords[0]
    if mode == "eval":
        return ast.parse(text, mode="eval").body
    return ast.parse(text).body[0]


def mutated_tree(spec: Mutation) -> ast.Module:
    """Check the pinned source and exact AST site, then compile-check its replacement."""
    path = ROOT / spec.path
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != spec.source_sha256:
        raise ValueError(f"{spec.id}: source hash mismatch: {spec.path}")
    tree = ast.parse(source, filename=str(path))
    wanted = ast.dump(snippet(spec.before, spec.mode))
    matches = [
        node
        for node in ast.walk(tree)
        if cast(object, getattr(node, "lineno", None)) == spec.line
        and cast(object, getattr(node, "col_offset", None)) == spec.column
        and ast.dump(node) == wanted
    ]
    if len(matches) != 1:
        raise ValueError(f"{spec.id}: AST site mismatch")
    site = matches[0]
    replacement = snippet(spec.after, spec.mode)
    marker = ast.Call(func=ast.Name(id=MARKER, ctx=ast.Load()), args=[], keywords=[])

    def marked_value(value: ast.expr) -> ast.expr:
        # Unlike a lambda, this preserves walrus bindings and zero-argument super().
        return ast.Subscript(
            value=ast.Tuple(elts=[marker, value], ctx=ast.Load()),
            slice=ast.Constant(value=1),
            ctx=ast.Load(),
        )

    if isinstance(replacement, ast.keyword):
        replacement.value = marked_value(replacement.value)
    elif isinstance(replacement, ast.expr):
        replacement = marked_value(replacement)
    else:
        assert isinstance(replacement, ast.stmt)
        replacement = ast.If(
            test=ast.Constant(value=True),
            body=[ast.Expr(value=marker), replacement],
            orelse=[],
        )
    # Attribute wrapper and snippet tracebacks to the original site.
    for node in ast.walk(replacement):
        ast.copy_location(node, site)

    class Replace(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> ast.AST:
            return replacement if node is site else cast(ast.AST, super().visit(node))

    mutated = ast.fix_missing_locations(cast(ast.Module, Replace().visit(tree)))
    compile(mutated, str(path), "exec")
    return mutated


STATE = Evidence()
EXCEPTIONS: dict[tuple[str, str], str | None] = {}
CURRENT_TEST: str | None = None


def mark_reached() -> None:
    """Record evaluation, attributing it only to the currently running test."""
    STATE.reached = True
    if CURRENT_TEST is not None and CURRENT_TEST not in STATE.reached_tests:
        STATE.reached_tests.append(CURRENT_TEST)


def pytest_runtest_logstart(nodeid: str) -> None:
    global CURRENT_TEST
    CURRENT_TEST = nodeid


def pytest_runtest_logfinish(nodeid: str) -> None:
    global CURRENT_TEST
    if CURRENT_TEST == nodeid:
        CURRENT_TEST = None


def install_mutation(spec: Mutation, patch: pytest.MonkeyPatch) -> None:
    if spec.module in sys.modules:
        raise ValueError("Mutation worker loaded after target import")
    filename = str(ROOT / spec.path)
    original_get_code = SourceFileLoader.get_code

    def get_code(loader: SourceFileLoader, fullname: str) -> CodeType | None:
        if fullname != spec.module:
            return original_get_code(loader, fullname)
        code = compile(mutated_tree(spec), filename, "exec")
        STATE.injected = True
        return code

    patch.setattr(builtins, MARKER, mark_reached, raising=False)
    patch.setattr(SourceFileLoader, "get_code", get_code)


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.failed:
        STATE.collection_errors.append(str(report.longrepr))


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> None:
    EXCEPTIONS[item.nodeid, call.when] = call.excinfo.typename if call.excinfo else None


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "call":
        STATE.calls += 1
    if report.failed:
        STATE.failures.append(
            {
                "nodeid": report.nodeid,
                "phase": report.when,
                "exception": EXCEPTIONS.get((report.nodeid, report.when)),
                "detail": str(report.longrepr),
            }
        )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    STATE.exitstatus = int(exitstatus)
    write_json(Path(os.environ["AGM_MUTATION_REPORT"]), STATE)


def classify(rc: int | None, evidence: Evidence | None, mutated: bool) -> tuple[str, str]:
    if rc is None:
        return "invalid", "timeout"
    if evidence is None:
        return "invalid", "missing plugin evidence"
    if evidence.collection_errors or any(
        failure["phase"] != "call" for failure in evidence.failures
    ):
        return "invalid", "collection/setup/teardown failure"
    if any(
        failure["exception"] in ("ImportError", "ModuleNotFoundError")
        for failure in evidence.failures
    ):
        return "invalid", "import failure"
    if rc not in (0, 1) or evidence.exitstatus != rc or not evidence.calls:
        return "invalid", "pytest infrastructure failure or no test calls"
    if mutated and not evidence.injected:
        return "invalid", "target was not imported"
    if evidence.failures:
        if not mutated:
            return "invalid", "unmutated baseline failed"
        reached_failures = [
            failure for failure in evidence.failures if failure["nodeid"] in evidence.reached_tests
        ]
        if not reached_failures:
            return "invalid", "failing test did not evaluate the mutation site"
        exception = reached_failures[0]["exception"]
        return "killed", (
            "assertion"
            if exception in ("AssertionError", "Failed")
            else "runtime test-call exception"
        )
    if rc != 0:
        return "invalid", "pytest failed without test-call evidence"
    if mutated and not evidence.reached_tests:
        return "unreached", "mutation site not evaluated by selected tests"
    return ("survived" if mutated else "passed"), "all selected tests passed"


def run_case(
    ident: str, tests: list[str], report: Path, timeout: float, spec: Mutation | None = None
) -> Result:
    raw_path = report / f"{ident}.raw.json"
    raw_path.unlink(missing_ok=True)
    log_path = report / f"{ident}.log"
    env = dict(os.environ)
    env.pop("AGM_MUTATION_ID", None)
    env["AGM_MUTATION_REPORT"] = str(raw_path)
    env["PYTHONPATH"] = os.pathsep.join((str(MANIFEST.parent), str(ROOT / "src"), str(ROOT)))
    if spec:
        env["AGM_MUTATION_ID"] = spec.id
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "-x",
        *tests,
    ]
    with log_path.open("w") as log:
        try:
            rc: int | None = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            ).returncode
        except subprocess.TimeoutExpired:
            rc = None
    evidence = decode_record(Evidence, read_json(raw_path)) if raw_path.exists() else None
    status, reason = classify(rc, evidence, spec is not None)
    print(f"{ident}: {status} ({reason})", flush=True)
    return {
        "id": ident,
        "status": status,
        "reason": reason,
        "returncode": rc,
        "command": command,
        "log": log_path.name,
        "evidence": evidence,
    }


def worker(args: list[str]) -> int:
    """Install the mutation before pytest imports conftest or AGM."""
    # Pytest fixture discovery requires a hashable plugin object.
    plugin = ModuleType("mutation_evidence")
    cast(dict[str, object], plugin.__dict__).update(
        pytest_collectreport=pytest_collectreport,
        pytest_runtest_logstart=pytest_runtest_logstart,
        pytest_runtest_logfinish=pytest_runtest_logfinish,
        pytest_runtest_makereport=pytest_runtest_makereport,
        pytest_runtest_logreport=pytest_runtest_logreport,
        pytest_sessionfinish=pytest_sessionfinish,
    )
    with pytest.MonkeyPatch.context() as patch:
        mutant_id = os.environ.get("AGM_MUTATION_ID")
        if mutant_id:
            install_mutation(next(spec for spec in read_manifest() if spec.id == mutant_id), patch)
        return int(pytest.main(args, plugins=[plugin]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", help="Mutation IDs (default: all)")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90, help="Seconds per pytest process")
    parser.add_argument("--report", type=Path, default=Path("mutation-report"))
    args = parser.parse_args()
    ids = cast(list[str] | None, args.ids)
    jobs = cast(int, args.jobs)
    timeout = cast(float, args.timeout)
    report = cast(Path, args.report).resolve()
    if jobs < 1 or timeout <= 0:
        parser.error("jobs and timeout must be positive")
    manifest = read_manifest()
    unknown = set(ids or ()) - {spec.id for spec in manifest}
    if unknown:
        parser.error(f"Unknown mutation IDs: {', '.join(sorted(unknown))}")
    selected = [spec for spec in manifest if ids is None or spec.id in ids]
    try:
        for spec in selected:
            mutated_tree(spec)
    except (ValueError, SyntaxError) as exc:
        parser.error(str(exc))
    report.mkdir(parents=True, exist_ok=True)
    # Each distinct test selection must pass unchanged before any mutant runs.
    suites = sorted({tuple(spec.tests) for spec in selected})
    baselines = [
        run_case(f"baseline-{index}", list(tests), report, timeout)
        for index, tests in enumerate(suites, 1)
    ]
    results: list[Result] = []
    if all(result["status"] == "passed" for result in baselines):

        def replay(spec: Mutation) -> Result:
            return run_case(spec.id, spec.tests, report, timeout, spec)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(replay, selected))
    counts = Counter(result["status"] for result in results)
    write_json(
        report / "results.json",
        {"baselines": baselines, "mutations": results, "counts": dict(counts)},
    )
    print(dict(counts))
    return 0 if len(results) == len(selected) and counts["killed"] == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(worker(sys.argv[2:]) if sys.argv[1:2] == ["--worker"] else main())
