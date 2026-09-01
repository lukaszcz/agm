"""A REPL session reuses its library image instead of recompiling it per entry.

Every REPL entry compiles a whole program: the entry module plus the library
modules the session already loaded. Those library modules are the same objects
entry after entry, so their infix resolution, scope resolution, and compiled
match sites are reused rather than recomputed.

These tests pin both halves of that contract: that the reuse actually happens
(a later entry does no library work), and that it stops the moment the library
image is no longer the one the artifacts were derived from. They also pin that
reuse never bridges REPL supersession, where a redeclared type must keep its
old and new identities apart, and that no entry outcome differs between a
session that reuses its library image and one that reparses and recompiles it
for every entry.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agm.agl import matchcompile, syntax
from agm.agl.matchcompile import stage as match_stage
from agm.agl.modules import loader as loader_module
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.modules.loader import ModuleGraph
from agm.agl.modules.parsed_module_cache import RESERVED_NODE_ID_BASE, ModuleBuilder
from agm.agl.modules.roots import assemble_roots
from agm.agl.parser import parse_program_seeded
from agm.agl.repl import EntryResult, ReplSession
from agm.agl.scope import program as scope_program
from agm.agl.semantics.values import BoolValue, IntValue, TextValue
from agm.agl.syntax.spans import SourceId
from agm.agl.typecheck import program as typecheck_program
from agm.core import fs
from agm.util.text import normalize_newlines

_STDLIB = Path(__file__).resolve().parents[1] / "stdlib"


def _open_session() -> ReplSession:
    """Open a default session the way a REPL host does."""
    session = ReplSession()
    assert session.open() == ()
    return session


def _session_with_root(root: Path) -> ReplSession:
    """Open a session that can also import modules from *root*."""
    session = ReplSession()
    session._roots = assemble_roots(
        invocation_root=root,
        stdlib_root=_STDLIB,
        lib_root=None,
        configured=[],
        cli=[],
        cwd=root,
    )
    assert session.open() == ()
    return session


# ---------------------------------------------------------------------------
# Recorders for the work each pass performs
# ---------------------------------------------------------------------------


@pytest.fixture
def resolved_modules(monkeypatch: pytest.MonkeyPatch) -> list[ModuleId]:
    """Record the module ids whose bodies scope resolution actually resolves."""
    recorded: list[ModuleId] = []
    original = scope_program._Resolver

    def recording(**kwargs: object) -> object:
        module_id = kwargs["module_id"]
        assert isinstance(module_id, ModuleId)
        recorded.append(module_id)
        return original(**kwargs)

    monkeypatch.setattr(scope_program, "_Resolver", recording)
    return recorded


@pytest.fixture
def infix_resolutions(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record one entry per module whose raw infix chains the loader rewrites."""
    recorded: list[object] = []
    original = loader_module.resolve_infix_chains

    def recording(program: object, *args: object, **kwargs: object) -> object:
        recorded.append(program)
        return original(program, *args, **kwargs)

    monkeypatch.setattr(loader_module, "resolve_infix_chains", recording)
    return recorded


@pytest.fixture
def checked_modules(monkeypatch: pytest.MonkeyPatch) -> list[ModuleId]:
    """Record the module ids whose bodies type checking actually checks."""
    recorded: list[ModuleId] = []
    original = typecheck_program._check_prepared_module

    def recording(*args: object, **kwargs: object) -> object:
        module_id = kwargs["module_id"]
        assert isinstance(module_id, ModuleId)
        recorded.append(module_id)
        return original(*args, **kwargs)

    monkeypatch.setattr(typecheck_program, "_check_prepared_module", recording)
    return recorded


@pytest.fixture
def compiled_match_owners(monkeypatch: pytest.MonkeyPatch) -> list[ModuleId]:
    """Record the module ids whose match sites are compiled from source."""
    recorded: list[ModuleId] = []
    original = match_stage._compile_owner_sites

    def recording(owner: object) -> object:
        module_id = getattr(owner, "module_id", ENTRY_ID)
        assert isinstance(module_id, ModuleId)
        recorded.append(module_id)
        return original(owner)

    monkeypatch.setattr(match_stage, "_compile_owner_sites", recording)
    return recorded


def _library(module_ids: list[ModuleId]) -> list[ModuleId]:
    return [module_id for module_id in module_ids if not module_id.is_entry]


# ---------------------------------------------------------------------------
# The reuse fires
# ---------------------------------------------------------------------------


class TestLibraryReuse:
    def test_a_later_entry_resolves_only_its_own_module(
        self, resolved_modules: list[ModuleId]
    ) -> None:
        session = _open_session()
        assert session.eval_entry("let x = 1").ok
        resolved_modules.clear()

        assert session.eval_entry("x + 1").ok

        assert resolved_modules == [ENTRY_ID]

    def test_a_later_entry_checks_only_its_own_module(
        self, checked_modules: list[ModuleId]
    ) -> None:
        session = _open_session()
        assert session.eval_entry("let x = 1").ok
        checked_modules.clear()

        assert session.eval_entry("x + 1").ok

        assert checked_modules == [ENTRY_ID]

    def test_a_later_entry_compiles_only_its_own_match_sites(
        self, compiled_match_owners: list[ModuleId]
    ) -> None:
        session = _open_session()
        assert session.eval_entry("let x = 1").ok
        compiled_match_owners.clear()

        assert session.eval_entry("case 1 of\n  | 1 => 2\n  | _ => 3").ok

        assert _library(compiled_match_owners) == []

    def test_a_later_entry_rewrites_only_its_own_infix_chains(
        self, infix_resolutions: list[object]
    ) -> None:
        session = _open_session()
        assert session.eval_entry("let x = 1").ok
        infix_resolutions.clear()

        assert session.eval_entry("1 + 2 * 3").ok

        assert len(infix_resolutions) == 1

    def test_reused_library_resolutions_are_the_same_objects(self) -> None:
        session = _open_session()
        assert session.eval_entry("let x = 1").ok
        first = dict(session._retained_resolved_modules)
        assert first

        assert session.eval_entry("let y = 2").ok

        for module_id, resolved in first.items():
            assert session._retained_resolved_modules[module_id] is resolved

    def test_a_reset_session_reuses_the_library_it_reloads(
        self, resolved_modules: list[ModuleId]
    ) -> None:
        """A reset drops session state, not the artifacts describing the library.

        Reopening reloads the very same library modules, so the retained
        resolutions still describe them exactly and are reused.
        """
        session = _open_session()
        assert session.eval_entry("let x = 1").ok
        session.reset()
        resolved_modules.clear()

        assert session.eval_entry("let y = 2").ok

        assert _library(resolved_modules) == []


# ---------------------------------------------------------------------------
# The reuse stops when the library image changes
# ---------------------------------------------------------------------------


class TestLibraryInvalidation:
    def test_a_newly_imported_module_is_resolved_and_used(
        self, tmp_path: Path, resolved_modules: list[ModuleId]
    ) -> None:
        (tmp_path / "later.agl").write_text("def twice(n: int) -> int = n * 2\n")
        session = _session_with_root(tmp_path)
        assert session.eval_entry("let x = 1").ok
        resolved_modules.clear()

        result = session.eval_entry("import later::*\ntwice(21)")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(42)
        assert ModuleId.from_path("later") in resolved_modules

    def test_a_newly_imported_module_has_its_infix_chains_rewritten(
        self, tmp_path: Path, infix_resolutions: list[object]
    ) -> None:
        (tmp_path / "ops.agl").write_text(
            "infixl <+> at 6\n"
            "def <+>(a: int, b: int) -> int = a + b + 1\n"
            "def combined() -> int = 1 <+> 2 <+> 3\n"
        )
        session = _session_with_root(tmp_path)
        assert session.eval_entry("let x = 1").ok
        infix_resolutions.clear()

        result = session.eval_entry("import ops::*\ncombined()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(8)
        assert len(infix_resolutions) > 1

    def test_a_later_import_of_a_second_module_still_sees_the_first(self, tmp_path: Path) -> None:
        (tmp_path / "one.agl").write_text("def one() -> int = 1\n")
        (tmp_path / "two.agl").write_text("def two() -> int = 2\n")
        session = _session_with_root(tmp_path)
        assert session.eval_entry("import one::*\none()").ok

        result = session.eval_entry("import two::*\none() + two()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(3)

    def test_a_reopened_session_rejects_a_different_module_of_the_same_name(
        self, tmp_path: Path
    ) -> None:
        """A reopened session keeps its memo, but only for the modules it reloads."""
        first = tmp_path / "first"
        first.mkdir()
        second = tmp_path / "second"
        second.mkdir()
        body = "def pick(n: int) -> int =\n  case n of\n    | 0 => %s\n    | _ => %s\n"
        (first / "shared.agl").write_text(body % (1, 2))
        (second / "shared.agl").write_text(body % (30, 40))

        session = _session_with_root(first)
        assert session.eval_entry("import shared::*\npick(0)").value == IntValue(1)

        session.reset()
        session._roots = assemble_roots(
            invocation_root=second,
            stdlib_root=_STDLIB,
            lib_root=None,
            configured=[],
            cli=[],
            cwd=second,
        )
        result = session.eval_entry("import shared::*\npick(0)")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(30)

    def test_a_reset_session_rereads_a_module_whose_file_changed(self, tmp_path: Path) -> None:
        """Reuse survives a reset only for modules the reopened session reloads."""
        module = tmp_path / "shifting.agl"
        module.write_text("def answer() -> int = 1\n")
        session = _session_with_root(tmp_path)
        assert session.eval_entry("import shifting::*\nanswer()").value == IntValue(1)

        session.reset()
        module.write_text("def answer() -> int = 2\n")
        session._roots = assemble_roots(
            invocation_root=tmp_path,
            stdlib_root=_STDLIB,
            lib_root=None,
            configured=[],
            cli=[],
            cwd=tmp_path,
        )
        result = session.eval_entry("import shifting::*\nanswer()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(2)


# ---------------------------------------------------------------------------
# Self-validation still covers everything compiled here
# ---------------------------------------------------------------------------


@pytest.fixture
def sealed_modules(monkeypatch: pytest.MonkeyPatch) -> list[ModuleId]:
    """Record the module ids whose checked artifact self-validation seals."""
    recorded: list[ModuleId] = []
    original = typecheck_program._assert_checked_module_closed

    def recording(module: object) -> None:
        module_id = getattr(module, "module_id")
        assert isinstance(module_id, ModuleId)
        recorded.append(module_id)
        original(module)

    monkeypatch.setattr(typecheck_program, "_assert_checked_module_closed", recording)
    return recorded


@pytest.fixture
def validated_site_owners(monkeypatch: pytest.MonkeyPatch) -> list[ModuleId]:
    """Record the module ids whose compiled match sites self-validation replays."""
    recorded: list[ModuleId] = []
    original = match_stage._validate_sites

    def recording(**kwargs: object) -> None:
        module_id = kwargs["module_id"]
        assert isinstance(module_id, ModuleId)
        recorded.append(module_id)
        original(**kwargs)

    monkeypatch.setattr(match_stage, "_validate_sites", recording)
    return recorded


class TestSelfValidationFollowsTheWork:
    """Self-validation seals every artifact this entry built, and only those.

    An artifact carried over from an earlier compilation is the very object
    self-validation already sealed, so re-sealing it can only repeat that
    verdict. An artifact this entry built is sealed as always.
    """

    def test_a_later_entry_seals_its_own_checked_module_and_no_other(
        self, sealed_modules: list[ModuleId]
    ) -> None:
        session = _open_session()
        assert session.eval_entry("let x = 1").ok
        sealed_modules.clear()

        assert session.eval_entry("x + 1").ok

        assert sealed_modules == [ENTRY_ID]

    def test_a_later_entry_replays_its_own_match_sites_and_no_other(
        self, validated_site_owners: list[ModuleId]
    ) -> None:
        session = _open_session()
        assert session.eval_entry("let x = 1").ok
        validated_site_owners.clear()

        assert session.eval_entry("case 1 of\n  | 1 => 2\n  | _ => 3").ok

        assert validated_site_owners == [ENTRY_ID]

    def test_a_newly_imported_module_is_sealed_and_replayed(
        self,
        tmp_path: Path,
        sealed_modules: list[ModuleId],
        validated_site_owners: list[ModuleId],
    ) -> None:
        (tmp_path / "fresh.agl").write_text(
            "def classify(n: int) -> int =\n  case n of\n    | 0 => 10\n    | _ => 20\n"
        )
        session = _session_with_root(tmp_path)
        assert session.eval_entry("let x = 1").ok
        sealed_modules.clear()
        validated_site_owners.clear()

        result = session.eval_entry("import fresh::*\nclassify(0)")

        assert result.ok, result.diagnostics
        fresh = ModuleId.from_path("fresh")
        assert fresh in sealed_modules
        assert fresh in validated_site_owners


# ---------------------------------------------------------------------------
# Skipping a seed module's infix rewrite changes nothing
# ---------------------------------------------------------------------------


_OPS_SOURCE = (
    "infixl <+> at 6\n"
    "def <+>(a: int, b: int) -> int = a + b + 1\n"
    "infixr <?> at 7\n"
    "def <?>(a: int, b: int) -> int = a * b + 2\n"
)


def _raw_chains(program: object) -> list[object]:
    """Return every raw infix chain still present in *program*."""
    found: list[object] = []

    def visit(node: object) -> None:
        if isinstance(node, syntax.RawInfixChain):
            found.append(node)

    syntax.walk(program, visit)
    return found


class TestSeedInfixSkip:
    """A skipped seed module leaves the entry's own infix resolution intact.

    The loop the skip sits in also computes the entry's ambient fixity table
    and rewrites the entry's chains, so these pin that a seed module in the
    skip set changes neither.
    """

    def test_an_operator_from_a_seed_module_resolves_to_the_same_tree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The entry's tree and ambient fixities are identical without the skip.

        The cold session names no module as already resolved, so its loop
        rewrites every module's chains; the warm one skips the seed. Both
        graphs must carry the same entry program and the same
        ``entry_infix_ambient``, which the skipped iteration also feeds.
        """
        (tmp_path / "ops.agl").write_text(_OPS_SOURCE)
        entry = "1 <+> 2 <?> 3 <+> 4"
        original = loader_module._resolve_graph_infix
        recorded: list[ModuleGraph] = []

        def recording(resolve: object) -> object:
            def wrapper(graph: object, *args: object, **kwargs: object) -> object:
                result = resolve(graph, *args, **kwargs)
                assert isinstance(result, ModuleGraph)
                recorded.append(result)
                return result

            return wrapper

        def cold_infix(graph: object, session_infix: object = None, **_kwargs: object) -> object:
            return original(graph, session_infix)

        monkeypatch.setattr(loader_module, "_resolve_graph_infix", recording(original))
        warm = _session_with_root(tmp_path)
        assert warm.eval_entry("import ops::*\n()").ok
        recorded.clear()
        warm_result = warm.eval_entry(entry)
        warm_graph = recorded[-1]

        monkeypatch.setattr(loader_module, "_resolve_graph_infix", recording(cold_infix))
        cold = _session_with_root(tmp_path)
        assert cold.eval_entry("import ops::*\n()").ok
        recorded.clear()
        cold_result = cold.eval_entry(entry)
        cold_graph = recorded[-1]

        assert warm_result.ok, warm_result.diagnostics
        assert warm_result.value == cold_result.value
        # The seed module is genuinely in the warm run's skip set.
        assert ModuleId.from_path("ops") in warm._loaded_lib_modules
        assert warm_graph.entry_infix_ambient
        assert warm_graph.entry_infix_ambient == cold_graph.entry_infix_ambient
        assert warm_graph.modules[ENTRY_ID].program == cold_graph.modules[ENTRY_ID].program

    def test_a_reused_library_program_holds_no_raw_infix_chain(self, tmp_path: Path) -> None:
        """Every retained library program is fully infix-resolved, not merely accepted."""
        ops_source = _OPS_SOURCE + "def use() -> int = 1 <+> 2 <?> 3\n"
        # The walk finds chains where there are chains, so an empty result below
        # means resolved, not unvisited.
        unresolved, _ = parse_program_seeded(
            ops_source, start_id=0, source=SourceId(label="probe"), resolve_infix=False
        )
        assert _raw_chains(unresolved)

        (tmp_path / "ops.agl").write_text(ops_source)
        session = _session_with_root(tmp_path)
        assert session.eval_entry("import ops::*\nuse()").ok
        assert session.eval_entry("use() + 1").ok

        retained = {
            module_id: module.program for module_id, module in session._loaded_lib_modules.items()
        }
        assert ModuleId.from_path("ops") in retained
        assert {
            mid: _raw_chains(program) for mid, program in retained.items() if _raw_chains(program)
        } == {}

    def test_a_session_operator_never_reaches_a_library_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session operator is visible to the entry alone, so a seed cannot see it.

        The skip alone would make this trivially true, so the resolution runs
        with the skip disabled: even re-resolved from scratch against the new
        session fixity, the library module comes back the program it was.
        """
        (tmp_path / "ops.agl").write_text(_OPS_SOURCE + "def use() -> int = 1 <+> 2 <?> 3\n")
        original = loader_module._resolve_graph_infix

        def cold_infix(graph: object, session_infix: object = None, **_kwargs: object) -> object:
            return original(graph, session_infix)

        monkeypatch.setattr(loader_module, "_resolve_graph_infix", cold_infix)
        session = _session_with_root(tmp_path)
        assert session.eval_entry("import ops::*\nuse()").ok
        before = {
            module_id: module.program for module_id, module in session._loaded_lib_modules.items()
        }
        assert ModuleId.from_path("ops") in before

        # A session-level operator that collides with the library's own name,
        # at a priority and associativity the library never declared.
        assert session.eval_entry("infixr <+> at 9\ndef <+>(a: int, b: int) -> int = a - b").ok
        assert session.eval_entry("use()").value == IntValue(10)

        for module_id, program in before.items():
            assert session._loaded_lib_modules[module_id].program == program


# ---------------------------------------------------------------------------
# Reuse never bridges supersession
# ---------------------------------------------------------------------------


class TestSupersessionUnderReuse:
    def test_redeclared_enum_keeps_old_and_new_member_identities_apart(self) -> None:
        session = _open_session()
        assert session.eval_entry("enum E\n  | A(old: int)").ok
        assert session.eval_entry("type OldA = E::A").ok
        assert session.eval_entry("let old: E = A(old = 1)").ok
        assert session.eval_entry("enum E\n  | B(fresh: text)").ok
        assert session.eval_entry('let new: E = B(fresh = "new")').ok

        assert session.eval_entry("old is A").value == BoolValue(True)
        assert session.eval_entry("(old as OldA).old").value == IntValue(1)
        assert session.eval_entry("(new as E::B).fresh").value == TextValue("new")
        assert not session.eval_entry("old as E::B").ok

    def test_redeclared_record_keeps_the_retained_value_at_its_old_shape(self) -> None:
        session = _open_session()
        assert session.eval_entry("record R\n  a: int").ok
        assert session.eval_entry("let old = R(a = 1)").ok
        assert session.eval_entry("record R\n  b: text").ok

        assert session.eval_entry("old.a").value == IntValue(1)
        assert not session.eval_entry("let mixed: R = old").ok
        assert session.eval_entry('R(b = "x").b').value == TextValue("x")


# ---------------------------------------------------------------------------
# Cold-versus-warm differential
# ---------------------------------------------------------------------------


class _Discarding(dict[object, object]):
    """A cache that accepts every write and never keeps one."""

    def __setitem__(self, key: object, value: object) -> None:
        return

    def update(self, *args: object, **kwargs: object) -> None:
        return


def _run_transcript(session: ReplSession, entries: tuple[str, ...]) -> list[object]:
    """Return one comparable outcome tuple per entry in *entries*."""

    def outcome(result: EntryResult) -> object:
        return (
            result.ok,
            result.kind,
            result.name,
            result.value,
            repr(result.value_type),
            tuple((d.message, d.line, d.column) for d in result.diagnostics),
            tuple(w.message for w in result.warnings),
        )

    return [outcome(session.eval_entry(entry)) for entry in entries]


# One REPL transcript per element, kept short so a run stays cheap: success and
# failure, records, enums, casts, supersession, exceptions, mutation, imported
# standard-library modules, operators, and a syntax error.
_DIFFERENTIAL_TRANSCRIPTS: tuple[tuple[str, ...], ...] = (
    ("let x = 1", "x + 1"),
    ("def f(n: int) -> int = n * 3", "f(4)"),
    ("record P\n  x: int\n  y: int", "let p = P(x = 1, y = 2)\np.x + p.y"),
    (
        "enum Shape\n  | Dot(n: int)\n  | Line(a: int, b: int)",
        "case Dot(n = 3) of\n  | Dot(n) => n\n  | Line(a, b) => a + b",
    ),
    ('let bad: int = "text"', "undefined_name"),
    ("let ( = 1", "1 + 2 * 3 - 4"),
    ("enum E\n  | A(old: int)", "let old: E = A(old = 1)\nold is A"),
    ("record R\n  a: int\nlet r = R(a = 1)", "record R\n  b: text"),
    ('let t = "12" as int', "t + 1"),
    ("exception Boom\n  why: text", 'try\n  raise Boom(why = "x")\nwith\n  | Boom(why) => why'),
    ("var counter = 0", "counter := counter + 5\ncounter"),
    ("import std/math", "std/math::abs(0 - 3)"),
    ("import std/array::*", "length([1, 2, 3])"),
)


def _go_cold(monkeypatch: pytest.MonkeyPatch, *, reparse: bool) -> ReplSession:
    """Open a session that keeps no artifact of an earlier compilation.

    It discards the retained resolution and checked tables, calls the loader
    without naming any module as already resolved, and offers no previous match
    compilation. With *reparse* it also parses every library file afresh,
    still from the reserved id band so those ids stay disjoint from the ones
    the graph seeds its own modules with.
    """
    original_infix = loader_module._resolve_graph_infix

    def cold_infix(graph: object, session_infix: object = None, **_kwargs: object) -> object:
        return original_infix(graph, session_infix)

    original_compile = matchcompile.compile_program_matches

    def cold_compile(checked: object, previous: object = None) -> object:
        return original_compile(checked)

    monkeypatch.setattr(loader_module, "_resolve_graph_infix", cold_infix)
    monkeypatch.setattr(matchcompile, "compile_program_matches", cold_compile)

    if reparse:
        next_reserved_id = [RESERVED_NODE_ID_BASE]

        def cold_parsed_module(
            module_id: ModuleId, path: Path, *, default_stdlib: bool, build: ModuleBuilder
        ) -> object:
            source_text = normalize_newlines(fs.read_text(path))
            module, next_reserved_id[0] = build(next_reserved_id[0], source_text)
            return module

        monkeypatch.setattr(loader_module, "cached_parsed_module", cold_parsed_module)

    cold = _open_session()
    cold._retained_resolved_modules = _Discarding()
    cold._retained_checked_modules = _Discarding()
    return cold


@pytest.mark.parametrize("entries", _DIFFERENTIAL_TRANSCRIPTS, ids=lambda e: e[0][:24])
def test_library_reuse_changes_no_entry_outcome(
    entries: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transcript produces identical results with and without library reuse."""
    warm_outcomes = _run_transcript(_open_session(), entries)

    cold_outcomes = _run_transcript(_go_cold(monkeypatch, reparse=False), entries)

    assert warm_outcomes == cold_outcomes


@pytest.mark.parametrize("entries", _DIFFERENTIAL_TRANSCRIPTS[:3], ids=lambda e: e[0][:24])
def test_reparsing_the_library_changes_no_entry_outcome(
    entries: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same holds when the cold session also reparses every library file.

    A few transcripts carry this variant rather than all of them, because the
    reparse is the expensive half and the parse cache has its own equivalence
    suite (``tests/test_agl_stdlib_module_cache.py``).
    """
    warm_outcomes = _run_transcript(_open_session(), entries)

    cold_outcomes = _run_transcript(_go_cold(monkeypatch, reparse=True), entries)

    assert warm_outcomes == cold_outcomes


@pytest.fixture(autouse=True)
def _drain_console(capsys: pytest.CaptureFixture[str]) -> Iterator[None]:
    """REPL entries may print; nothing here asserts on the console."""
    yield
    capsys.readouterr()
