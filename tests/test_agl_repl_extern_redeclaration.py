"""REPL-level extern boundary behavior across nominal type redeclaration.

A companion's captured nominal class -- a module global, a closure, a
default argument -- denotes the declaration that was current when the
companion imported it, permanently. Redeclaring that record, enum, or
exception in a later REPL entry never re-shapes an already-synthesized
class: the captured class keeps constructing and recognizing values of the
declaration it was captured from, while the redeclaration gets a fresh
class of its own. A companion importing after the redeclaration sees
whichever declaration currently bears the name -- never a superseded one,
and never one from an entry that failed before promotion.

Companion loading mechanics, capability gating, and ``:reset`` are covered
by ``TestExternRepl`` in ``test_agl_repl_session.py``; this module is scoped
to what changes specifically around a nominal redeclaration.
"""

from __future__ import annotations

from pathlib import Path

from agm.agl.repl import ReplSession
from agm.agl.semantics.values import (
    ArrayValue,
    ExceptionValue,
    IntValue,
    RecordValue,
    TextValue,
)

_IDENTITY_LIB_AGL = "extern def identity[T](x: T) -> T\n"
_IDENTITY_LIB_PY = "def identity(x):\n    return x\n"


def _make_session_with_root(root: Path) -> ReplSession:
    from agm.agl.modules.roots import assemble_roots

    roots = assemble_roots(
        invocation_root=root,
        stdlib_root=Path(__file__).resolve().parents[1] / "stdlib",
        lib_root=None,
        configured=[],
        cli=[],
        cwd=root,
    )
    session = ReplSession()
    session._roots = roots
    return session


def _write_extern_lib(root: Path, name: str, agl: str, py: str) -> None:
    (root / f"{name}.agl").write_text(agl)
    (root / f"{name}.py").write_text(py)


# ---------------------------------------------------------------------------
# A captured class keeps constructing/recognizing the declaration it was
# captured from -- module global, closure, and default argument -- and is
# never re-shaped by a later redeclaration.
# ---------------------------------------------------------------------------


class TestCapturedClassSurvivesRedeclaration:
    def test_default_argument_captured_record_class_keeps_the_old_shape(
        self, tmp_path: Path
    ) -> None:
        _write_extern_lib(
            tmp_path,
            "capture_default_arg",
            "extern def make_and_read(v: int) -> int\n",
            (
                "from agl import Box\n"
                "def make_and_read(v, box_cls=Box):\n"
                "    return box_cls(value=v).value\n"
            ),
        )
        s = _make_session_with_root(tmp_path)
        assert s.eval_entry("record Box\n  value: int").ok
        assert s.eval_entry("open import capture_default_arg").ok

        assert s.eval_entry("record Box\n  value: int\n  extra: int").ok

        r = s.eval_entry("make_and_read(5)")

        assert r.ok, r.diagnostics
        assert r.value == IntValue(5)

    def test_module_global_captured_record_class_keeps_the_old_shape(self, tmp_path: Path) -> None:
        _write_extern_lib(
            tmp_path,
            "capture_global",
            "extern def make_and_read(v: int) -> int\n",
            (
                "from agl import Box\n"
                "CAPTURED = Box\n"
                "def make_and_read(v):\n"
                "    return CAPTURED(value=v).value\n"
            ),
        )
        s = _make_session_with_root(tmp_path)
        assert s.eval_entry("record Box\n  value: int").ok
        assert s.eval_entry("open import capture_global").ok

        assert s.eval_entry("record Box\n  value: int\n  extra: int").ok

        r = s.eval_entry("make_and_read(7)")

        assert r.ok, r.diagnostics
        assert r.value == IntValue(7)

    def test_closure_captured_record_class_keeps_the_old_shape(self, tmp_path: Path) -> None:
        _write_extern_lib(
            tmp_path,
            "capture_closure",
            "extern def make_and_read(v: int) -> int\n",
            (
                "from agl import Box\n"
                "def _make_factory(box_cls):\n"
                "    def factory(v):\n"
                "        return box_cls(value=v).value\n"
                "    return factory\n"
                "make_and_read = _make_factory(Box)\n"
            ),
        )
        s = _make_session_with_root(tmp_path)
        assert s.eval_entry("record Box\n  value: int").ok
        assert s.eval_entry("open import capture_closure").ok

        assert s.eval_entry("record Box\n  value: int\n  extra: int").ok

        r = s.eval_entry("make_and_read(9)")

        assert r.ok, r.diagnostics
        assert r.value == IntValue(9)

    def test_captured_exception_class_keeps_the_old_shape(self, tmp_path: Path) -> None:
        _write_extern_lib(
            tmp_path,
            "capture_exception",
            "extern def make_and_read(detail: text) -> text\n",
            (
                "from agl import Problem\n"
                "def make_and_read(detail, problem_cls=Problem):\n"
                "    box = problem_cls(message='boom', detail=detail)\n"
                "    return box.detail\n"
            ),
        )
        s = _make_session_with_root(tmp_path)
        assert s.eval_entry("exception Problem extends Exception\n  detail: text").ok
        assert s.eval_entry("open import capture_exception").ok

        assert s.eval_entry("exception Problem extends Exception\n  detail: text\n  code: int").ok

        r = s.eval_entry('make_and_read("bad")')

        assert r.ok, r.diagnostics
        assert r.value == TextValue("bad")

    def test_captured_enum_class_keeps_old_variants_and_never_gains_new_ones(
        self, tmp_path: Path
    ) -> None:
        """Variants a redeclaration adds or drops never reach an already-captured class."""
        _write_extern_lib(
            tmp_path,
            "capture_enum",
            "extern def variant_report() -> text\n",
            (
                "from agl import Choice\n"
                "def variant_report():\n"
                "    names = sorted(n for n in vars(Choice) if not n.startswith('_'))\n"
                "    return ','.join(names)\n"
            ),
        )
        s = _make_session_with_root(tmp_path)
        assert s.eval_entry("enum Choice\n  | Some(value: int)\n  | Gone").ok
        assert s.eval_entry("open import capture_enum").ok
        before = s.eval_entry("variant_report()")

        # Drops ``Gone`` and adds ``Other``.
        assert s.eval_entry("enum Choice\n  | Some(value: int)\n  | Other").ok
        after = s.eval_entry("variant_report()")

        assert before.ok, before.diagnostics
        assert after.ok, after.diagnostics
        assert before.value == TextValue("Gone,Some")
        assert after.value == TextValue("Gone,Some")


# ---------------------------------------------------------------------------
# A companion importing after a redeclaration sees the declaration that
# currently bears the name -- never a superseded one, never one from an
# unpromoted entry -- under both its bare alias and its ``nominals`` path.
# ---------------------------------------------------------------------------


class TestFreshImportSeesTheCurrentDeclaration:
    def test_fresh_import_sees_the_new_record_shape_not_the_old_one(self, tmp_path: Path) -> None:
        _write_extern_lib(
            tmp_path,
            "capture_after",
            "extern def make_and_report(v: int) -> text\n",
            (
                "from agl import Box, nominals\n"
                "def make_and_report(v):\n"
                "    assert Box is nominals.entry.Box\n"
                "    fields = sorted(Box._agl_fields)\n"
                "    Box(**{name: v for name in fields})\n"
                "    return ','.join(fields)\n"
            ),
        )
        s = _make_session_with_root(tmp_path)
        assert s.eval_entry("record Box\n  value: int").ok
        assert s.eval_entry("record Box\n  value: int\n  extra: int").ok

        assert s.eval_entry("open import capture_after").ok
        r = s.eval_entry("make_and_report(1)")

        assert r.ok, r.diagnostics
        assert r.value == TextValue("extra,value")

    def test_fresh_import_after_an_unpromoted_redeclaration_sees_the_promoted_shape(
        self, tmp_path: Path
    ) -> None:
        """A failed, never-promoted redeclaration still lands its own identity
        in the type table (see ``TestUnpromotedNominalDeclarationEffects`` in
        ``test_agl_repl_session.py``): a fresh import afterward must see the
        declaration that survived promotion, not the orphaned failed one,
        and not the declaration before it either.
        """
        _write_extern_lib(
            tmp_path,
            "capture_after_failure",
            "extern def make_and_report(v: int) -> text\n",
            (
                "from agl import Box, nominals\n"
                "def make_and_report(v):\n"
                "    assert Box is nominals.entry.Box\n"
                "    fields = sorted(Box._agl_fields)\n"
                "    Box(**{name: v for name in fields})\n"
                "    return ','.join(fields)\n"
            ),
        )
        s = _make_session_with_root(tmp_path)
        assert s.eval_entry("record Box\n  value: int").ok
        assert s.eval_entry("record Box\n  value: int\n  extra: int").ok
        failed = s.eval_entry(
            "let z: decimal = 1 / 0\nrecord Box\n  value: int\n  extra: int\n  more: int"
        )
        assert not failed.ok

        assert s.eval_entry("open import capture_after_failure").ok
        r = s.eval_entry("make_and_report(1)")

        assert r.ok, r.diagnostics
        assert r.value == TextValue("extra,value")

    def test_fresh_import_sees_the_new_enum_members_not_the_old_ones(self, tmp_path: Path) -> None:
        _write_extern_lib(
            tmp_path,
            "capture_enum_after",
            "extern def variant_report() -> text\n",
            (
                "from agl import Choice, nominals\n"
                "def variant_report():\n"
                "    assert Choice is nominals.entry.Choice\n"
                "    names = sorted(n for n in vars(Choice) if not n.startswith('_'))\n"
                "    return ','.join(names)\n"
            ),
        )
        s = _make_session_with_root(tmp_path)
        assert s.eval_entry("enum Choice\n  | Some(value: int)\n  | Gone").ok
        assert s.eval_entry("enum Choice\n  | Some(value: int)\n  | Other").ok

        assert s.eval_entry("open import capture_enum_after").ok
        r = s.eval_entry("variant_report()")

        assert r.ok, r.diagnostics
        assert r.value == TextValue("Other,Some")

    def test_fresh_import_does_not_see_a_never_promoted_declaration(self, tmp_path: Path) -> None:
        """A declaration from a failed entry is not a name the boundary offers.

        Its name never reached the session, so exposing a class for it would
        let a companion construct values of a type the session cannot name.
        """
        _write_extern_lib(
            tmp_path,
            "visible_nominals",
            "extern def visible() -> text\n",
            (
                "from agl import nominals\n"
                "def visible():\n"
                "    return ','.join(\n"
                "        sorted(n for n in vars(nominals.entry) if not n.startswith('_'))\n"
                "    )\n"
            ),
        )
        s = _make_session_with_root(tmp_path)
        assert s.eval_entry("record Kept\n  value: int").ok
        failed = s.eval_entry(
            'let stop: int = raise Abort(message = "stop")\nrecord Ghost\n  value: int'
        )
        assert not failed.ok

        assert s.eval_entry("open import visible_nominals").ok
        r = s.eval_entry("visible()")

        assert r.ok, r.diagnostics
        assert r.value == TextValue("Kept")


# ---------------------------------------------------------------------------
# Encoding and decoding both directions across the boundary, for both the
# old and the new identity, in the same session.
# ---------------------------------------------------------------------------


class TestBoundaryRoundTripAcrossRedeclaration:
    def test_record_values_round_trip_for_both_old_and_new_identities(self, tmp_path: Path) -> None:
        _write_extern_lib(tmp_path, "identity_lib", _IDENTITY_LIB_AGL, _IDENTITY_LIB_PY)
        s = _make_session_with_root(tmp_path)
        assert s.eval_entry("record Box\n  value: int").ok
        assert s.eval_entry("let old = Box(value = 1)").ok
        assert s.eval_entry("record Box\n  value: int\n  extra: int").ok
        assert s.eval_entry("open import identity_lib").ok

        old_round_trip = s.eval_entry("identity(old)")
        new_round_trip = s.eval_entry("let new = Box(value = 2, extra = 3)\nidentity(new)")

        assert old_round_trip.ok, old_round_trip.diagnostics
        assert new_round_trip.ok, new_round_trip.diagnostics
        assert isinstance(old_round_trip.value, RecordValue)
        assert isinstance(new_round_trip.value, RecordValue)
        assert old_round_trip.value.fields == {"value": IntValue(1)}
        assert new_round_trip.value.fields == {"value": IntValue(2), "extra": IntValue(3)}
        assert old_round_trip.value.nominal != new_round_trip.value.nominal

    def test_enum_values_round_trip_for_both_old_and_new_identities(self, tmp_path: Path) -> None:
        _write_extern_lib(tmp_path, "identity_lib", _IDENTITY_LIB_AGL, _IDENTITY_LIB_PY)
        s = _make_session_with_root(tmp_path)
        assert s.eval_entry("enum Choice\n  | Some(value: int)\n  | Gone").ok
        assert s.eval_entry("let old = Choice::Gone").ok
        assert s.eval_entry("enum Choice\n  | Some(value: int)\n  | Other").ok
        assert s.eval_entry("open import identity_lib").ok

        old_round_trip = s.eval_entry("identity(old)")
        new_round_trip = s.eval_entry("let new = Choice::Other\nidentity(new)")

        assert old_round_trip.ok, old_round_trip.diagnostics
        assert new_round_trip.ok, new_round_trip.diagnostics
        assert isinstance(old_round_trip.value, RecordValue)
        assert isinstance(new_round_trip.value, RecordValue)
        assert old_round_trip.value.display_name.rsplit("::", maxsplit=1)[-1] == "Gone"
        assert new_round_trip.value.display_name.rsplit("::", maxsplit=1)[-1] == "Other"
        assert old_round_trip.value.nominal != new_round_trip.value.nominal

    def test_exception_values_round_trip_for_both_old_and_new_identities(
        self, tmp_path: Path
    ) -> None:
        _write_extern_lib(tmp_path, "identity_lib", _IDENTITY_LIB_AGL, _IDENTITY_LIB_PY)
        s = _make_session_with_root(tmp_path)
        assert s.eval_entry("exception Problem extends Exception\n  detail: text").ok
        assert s.eval_entry('let old = Problem(message = "old", detail = "x")').ok
        assert s.eval_entry("exception Problem extends Exception\n  detail: text\n  code: int").ok
        assert s.eval_entry("open import identity_lib").ok

        old_round_trip = s.eval_entry("identity(old)")
        new_round_trip = s.eval_entry(
            'let new = Problem(message = "new", detail = "y", code = 1)\nidentity(new)'
        )

        assert old_round_trip.ok, old_round_trip.diagnostics
        assert new_round_trip.ok, new_round_trip.diagnostics
        assert isinstance(old_round_trip.value, ExceptionValue)
        assert isinstance(new_round_trip.value, ExceptionValue)
        assert old_round_trip.value.fields["detail"] == TextValue("x")
        assert "code" not in old_round_trip.value.fields
        assert new_round_trip.value.fields["code"] == IntValue(1)
        assert old_round_trip.value.nominal != new_round_trip.value.nominal


# ---------------------------------------------------------------------------
# Regression guard: live array/dict views are untouched by any of the above.
# ---------------------------------------------------------------------------


class TestLiveViewsUnaffectedByNominalRedeclaration:
    def test_array_view_mutation_still_works_around_an_unrelated_redeclaration(
        self, tmp_path: Path
    ) -> None:
        _write_extern_lib(
            tmp_path,
            "touch_array",
            "extern def touch(xs: array[int]) -> unit\n",
            "def touch(xs):\n    xs.append(99)\n",
        )
        s = _make_session_with_root(tmp_path)
        assert s.eval_entry("record Box\n  value: int").ok
        assert s.eval_entry("open import touch_array").ok
        assert s.eval_entry("record Box\n  value: int\n  extra: int").ok

        r = s.eval_entry("let xs: array[int] = [1, 2]\ntouch(xs)\nxs")

        assert r.ok, r.diagnostics
        assert r.value == ArrayValue([IntValue(1), IntValue(2), IntValue(99)])


# ---------------------------------------------------------------------------
# A companion-import rejection registers its declarations' nominal identity
# with the extern registry before it fails, so it must not release its
# node-id range: doing so would let a later entry's redeclaration reuse an
# identity the (insert-only) registry already has on file under the old shape.
# ---------------------------------------------------------------------------


class TestRejectedCompanionImportReleasesNoStaleIdentity:
    def test_redeclaration_after_a_failed_companion_import_gets_the_new_shape(
        self, tmp_path: Path
    ) -> None:
        _write_extern_lib(
            tmp_path,
            "broken_companion",
            "extern def noop() -> int\n",
            "raise RuntimeError('boom')\n",
        )
        _write_extern_lib(
            tmp_path,
            "report_shape",
            "extern def make_and_report(v: text) -> text\n",
            (
                "from agl import R, nominals\n"
                "def make_and_report(v):\n"
                "    assert R is nominals.entry.R\n"
                "    fields = sorted(R._agl_fields)\n"
                "    R(**{name: v for name in fields})\n"
                "    return ','.join(fields)\n"
            ),
        )
        s = _make_session_with_root(tmp_path)

        rejected = s.eval_entry("open import broken_companion\nrecord R\n  a: int")
        assert not rejected.ok

        redeclared = s.eval_entry("open import report_shape\nrecord R\n  b: text")
        assert redeclared.ok, redeclared.diagnostics

        r = s.eval_entry('make_and_report("x")')

        assert r.ok, r.diagnostics
        assert r.value == TextValue("b")
