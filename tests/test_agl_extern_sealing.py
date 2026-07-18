"""Sealing-contract tests for `extern def` type variables, through real programs.

Every extern with a type parameter seals values at type-variable positions in
an opaque handle minted per call, per variable. This suite drives that
contract entirely through real companion Python callables and real AgL
programs:
- parametric utilities (`reverse`, `count`, `merge`, `identity`) at several
  instantiations, including records and closures (a type variable
  instantiated at a function type by the caller is legal — only function/
  agent types in the extern's own *declared* signature are banned).
- handle `==`/`hash`/`repr` support in Python: a companion deduplicating via
  a set, sorting stably by `repr`, and counting distinct log entries.
- every sealing violation, each surfacing as a catchable `ExternError`: a
  forged raw value at a type-variable return position, a handle stashed from
  a previous call, a handle swapped between two type variables of one call,
  and partial forgery inside a returned `list[T]`/`dict[T]`.
- an extern called from inside another generic AgL function, at a rigid type
  variable.

`tests/test_agl_extern_boundary.py` covers `SealedHandle` and the
encode/decode walkers directly with hand-built contracts; this suite never
calls those directly, only through `extern def` calls evaluated end to end.
"""

from __future__ import annotations

from pathlib import Path

from agm.agl.semantics.values import BoolValue, IntValue, ListValue, TextValue
from tests.agl.ir_harness import evaluate_ir_raises_with_externs, evaluate_ir_with_externs

# ---------------------------------------------------------------------------
# Parametric utilities at several instantiations
# ---------------------------------------------------------------------------


class TestParametricUtilitiesAtSeveralInstantiations:
    def test_reverse_at_int(self, tmp_path: Path) -> None:
        source = "extern def reverse[T](xs: list[T]) -> list[T]\nlet r = reverse([1, 2, 3])\nr\n"
        companion = "def reverse(xs):\n    return list(reversed(xs))\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == ListValue((IntValue(3), IntValue(2), IntValue(1)))

    def test_reverse_at_text(self, tmp_path: Path) -> None:
        source = (
            'extern def reverse[T](xs: list[T]) -> list[T]\nlet r = reverse(["a", "b", "c"])\nr\n'
        )
        companion = "def reverse(xs):\n    return list(reversed(xs))\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == ListValue((TextValue("c"), TextValue("b"), TextValue("a")))

    def test_reverse_at_record(self, tmp_path: Path) -> None:
        source = (
            "record Box\n  value: int\n"
            "extern def reverse[T](xs: list[T]) -> list[T]\n"
            "let r = reverse([Box(value = 1), Box(value = 2)])\n"
            "let first = r[0].value\n"
            "first\n"
        )
        companion = "def reverse(xs):\n    return list(reversed(xs))\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["first"] == IntValue(2)

    def test_reverse_at_closure(self, tmp_path: Path) -> None:
        """A type variable instantiated at a function type by the caller is
        legal: `T` still seals opaquely regardless of what it is instantiated
        to (only a function/agent type in the extern's own declared signature
        is a static error)."""
        source = (
            "extern def reverse[T](xs: list[T]) -> list[T]\n"
            "def inc(x: int) -> int = x + 1\n"
            "def dec(x: int) -> int = x - 1\n"
            "let fs = reverse([inc, dec])\n"
            "let r = fs[0](10)\n"
            "r\n"
        )
        companion = "def reverse(xs):\n    return list(reversed(xs))\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(9)

    def test_count_at_int(self, tmp_path: Path) -> None:
        source = "extern def count[T](xs: list[T]) -> int\nlet r = count([1, 2, 3, 4])\nr\n"
        companion = "def count(xs):\n    return len(xs)\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(4)

    def test_count_at_record(self, tmp_path: Path) -> None:
        source = (
            "record Box\n  value: int\n"
            "extern def count[T](xs: list[T]) -> int\n"
            "let r = count([Box(value = 1), Box(value = 2), Box(value = 3)])\n"
            "r\n"
        )
        companion = "def count(xs):\n    return len(xs)\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(3)

    def test_merge_at_text(self, tmp_path: Path) -> None:
        source = (
            "extern def merge[T](a: list[T], b: list[T]) -> list[T]\n"
            'let r = merge(["a"], ["b", "c"])\n'
            "r\n"
        )
        companion = "def merge(a, b):\n    return a + b\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == ListValue((TextValue("a"), TextValue("b"), TextValue("c")))

    def test_identity_at_int(self, tmp_path: Path) -> None:
        source = "extern def identity[T](x: T) -> T\nlet r = identity(42)\nr\n"
        companion = "def identity(x):\n    return x\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(42)


# ---------------------------------------------------------------------------
# Handle ==/hash/repr support in Python
# ---------------------------------------------------------------------------


class TestHandleEqualityHashReprInPython:
    def test_companion_deduplicates_int_handles_via_a_python_set(self, tmp_path: Path) -> None:
        source = (
            "extern def unique_count[T](xs: list[T]) -> int\n"
            "let r = unique_count([1, 1, 2, 2, 2, 3])\n"
            "r\n"
        )
        companion = "def unique_count(xs):\n    return len(set(xs))\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(3)

    def test_companion_deduplicates_equal_record_handles(self, tmp_path: Path) -> None:
        source = (
            "record Box\n  value: int\n"
            "extern def unique_count[T](xs: list[T]) -> int\n"
            "let r = unique_count([Box(value = 1), Box(value = 1), Box(value = 2)])\n"
            "r\n"
        )
        companion = "def unique_count(xs):\n    return len(set(xs))\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(2)

    def test_companion_sorts_handles_stably_via_repr(self, tmp_path: Path) -> None:
        source = (
            "extern def sort_by_repr[T](xs: list[T]) -> list[T]\n"
            'let r = sort_by_repr(["banana", "apple", "cherry"])\n'
            "r\n"
        )
        companion = "def sort_by_repr(xs):\n    return sorted(xs, key=repr)\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == ListValue(
            (TextValue("apple"), TextValue("banana"), TextValue("cherry"))
        )

    def test_companion_counts_distinct_log_entries_built_from_reprs(self, tmp_path: Path) -> None:
        """Reprs render the wrapped AgL value for debugging but expose no
        other surface; assert the observable count of distinct entries, never
        the rendered text itself."""
        source = "extern def log_count[T](xs: list[T]) -> int\nlet r = log_count([1, 2, 1])\nr\n"
        companion = (
            "def log_count(xs):\n    log = [repr(x) for x in xs]\n    return len(set(log))\n"
        )
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(2)

    def test_companion_compares_two_handles_of_the_same_type_variable_by_value(
        self, tmp_path: Path
    ) -> None:
        source = "extern def same[T](a: T, b: T) -> bool\nlet r = same(1, 1)\nr\n"
        companion = "def same(a, b):\n    return a == b\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == BoolValue(True)

    def test_companion_compares_equal_dict_handles_with_different_render_order(
        self, tmp_path: Path
    ) -> None:
        source = (
            "extern def same[T](a: T, b: T) -> bool\n"
            'let r = same({"a": 1, "b": 2}, {"b": 2, "a": 1})\n'
            "r\n"
        )
        companion = "def same(a, b):\n    return a == b\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == BoolValue(True)


# ---------------------------------------------------------------------------
# Sealing violations: every one an ExternError catchable in AgL
# ---------------------------------------------------------------------------


class TestSealingViolations:
    def test_forged_raw_value_at_a_type_var_return_position_rejected(self, tmp_path: Path) -> None:
        exc = evaluate_ir_raises_with_externs(
            "extern def identity[T](x: T) -> T\nidentity(1)\n()\n",
            "def identity(x):\n    return 999\n",
            tmp_path,
        )
        assert exc.display_name == "ExternError"

    def test_forged_value_is_catchable_with_try(self, tmp_path: Path) -> None:
        source = (
            "extern def identity[T](x: T) -> T\n"
            "let r = try\n"
            "  identity(1)\n"
            "catch ExternError =>\n"
            "  -1\n"
            "r\n"
        )
        companion = "def identity(x):\n    return 999\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(-1)

    def test_handle_stashed_from_a_previous_call_rejected(self, tmp_path: Path) -> None:
        source = (
            "extern def stash[T](x: T) -> T\n"
            "extern def leak[T]() -> T\n"
            "stash(1)\n"
            "leak::[int]()\n"
            "()\n"
        )
        companion = (
            "_stash = None\n"
            "def stash(x):\n"
            "    global _stash\n"
            "    _stash = x\n"
            "    return x\n"
            "def leak():\n"
            "    return _stash\n"
        )
        exc = evaluate_ir_raises_with_externs(source, companion, tmp_path)
        assert exc.display_name == "ExternError"

    def test_handle_swapped_between_two_type_variables_of_one_call_rejected(
        self, tmp_path: Path
    ) -> None:
        source = 'extern def pair[A, B](a: A, b: B) -> B\npair(1, "x")\n()\n'
        companion = "def pair(a, b):\n    return a\n"
        exc = evaluate_ir_raises_with_externs(source, companion, tmp_path)
        assert exc.display_name == "ExternError"

    def test_partial_forgery_inside_a_returned_list_rejected(self, tmp_path: Path) -> None:
        source = "extern def process[T](xs: list[T]) -> list[T]\nprocess([1, 2])\n()\n"
        companion = "def process(xs):\n    return [xs[0], 999]\n"
        exc = evaluate_ir_raises_with_externs(source, companion, tmp_path)
        assert exc.display_name == "ExternError"

    def test_partial_forgery_inside_a_returned_dict_rejected(self, tmp_path: Path) -> None:
        source = (
            "extern def process[T](d: dict[text, T]) -> dict[text, T]\nprocess({a: 1, b: 2})\n()\n"
        )
        companion = "def process(d):\n    return {'a': d['a'], 'b': 999}\n"
        exc = evaluate_ir_raises_with_externs(source, companion, tmp_path)
        assert exc.display_name == "ExternError"

    def test_companion_minted_handle_from_public_helpers_is_rejected(self, tmp_path: Path) -> None:
        source = "extern def forge[T]() -> T\nforge::[int]()\n()\n"
        companion = (
            "from agm.agl.ir.contracts import BoundarySealVar\n"
            "from agm.agl.runtime.externs import encode_boundary_value\n"
            "from agm.agl.semantics.values import IntValue\n"
            "def forge():\n"
            "    schema = BoundarySealVar('T')\n"
            "    return encode_boundary_value(schema, IntValue(999), {'T': object()})\n"
        )
        exc = evaluate_ir_raises_with_externs(source, companion, tmp_path)
        assert exc.display_name == "ExternError"


# ---------------------------------------------------------------------------
# Extern calls at a rigid type variable
# ---------------------------------------------------------------------------


class TestExternCallAtRigidTypeVariable:
    def test_generic_agl_function_calls_a_generic_extern_with_its_own_type_variable(
        self, tmp_path: Path
    ) -> None:
        source = (
            "extern def identity[T](x: T) -> T\n"
            "def wrap[U](x: U) -> U = identity(x)\n"
            "let r = wrap(5)\n"
            "r\n"
        )
        companion = "def identity(x):\n    return x\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(5)

    def test_generic_agl_function_calls_a_generic_extern_at_a_record_type_variable(
        self, tmp_path: Path
    ) -> None:
        source = (
            "record Box\n  value: int\n"
            "extern def identity[T](x: T) -> T\n"
            "def wrap[U](x: U) -> U = identity(x)\n"
            "let r = wrap(Box(value = 9))\n"
            "let v = r.value\n"
            "v\n"
        )
        companion = "def identity(x):\n    return x\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["v"] == IntValue(9)
