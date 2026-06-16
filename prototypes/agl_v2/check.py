#!/usr/bin/env python3
"""
AgL v2 grammar conflict-guard and corpus checker.

Usage:
    uv run python prototypes/agl_v2/check.py

Exits 0 if:
  - Zero LALR shift/reduce and reduce/reduce conflicts.
  - All positive corpus examples parse successfully.
  - All negative examples raise parse errors.
  - Parse-tree shape assertions pass for the three discriminating cases.

Exits non-zero otherwise.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
import textwrap
from pathlib import Path

import lark
from lark.indenter import Indenter

# ---------------------------------------------------------------------------
# Indenter for significant indentation
# ---------------------------------------------------------------------------

class AglIndenter(Indenter):
    NL_type = "_NEWLINE"
    OPEN_PAREN_types = ["LPAR", "LSQB", "LBRACE"]
    CLOSE_PAREN_types = ["RPAR", "RSQB", "RBRACE"]
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    tab_len = 4


# ---------------------------------------------------------------------------
# Grammar loading
# ---------------------------------------------------------------------------

GRAMMAR_PATH = Path(__file__).parent / "agl_v2.lark"


def build_parser(*, debug: bool = False) -> lark.Lark:
    grammar_text = GRAMMAR_PATH.read_text(encoding="utf-8")
    return lark.Lark(
        grammar_text,
        parser="lalr",
        lexer="contextual",
        postlex=AglIndenter(),
        propagate_positions=False,
        maybe_placeholders=False,
    )


# ---------------------------------------------------------------------------
# Conflict guard (mirrors TestConflictGuard in tests/test_agl_parser.py)
# ---------------------------------------------------------------------------

def check_conflicts() -> int:
    """Build the parser with DEBUG logging; return the number of conflicts found."""
    handler = logging.handlers.MemoryHandler(capacity=10000)
    lark.logger.addHandler(handler)
    old_level = lark.logger.level
    lark.logger.setLevel(logging.DEBUG)
    try:
        build_parser()
    finally:
        lark.logger.setLevel(old_level)
        lark.logger.removeHandler(handler)

    records = handler.buffer
    conflicts = [
        r
        for r in records
        if "Shift/Reduce" in r.getMessage() or "Reduce/Reduce" in r.getMessage()
    ]
    if conflicts:
        print(f"LALR CONFLICTS ({len(conflicts)}):")
        for r in conflicts:
            print(f"  {r.getMessage()}")
    else:
        print("Conflicts: 0 shift/reduce, 0 reduce/reduce  [PASS]")
    return len(conflicts)


# ---------------------------------------------------------------------------
# Positive corpus
# ---------------------------------------------------------------------------

POSITIVE_CORPUS = [
    # --- Declarations ---
    (
        "record",
        textwrap.dedent("""\
            record Issue
                title: text
                severity: int
        """),
    ),
    (
        "enum",
        textwrap.dedent("""\
            enum Review
                | Pass
                | Fail(issues: list[text])
        """),
    ),
    (
        "agent bare",
        "agent reviewer",
    ),
    (
        "agent with runner",
        'agent planner = "claude -p"',
    ),
    (
        "type alias",
        "type Score = int",
    ),
    (
        "input decl no type",
        "input name",
    ),
    (
        "input decl with type",
        "input name: text",
    ),
    (
        "config pragma bool",
        "config debug = true",
    ),
    (
        "config pragma int",
        "config retries = 3",
    ),
    (
        "config pragma string",
        'config model = "claude"',
    ),
    # --- Function defs ---
    (
        "def simple",
        'def greet(name: text, greeting: text = "Hi") -> text = "Hi"',
    ),
    (
        "def with if body (single-line branches in suite)",
        textwrap.dedent("""\
            def classify(n: int) -> text =
                if n > 0 => "pos" | n < 0 => "neg" | else => "zero"
        """),
    ),
    (
        "def recursive",
        'def fact(n: int) -> int = if n <= 1 => 1 | else => n * fact(n - 1)',
    ),
    (
        "def zero params",
        textwrap.dedent("""\
            def run() -> unit =
                let x = classify(-4)
                print x
                ()
        """),
    ),
    (
        "def do-until",
        textwrap.dedent("""\
            def loopy() -> unit =
                var acc = 0
                do
                    set acc = acc + 1
                until acc > 10
        """),
    ),
    (
        "def case (single-line in suite)",
        textwrap.dedent("""\
            def handle(v: Review) -> text =
                case v of | Pass => "ok" | Fail(issues) => "bad"
        """),
    ),
    # --- Binders ---
    (
        "let with juxt ask",
        'let s = ask "Hello?"',
    ),
    (
        "let with paren-call ask named args",
        'let r: Review = ask("Review stuff", agent: reviewer, on_parse_error: Retry(n: 2))',
    ),
    (
        "let exec juxt",
        'let res = exec "ls -la"',
    ),
    # --- Expressions ---
    (
        "field access (paren-call required for field_access arg)",
        "print(res.stdout)",
    ),
    (
        "print paren-call",
        "print(classify(-4))",
    ),
    (
        "lambda fn keyword explicit return type",
        "let dbl = fn(x: int) -> int => x * 2",
    ),
    (
        "lambda fn omitted return type (inferred)",
        "let dbl = fn(x: int) => x * 2",
    ),
    (
        "lambda fn zero params omitted return type",
        "let f = fn() => 1",
    ),
    (
        "lambda fn multi-param omitted return type",
        "let add = fn(x: int, y: int) => x + y",
    ),
    (
        "juxt field access single dot",
        "print res.stdout",
    ),
    (
        "juxt field access chained",
        "print a.b.c",
    ),
    (
        "function type annotation",
        "let g: (int) -> int = dbl",
    ),
    (
        "multi-param function type annotation",
        "let h: (int, text) -> Review = something",
    ),
    (
        "call result",
        "let n = dbl(21)",
    ),
    (
        "list literal",
        "let lst = [1, 2, 3]",
    ),
    (
        "dict literal",
        "let d = {a: 1, b: 2}",
    ),
    # --- Operators ---
    (
        "arithmetic",
        "let x = 1 + 2 * 3 - 4 / 2",
    ),
    (
        "comparison",
        "let b = x > 0",
    ),
    (
        "and/or/not",
        "let b = not x or y and z",
    ),
    (
        "is test",
        "let b = v is Pass",
    ),
    (
        "is not test",
        "let b = v is not Fail",
    ),
    (
        "in test",
        "let b = x in lst",
    ),
    # --- Constructor expressions ---
    (
        "constructor bare",
        "let x = Pass",
    ),
    (
        "constructor with payload",
        'let x = Fail(issues: ["x"])',
    ),
    (
        "qualified constructor",
        "let x = Review.Pass",
    ),
    # --- Control flow expressions ---
    (
        "if expr inline",
        'let label = if x > 0 => "pos" | else => "neg"',
    ),
    (
        "if expr with pipe-first",
        'let label = if | x > 0 => "pos" | else => "neg"',
    ),
    (
        "case expr",
        'let s = case v of | Pass => "ok" | Fail(issues) => "bad"',
    ),
    (
        "do-until standalone",
        textwrap.dedent("""\
            do
                set acc = acc + 1
            until acc > 10
        """),
    ),
    (
        "do-until with bound (LOOP_BOUND token)",
        textwrap.dedent("""\
            do [5]
                set acc = acc + 1
            until acc > 10
        """),
    ),
    (
        "try-catch",
        textwrap.dedent("""\
            try
                let x = risky()
            catch Error as e => print e
        """),
    ),
    (
        "raise",
        "raise MyError",
    ),
    # --- Multi-statement blocks ---
    (
        "semicolon-separated",
        "let a = 1; let b = 2; a + b",
    ),
    (
        "newline-separated block",
        textwrap.dedent("""\
            let a = 1
            let b = 2
            a + b
        """),
    ),
    # --- Unit type and literal ---
    (
        "unit literal in paren",
        "let u = ()",
    ),
    (
        "zero-arg call",
        "let x = f()",
    ),
    (
        "function type unit return",
        "let f: () -> text = something",
    ),
    # --- Negation and unary ---
    (
        "unary neg",
        "let x = -5",
    ),
    (
        "unary neg expr",
        "let x = -(y + 1)",
    ),
    # --- Nested calls ---
    (
        "nested paren calls",
        "let x = f(g(h(1)))",
    ),
    (
        "chained field access",
        "let x = a.b.c",
    ),
    # --- Additional patterns ---
    (
        "wildcard pattern",
        'case v of | _ => "whatever"',
    ),
    (
        "literal pattern",
        'case v of | 0 => "zero" | _ => "other"',
    ),
]


# ---------------------------------------------------------------------------
# Negative corpus (must all fail to parse)
# ---------------------------------------------------------------------------

NEGATIVE_CORPUS = [
    (
        "multi-arg juxtaposition f a b",
        "f a b",
    ),
    (
        "removed bracket syntax ask[agent: r]",
        'ask[agent: reviewer] "hello"',
    ),
    (
        "EQ_EQ operator",
        "let b = x == y",
    ),
    (
        "chained comparison",
        "let b = 1 < x < 10",
    ),
    (
        "juxt call result not in sugar (must use parens)",
        "print classify(x)",
    ),
]


# ---------------------------------------------------------------------------
# Parse-tree shape assertions
# ---------------------------------------------------------------------------

def find_rule(tree: lark.Tree, rule: str) -> lark.Tree | None:
    """Find the first subtree matching a rule name (BFS)."""
    queue = [tree]
    while queue:
        node = queue.pop(0)
        if isinstance(node, lark.Tree):
            if node.data == rule:
                return node
            queue.extend(node.children)
    return None


def check_tree_shape(parser: lark.Lark) -> list[tuple[str, bool, str]]:
    """
    Check parse-tree shapes for the three discriminating cases.

    Returns list of (case_name, passed, detail).
    """
    results = []

    # Case 1: `print x + 1` should parse as (print x) + 1
    # i.e. bin_add( juxt_call(var_ref('print'), var_ref('x')), lit_int(1) )
    src1 = "print x + 1"
    try:
        tree1 = parser.parse(src1 + "\n")
        # Look for bin_add at the top of the expression
        add_node = find_rule(tree1, "bin_add")
        if add_node is None:
            results.append((src1, False, "no bin_add found in tree"))
        else:
            # Left child of bin_add should be a juxt_call
            left = add_node.children[0]
            if isinstance(left, lark.Tree) and left.data == "juxt_call":
                results.append((src1, True, "bin_add( juxt_call(print, x), 1 )  [PASS]"))
            else:
                results.append((src1, False, f"expected juxt_call as left of bin_add, got: {left}"))
    except Exception as e:
        results.append((src1, False, f"parse error: {e}"))

    # Case 2: `f (x + 1)` should parse as a paren-call f(x+1)
    # i.e. call( var_ref('f'), arg_list( pos_arg( bin_add(...) ) ) )
    src2 = "f (x + 1)"
    try:
        tree2 = parser.parse(src2 + "\n")
        call_node = find_rule(tree2, "call")
        if call_node is None:
            results.append((src2, False, "no call node found"))
        else:
            # First child should be var_ref('f')
            fn = call_node.children[0]
            if isinstance(fn, lark.Tree) and fn.data == "var_ref":
                results.append((src2, True, "call( var_ref('f'), paren_arg(x+1) )  [PASS]"))
            else:
                results.append((src2, False, f"expected var_ref as callee, got: {fn}"))
    except Exception as e:
        results.append((src2, False, f"parse error: {e}"))

    # Case 3: `f -1` should parse as subtraction f - 1
    # i.e. bin_sub( juxt(var_ref('f')), unary_neg(lit_int(1)) )
    # Note: juxt is a CONCRETE non-terminal, so `f` alone produces juxt(var_ref('f')).
    # The key assertion is that bin_sub is found (not juxt_call) and the left operand
    # contains var_ref('f') (possibly wrapped in a juxt node).
    src3 = "f -1"
    try:
        tree3 = parser.parse(src3 + "\n")
        sub_node = find_rule(tree3, "bin_sub")
        juxt_call_node = find_rule(tree3, "juxt_call")
        if sub_node is None:
            results.append((src3, False, "no bin_sub found; parsed as juxt_call?"))
        elif juxt_call_node is not None:
            results.append((src3, False, "found juxt_call — f was applied to -1 instead of subtraction"))
        else:
            # The left child is juxt(var_ref('f')) since juxt is concrete
            left = sub_node.children[0]
            vref = find_rule(left, "var_ref") if isinstance(left, lark.Tree) else None
            if vref is not None:
                results.append((src3, True, "bin_sub( juxt(var_ref('f')), unary_neg(1) )  [PASS]"))
            else:
                results.append((src3, False, f"expected var_ref inside left of bin_sub, got: {left}"))
    except Exception as e:
        results.append((src3, False, f"parse error: {e}"))

    # Case 4: `print res.stdout` should parse as juxt_call with a juxt_field_access argument.
    # i.e. juxt_call( var_ref('print'), juxt_field_access( var_ref('res'), DOT, 'stdout' ) )
    src4 = "print res.stdout"
    try:
        tree4 = parser.parse(src4 + "\n")
        jc_node = find_rule(tree4, "juxt_call")
        if jc_node is None:
            results.append((src4, False, "no juxt_call found in tree"))
        else:
            # Second child of juxt_call (index 1) should be juxt_field_access
            arg = jc_node.children[1]
            if isinstance(arg, lark.Tree) and arg.data == "juxt_field_access":
                # First child of juxt_field_access should be var_ref('res')
                base = arg.children[0]
                if isinstance(base, lark.Tree) and base.data == "var_ref":
                    results.append(
                        (src4, True, "juxt_call( var_ref('print'), juxt_field_access(var_ref('res'), stdout) )  [PASS]")
                    )
                else:
                    results.append((src4, False, f"expected var_ref as base of juxt_field_access, got: {base}"))
            else:
                results.append((src4, False, f"expected juxt_field_access as juxt arg, got: {arg}"))
    except Exception as e:
        results.append((src4, False, f"parse error: {e}"))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ok = True

    # 1. Conflict guard
    print("=" * 60)
    print("LALR(1) CONFLICT GUARD")
    print("=" * 60)
    n_conflicts = check_conflicts()
    if n_conflicts > 0:
        ok = False

    # 2. Build parser for corpus tests
    print()
    print("=" * 60)
    print("POSITIVE CORPUS")
    print("=" * 60)
    try:
        parser = build_parser()
    except Exception as e:
        print(f"FATAL: could not build parser: {e}")
        return 1

    pos_pass = 0
    pos_fail = 0
    for name, src in POSITIVE_CORPUS:
        # Ensure trailing newline for indenter
        src_nl = src if src.endswith("\n") else src + "\n"
        try:
            parser.parse(src_nl)
            print(f"  PASS  {name}")
            pos_pass += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            print(f"        {e}")
            pos_fail += 1
            ok = False

    print(f"\nPositive: {pos_pass} passed, {pos_fail} failed")

    # 3. Negative corpus
    print()
    print("=" * 60)
    print("NEGATIVE CORPUS (must all raise parse errors)")
    print("=" * 60)
    neg_pass = 0
    neg_fail = 0
    for name, src in NEGATIVE_CORPUS:
        src_nl = src if src.endswith("\n") else src + "\n"
        try:
            parser.parse(src_nl)
            print(f"  FAIL  {name}  (parsed but should have failed)")
            neg_fail += 1
            ok = False
        except Exception:
            print(f"  PASS  {name}  (correctly rejected)")
            neg_pass += 1

    print(f"\nNegative: {neg_pass} correctly rejected, {neg_fail} incorrectly accepted")

    # 4. Parse-tree shape assertions
    print()
    print("=" * 60)
    print("PARSE-TREE SHAPE ASSERTIONS")
    print("=" * 60)
    shape_results = check_tree_shape(parser)
    shape_pass = 0
    shape_fail = 0
    for name, passed, detail in shape_results:
        if passed:
            print(f"  PASS  `{name}`: {detail}")
            shape_pass += 1
        else:
            print(f"  FAIL  `{name}`: {detail}")
            shape_fail += 1
            ok = False

    print(f"\nShape assertions: {shape_pass} passed, {shape_fail} failed")

    # 5. Summary
    print()
    print("=" * 60)
    if ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 60)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
