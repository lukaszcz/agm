# Plan: Redesign AgL `if` syntax (optional leading pipe + `if` expressions)

## Overview

AgL's `if` is currently **statement-only**. The grammar is:

```ebnf
if_stmt   ::= "if" if_branch ("|" if_branch)*
if_branch ::= bar_expr "=>" branch_body
            | "else" "=>" branch_body
```

The very first branch has **no** leading `|`, which makes the multi-branch,
multi-line "aligned pipe" layout asymmetric (the first arm sticks out). And `if`
cannot be used where a value is expected — only `case` has an expression form
(`CaseExpr`).

This plan delivers three things requested by the owner:

1. **Optional leading `|` after `if`** — `if | A => B1 | else => B2` becomes legal
   (the existing no-pipe form `if A => B1 | else => B2` stays legal everywhere).
2. **Leading-pipe form preferred in docs/examples** whenever an `if` has **more
   than one branch**.
3. **`if` as an expression** — `print (if | A => B1 | else => B2)` becomes legal.

The implementation mirrors the **existing `case` statement/expression duality**
(`CaseStmt`/`CaseExpr`), which already solves every hard problem here: LALR(1)
conflict-freeness via statement-vs-expression reachability stratification,
bar-safe positioning, branch-type unification, and parenthesization in bar-safe
contexts. We reuse that proven shape rather than inventing a new one.

## Decisions

### Resolved with the owner

1. **`if` expressions require `else` (statically).** An `if` used in expression
   position must have an `else` branch; otherwise the type checker raises a clear
   error. This makes every `if`-expression total and gives it a single, always-
   defined result type — no nullable/optional machinery needed (AgL has none).
   *(Note: this is intentionally stricter than `case_expr`, which permits a
   runtime `MatchError` on non-exhaustive match. `if` has no patterns to drive
   exhaustiveness, so a static `else` requirement is the clean analogue.)*

2. **`if` expressions are allowed bare in general expression positions**, exactly
   like `case_expr`: unparenthesized wherever a general `expr` is accepted, but
   **parenthesized** in bar-safe positions (conditions, branch bodies, `do`/`until`
   conditions, etc.). The owner's `print (if …)` example is one such use; bare
   `print if … => … | else => …` is also legal because `print` takes a general
   `expr`.

3. **Dual AST nodes: keep `IfStmt`, add `IfExpr`.** Parallels `CaseStmt`/`CaseExpr`
   exactly. Statement branch bodies remain statement-lists/suites; expression
   branch bodies are single bar-safe exprs.

4. **`if`-expression branch types unify like `case_expr`** — all branch bodies
   must share one type, with `int → decimal` widening as the only coercion
   (reuses the existing unification logic in `_check_case_expr`).

### Derived directly from the requirements (not open questions)

- The optional leading `|` applies to **both** the statement and expression forms,
  **unconditionally** — so `if | A => B` (single branch, leading pipe) is legal,
  and the legacy no-pipe form `if A => B1 | …` remains legal in both forms.
- Docs/examples switch to the leading-pipe form **only when there is more than
  one branch**; single-branch `if` stays pipe-less in docs.
- `else`-must-be-last continues to hold for both forms (already enforced for
  `IfStmt`; replicated for `IfExpr`).

## Grammar changes (`src/agm/agl/grammar/agl.lark`)

### 1. Optional leading pipe on the statement form

```ebnf
if_stmt: "if" PIPE? if_branch (PIPE if_branch)*
```

`PIPE?` is safe: an `if_branch` begins with `bar_expr` or `else`, never with
`PIPE`, so there is no shift/reduce ambiguity on the token following `if`.

### 2. New expression form

Add `if_expr` parallel to `case_expr`, with bar-safe branch bodies:

```ebnf
if_expr: "if" PIPE? if_expr_branch (PIPE if_expr_branch)*

if_expr_branch: bar_expr ARROW bar_expr   -> if_expr_cond_branch
              | "else" ARROW bar_expr      -> if_expr_else_branch
```

Wire it into the general expression rule alongside `case_expr`:

```ebnf
?expr: case_expr | if_expr | or_expr
```

`paren_expr: LPAR expr RPAR` already re-admits any `expr` (incl. `if_expr`) into
bar-safe positions, so the parenthesized form works for free.

### Why this stays LALR(1) conflict-free (the critical risk)

The grammar's conflict-guard test (`tests/test_agl_parser.py::TestConflictGuard`)
is a hard gate. The design is conflict-free by the **same argument that already
makes `case_stmt`/`case_expr` work**:

- `expr_stmt: or_expr` (not `expr`) — so at **statement** level a bare `if` can
  only be reduced as `if_stmt` (reached via `open_stmt`); `if_expr` is
  unreachable there. Identical to the existing rule that keeps statement-level
  bare `case` as `case_stmt`.
- In **expression** positions (`paren_expr`, list/dict elements, interpolation,
  `let`/`var`/`set`/`print` RHS, `case` subject, …) only `if_expr` is reachable;
  `if_stmt` is not. The bodies after `=>` therefore differ by left context
  (`branch_body` vs `bar_expr`) in distinct LALR states — no merge, no conflict.
- `if` is already a reserved keyword token (distinct from `VAR_NAME` and from
  `or_expr`'s FIRST set), and `PIPE?` adds no ambiguity.

**Plan gate:** rebuild the parser and run the conflict guard *first*, before any
downstream work. If an unexpected conflict appears, stop and reassess (do not
suppress it).

### Layout interaction (leading pipe on its own line)

The lexer's `|`-continuation rule (`src/agm/agl/lexer/layout.py`) already
suppresses the `_NEWLINE` before a line-leading `|`. Combined with `PIPE?`, this
enables the fully-aligned multi-line form:

```agl
if
| code is Fail => set x = a
| else => set x = b
```

Add explicit lexer/parser tests for this (it is a new, now-reachable shape).

## AST changes (`src/agm/agl/syntax/nodes.py`)

Add expression-form nodes mirroring `CaseExprBranch`/`CaseExpr`:

```python
@dataclass(frozen=True, slots=True)
class IfExprBranch:
    cond: Expr | ElseSentinel   # ELSE sentinel for the else arm
    body: Expr                  # single bar-safe expression
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)

@dataclass(frozen=True, slots=True)
class IfExpr:
    branches: tuple[IfExprBranch, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
```

Add `IfExpr` to the `Expr` union (next to `CaseExpr`). `IfStmt`/`IfBranch` are
unchanged structurally.

## Transformer changes (`src/agm/agl/parser/transform.py`)

- `if_stmt`: unchanged handler — the new `PIPE?` produces no extra child (it's a
  filtered terminal), and `if_branch` collection by `isinstance` is unaffected.
- Add `if_expr_cond_branch`, `if_expr_else_branch`, and `if_expr` handlers,
  parallel to the `case_expr*` handlers and the existing `if_*` handlers.
- `if_expr`: collect branches; enforce **else-must-be-last** here (reuse the same
  check as `if_stmt`, factored into a shared helper to avoid duplication per
  CLAUDE.md). The *else-required* rule lives in the checker (decision 1), not the
  transformer, so the diagnostic is a type error with proper typing context;
  confirm placement during implementation and keep it in exactly one place.

## Type checker changes (`src/agm/agl/typecheck/checker.py`)

- Add `IfExpr` to the `_check_expr` dispatch (next to the `CaseExpr` branch).
- Add `_check_if_expr(node, *, expected)`:
  - Each non-`else` `cond` must be `bool` — reuse `_require_bool_condition`
    (already parameterized by keyword: pass `"if"`).
  - **Require an `else` branch**; otherwise `AglTypeError` ("an `if` used as an
    expression must have an `else` branch", span at the `if`).
  - Unify branch-body types using the **same** logic as `_check_case_expr`
    (`int → decimal` widening only). Factor the unification loop into a shared
    helper used by both `_check_case_expr` and `_check_if_expr` (avoid
    duplication).
- `_check_if` (statement) is unchanged.

## Interpreter changes (`src/agm/agl/eval/interpreter.py`)

- Add `IfExpr` to the `_eval_expr` dispatch (next to `CaseExpr`).
- Add `_eval_if_expr(expr, scope)`: evaluate conditions left-to-right in fresh
  branch scopes; return the first true branch's body value; the `else` body is
  the guaranteed fallback (the checker proved `else` exists, so no runtime
  "no-match" path is reachable — but include a defensive `assert_never`-style
  guard consistent with the codebase).
- `_exec_if` (statement) is unchanged.

## Visitor changes (`src/agm/agl/syntax/visitor.py`)

Register `IfExpr`/`IfExprBranch` in:
- the `visit_*` protocol method stubs,
- the node-tuple import block,
- the structural `walk`/dispatch `isinstance` chain,
mirroring the existing `CaseExpr`/`CaseExprBranch` and `IfStmt`/`IfBranch` entries.

## Other source touchpoints to audit

- `src/agm/agl/capabilities.py` and `src/agm/agl/diagnostics.py` — grep for
  `CaseExpr`/`IfStmt` handling and add `IfExpr` symmetrically if these enumerate
  node kinds.
- REPL (`src/agm/agl/repl/`) — confirm an `if`-expression typed at the prompt is
  echoed/evaluated like a `case`-expression (it routes through `_eval_expr`, so
  likely free, but add a REPL test).
- `src/agm/agl/scope/` — if scope analysis walks expression nodes, ensure branch
  conditions/bodies of `IfExpr` are traversed (the visitor change should cover
  this; verify).

## Tests (TDD — write failing tests first)

Per the repo's TDD policy and 100% coverage requirement. Group by existing files
(`tests/test_agl_parser.py`, typecheck, eval, plus `tests/agl/programs/*.agl`
fixtures).

**Parser / grammar**
- Conflict guard still passes (mandatory, run first).
- `if | A => B | else => C` (statement, leading pipe) parses to the same `IfStmt`
  as the no-pipe form.
- Single-branch `if | A => B` (leading pipe) parses.
- Leading pipe on its own line (layout interaction), multi-line aligned form.
- `if_expr`: `let x = if A => 1 | else => 2`, `print (if | A => 1 | else => 2)`,
  bare `print if A => 1 | else => 2`, nested in list/dict/interpolation.
- `if_expr` in a bar-safe position **requires** parens (e.g. as an `if`/`until`
  condition or another branch body) — assert the unparenthesized form is a
  syntax error and the parenthesized form parses.
- `else`-not-last is a syntax error for both forms.

**Type checker**
- `if`-expression without `else` → error.
- Branch-type mismatch → error; `int`/`decimal` branches widen to `decimal`.
- Non-`bool` condition in an `if`-expression → error (same message family as the
  statement form).

**Interpreter**
- `if`-expression returns the first true branch's value; `else` fallback taken
  when no condition holds; branch scopes isolate bindings; widening materializes.

**Program fixtures (`tests/agl/programs/`)**
- Add a fixture exercising `if`-expressions and the leading-pipe statement form
  (with an expected-output/golden file, matching the existing fixture harness).

## Documentation updates (required by area CLAUDE.md)

Each AgL syntax change MUST update the reference docs and keep arch docs current.

- `docs/agl/reference/control-flow.md` — update the `if` EBNF (optional leading
  pipe), switch multi-branch examples to the leading-pipe form, and add an
  `if`-expression subsection (require-`else` rule, parenthesization in bar-safe
  positions, branch-type unification). Cross-link to the `case` expression
  section for the parallel design.
- `docs/agl/reference/grammar.md` — update the `if` block (`if_stmt` +
  `if_expr`/`if_expr_branch`) and the `Expressions` rule (`expr ::= case_expr |
  if_expr | bar_expr`).
- `docs/agl/reference/expressions.md` — add `if` to the list of expression forms
  (next to `case`).
- `docs/agl-grammar.md` — keep the implementation-facing grammar notes in sync
  (statement taxonomy / bar-safe stratification commentary; note `if_expr`
  alongside `case_expr`).
- `docs/arch/agl.md` — note that `if` now has a statement/expression duality
  mirroring `case`, if the arch doc enumerates control-flow node kinds.
- Any other multi-branch `if` examples found across `docs/` (and example `.agl`
  files referenced by docs) → convert to the leading-pipe form.

## Out of scope

- Removing/deprecating the no-leading-pipe form (it stays legal).
- `elif` or other new branch keywords.
- Nullable/optional types or a non-`else` `if`-expression result.
- Changes to `case` semantics (only shared helpers are refactored, not behavior).

## Suggested implementation order

1. Grammar: add `PIPE?` + `if_expr` rules; rebuild; **run conflict guard** (gate).
2. AST nodes (`IfExpr`/`IfExprBranch`) + `Expr` union + visitor registration.
3. Transformer handlers (+ shared else-last helper).
4. Type checker (`_check_if_expr` + shared branch-unification helper +
   else-required rule).
5. Interpreter (`_eval_if_expr`).
6. Audit capabilities/diagnostics/REPL/scope touchpoints.
7. Tests (write per-layer failing tests ahead of each step where practical).
8. Documentation sweep.
9. `just check` (lint + tests + strict mypy) — must pass with no `type: ignore`,
   `noqa`, or formatter suppressions (ask the owner if any seems unavoidable).
