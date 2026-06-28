# AgL Integer-Range `for` Loops — Implementation Plan

Status: planned · Date: 2026-06-28 · **Every** design decision below is owner-approved.

This plan extends the existing unified loop construct
([2026-06-28-agl-loop-syntax.md](2026-06-28-agl-loop-syntax.md)) with **integer-range** `for`
clauses. It is purely additive to the loop work already landed (`Loop`/`Break`/`Continue` AST,
single `IrLoop` primitive, iterator-based collection `for`); it changes only the `for`-clause
surface, the typecheck of the `for` collection, and the loop lowering.

## 1. Goal

Add range iteration as a second shape of the existing `for` clause, reusing `in`:

```
for i in a to b do … done             # i = a, a+1, …, b          (inclusive)
for i in a to b by 3 do … done        # i = a, a+3, …  (≤ b)
for i in a downto b do … done         # i = a, a-1, …, b          (inclusive)
for i in a downto b by k do … done    # i = a, a-k, …  (≥ b)
```

The range form is an alternative tail on the existing `for NAME in EXPR` clause: when the collection
expression is followed by `to`/`downto`, the clause iterates an **`int` counter** instead of a
collection. Everything else about the loop (`while`, the `[n]` bound, `until`/`done`, `break`/
`continue`, nesting) is unchanged and composes with the range form exactly as with the collection
form.

## 2. Non-goals

- **No range type or range value.** `a to b` is grammar that exists *only* inside a `for` clause; it
  is never a first-class expression, never assignable, never returned. (Consistent with the loop
  plan's "no range type" non-goal.)
- **No `decimal`/`text` ranges.** Bounds and step are `int` only ("integer ranges").
- **No reverse-by-negative-step.** Direction comes solely from `to`/`downto`; `by` is always
  positive (D3).
- **No new comprehension, `else`-on-loop, or change to `while`/collection-`for` semantics.**
- **No new IR loop kind.** The range desugars into the existing single `IrLoop(body)` primitive
  (D5), so the evaluator's loop handler is untouched.

## 3. Settled design decisions (authoritative)

Each was settled one-by-one with the owner.

### D1 — Bounds are **inclusive**
`for i in a to b` runs while `i ≤ b`; `for i in a downto b` runs while `i ≥ b`. `for i in 1 to 3`
binds `i = 1, 2, 3`. The words `to`/`downto` read as "up/down **to and including** b" (matching
Pascal/Kotlin/Ruby). With a `by` step the last value is the greatest `a + k·step ≤ b` (for `to`) or
least `a − k·step ≥ b` (for `downto`); the step may overshoot `b` and simply stops.

### D2 — `to`/`downto`/`by` are **reserved keywords, admitted in `field_name`**
They become reserved keywords (added to `KEYWORDS`, `%declare`d as terminals `TO`/`DOWNTO`/`BY`).
Because `by` is already used as a record/field name in existing programs (`tagged(by: T)`,
`Holder.tagged(by: 7)`, `| tagged(by: who)`), all three are **admitted in `field_name`** exactly as
the reserved word `agent` already is:

```lark
?field_name: NAME | AGENT | TO | DOWNTO | BY
```

This preserves every existing field/key use while making them keywords in expression position. The
LALR conflict-guard (0 shift/reduce, 0 reduce/reduce) **must be re-verified** with these admissions
(§9, M1 gate).

### D3 — Step keyword is **`by`**, **positive-only**
The optional step is written `by EXPR`. The step must be a **positive** `int`; direction is given by
`to`/`downto`, never by the step's sign. An omitted step defaults to `1`.

### D4 — A non-positive `by` raises **`RangeError`** at runtime
`by` is an arbitrary `int` expression, so `by ≤ 0` cannot always be caught statically. A new builtin
prelude exception **`RangeError`** is raised once at loop entry when the evaluated step is `≤ 0`. It
is an ordinary catchable AgL exception (fields `message`, `trace_id`, like the other builtins). When
`by` is a literal `≤ 0`, typecheck *may* additionally reject it statically (a cheap, optional
nicety — see §5.3); the runtime check is the authority.

### D5 — Lowering is an **integer-counter desugar** (no iterator)
The range lowers to a mutable `int` cursor plus a step-add and a direction comparison, emitted
directly into the existing `IrLoop` body — **not** through `IteratorValue`/`IrIterInit`. A range
never materializes: `for i in 1 to 1_000_000` uses O(1) memory. No new IR node, no new runtime
value, no `IterKind` member. (See §6.)

### Accepted assumptions (no objection raised)
- **A1.** Bounds `a`, `b` and step `by k` are each evaluated **exactly once at loop entry**, in the
  enclosing scope, in source order (`a`, then `b`, then `k`) — the same timing as the existing
  collection `for_iter` and the `[n]` bound. They cannot reference the loop variable.
- **A2.** The loop variable is `int`, a fresh **immutable** binding each iteration (`:=` to it is a
  static error), in scope in `while`/body/`until` — identical to the collection `for` variable.
- **A3.** A degenerate range runs the body **zero times** and completes normally (`a > b` for `to`,
  `a < b` for `downto`), matching `for` over an empty collection.
- **A4.** The loop expression's type stays `unit`; the range is a `for`-clause variant only.
- **A5.** `to`/`downto`/`by` become reserved words; the corpus/tests are grepped for non-field
  identifier collisions as the first M1 task. (`by` field-name uses are preserved by D2; `to`/
  `downto` had no identifier uses in the corpus.)

## 4. Surface syntax & grammar

### 4.1 Grammar (`src/agm/agl/grammar/agl.lark`)

Extend `for_clause` with an optional range tail; add the `field_name` admissions (D2):

```lark
for_clause   : "for" name "in" or_expr range_tail? _NEWLINE?

range_tail   : range_dir or_expr range_by?
range_dir    : TO     -> range_to
             | DOWNTO -> range_downto
range_by     : BY or_expr

?field_name  : NAME | AGENT | TO | DOWNTO | BY
```

- `TO`/`DOWNTO`/`BY` are `%declare`d terminals (like `AGENT`), emitted by `AglLexer` from the
  lowercase keywords; using declared terminals (not string literals) is what lets them appear in
  `field_name`.
- When `range_tail` is **absent**, the clause is the existing collection `for` (over `list`/`dict`/
  `text`). When **present**, it is a range `for`. The two are distinguished purely by the presence of
  `to`/`downto` after the first `or_expr`.
- `range_by` is optional; absent ⇒ step `1`.

**LALR conflict-freeness must be verified, not assumed.** `TO`/`DOWNTO` follow `or_expr` only inside
`range_tail`; `BY` follows `or_expr` only inside `range_by`; in `field_name` they sit before `:`/
after `.`, distinct positions. The expectation is 0/0, but the §9 M1 gate is the conflict-guard test
(`tests/test_agl_parser.py`) run against the real `agl.lark` + `AglLexer`. If any conflict appears,
stop and revisit with the owner before proceeding.

### 4.2 Lexer (`src/agm/agl/lexer/tokens.py`)

- Add `KW_TO = "to"`, `KW_DOWNTO = "downto"`, `KW_BY = "by"` and include them in `KEYWORDS`. The
  existing uppercase remap (`kw.upper()`) yields the `TO`/`DOWNTO`/`BY` terminal names the grammar
  `%declare`s. No layout change is needed — the range tail introduces no new indentation or
  continuation rules (it lives on the `for` header line, before the existing optional `_NEWLINE`).
- `in`/`do`/`until`/`done` are already keywords; nothing else changes in the lexer.

## 5. Frontend passes

### 5.1 AST (`src/agm/agl/syntax/nodes.py`)

The range needs three new optional fields on `Loop` (or a small sub-node) to carry the `to` bound,
the direction, and the step. Recommended: extend `Loop` so collection and range `for` share one node
and the lowerer/typechecker branch on `for_range_to is not None`.

```python
@dataclass(frozen=True, slots=True)
class Loop:
    for_var: str | None          # unchanged
    for_iter: Expr | None        # collection form: present iff collection `for`
    # --- new: integer-range form (present iff this is a range `for`) ---
    for_range_to: Expr | None    # the `to`/`downto` bound `b`; None ⇒ not a range
    for_range_down: bool         # True for `downto`, False for `to`
    for_range_by: Expr | None    # the `by` step; None ⇒ default 1
    # ---
    while_cond: Expr | None      # unchanged
    bound: Expr | None           # unchanged ([n] safety bound)
    body: Expr
    until_cond: Expr | None
    span; node_id
```

Invariants (assert in the parser transformer): a range `for` has `for_var is not None`,
`for_iter is not None` (the **lower bound `a`** is stored in `for_iter`), and
`for_range_to is not None`. A collection `for` has `for_range_to is None`. Reusing `for_iter` for
the lower bound keeps `walk`/`spans` traversal of the start expression unchanged; add traversal of
`for_range_to` and `for_range_by` to `syntax/visitor.py::walk` and `syntax/spans.py`.

Parser transformer (`parser/transform.py`): `for_clause` already returns `(name, start_expr)`;
extend it to also surface the optional `range_tail` (direction + `to` bound + optional `by`), and
thread those into the `Loop` built by `loop_expr` / `loop_clauses`.

### 5.2 Scope / name resolution (`src/agm/agl/scope/`)

Mirror the existing collection-`for` scoping (A1/A2):

1. Resolve `for_iter` (the start `a`), `for_range_to` (`b`), and `for_range_by` (`k`) **in the
   enclosing scope** — none may see the loop variable.
2. Open the loop scope; bind `for_var` as the existing **immutable** loop-variable binder kind (so
   `:=` to it is rejected with the established message).
3. Resolve `while_cond`, `body`, `until_cond` in that scope.

The `[n]` bound stays resolved in the enclosing scope as today. No new scope rule is required beyond
walking the two new expressions.

### 5.3 Type checking (`src/agm/agl/typecheck/checker.py::_check_loop`)

Branch on the range form. When `for_range_to is not None`:

- `for_iter` (start `a`), `for_range_to` (`b`), and `for_range_by` (`k`, when present) must each be
  `int` (`AglTypeError` otherwise, pointing at the offending expression).
- The loop variable's type is `int` (`set_binding_type(node.node_id, IntType())`).
- **Optional static guard (D4):** if `for_range_by` is a literal `int` constant `≤ 0`, raise an
  `AglTypeError` ("loop step must be positive"). This is a convenience only; the runtime `RangeError`
  remains the authority for non-literal steps. (Implement only if it's a clean constant-check; do not
  build dataflow for it.)

The collection branch (`for_range_to is None`) is the existing `list`/`dict`/`text` logic, unchanged.
`while`/`until` stay `bool`, `[n]` stays `int`, the loop stays `unit`.

## 6. Execution IR & lowering

**No new IR nodes, no new runtime value, no `IterKind` member** (D5). The range desugars in
`lower/lowerer.py::_lower_loop`, replacing **items 1–2** (the iterator exhaustion check + iterator
bind) of the existing desugar with counter-based equivalents. Items 3–7 (`while`, `[n]` check,
`[n]` count, body, `until`) are emitted **unchanged**.

### 6.1 Pre-loop (enclosing frame, once, in source order)

For a range `for i in a (to|downto) b [by k]`:

- `var __cur  = lower_coerced(a, int)`   — mutable synthetic cursor.
- `let __end  = lower_coerced(b, int)`   — immutable synthetic.
- `let __step = lower_coerced(k, int)` *(or `IrConstInt(1)` when `by` is omitted)* — immutable.
- **Step guard:** `if __step <= 0 => raise RangeError(...)` (§6.3). Emitted once, after `__step` is
  bound, before the loop.

These precede the existing optional `[n]` bound pre-loop items (`__n`, `__count`).

### 6.2 Loop body items (replace items 1–2; keep 3–7)

Emitted into the same `IrLoop(body = IrBlock[…])`:

1. **Range termination check** (replaces the iterator exhaustion check):
   - `to`:     `if __cur >  __end => IrBreak`
   - `downto`: `if __cur <  __end => IrBreak`
   (strict comparison ⇒ inclusive bound, D1.)
2. **Range bind + advance** (replaces the iterator bind):
   - `let i = __cur`            *(immutable loop var, into the frame slot)*
   - `to`:     `__cur := __cur + __step`
   - `downto`: `__cur := __cur - __step`

The bind reads `__cur` **before** the advance, so `i` holds the current value and `__cur` already
points at the next — mirroring `IrIterNext` (return-current-then-advance). This is what makes
`continue` correct: `IrContinue` re-enters at item 1, which tests the **already-advanced** `__cur`,
so a continued iteration still advances and still cannot loop forever.

Items 3–7 are reused verbatim from the collection desugar: `while` guard, `[n]` check, `[n]` count,
`body`, `until` guard.

### 6.3 `RangeError` and the step guard

Add `RangeError` to `BUILTIN_EXCEPTIONS` (`semantics/types.py`) with the standard
`{message: text, trace_id: text}` fields (same shape as `MatchError`/`ArithmeticError` minus the
extra field). The step guard fully desugars to an ordinary
`IrRaise(IrMakeException(nominal=NominalId(PRELUDE_ID, "RangeError"), fields=…))` — exactly the
mechanism the loop plan uses for `MaxIterationsExceeded`, so **no dedicated IR node and no evaluator
special-case**:

- `message` → `IrConstText("loop step must be positive")` (or an `IrRenderTemplate` including the
  bad value, if cheap);
- `trace_id` → `AutoTraceField()`.

`PRELUDE_ID`, `IrMakeException`, `AutoTraceField`, `IrRaise`, the int comparison/`+`/`-`/`:=` nodes
all already exist.

### 6.4 Worked traces (✓ = matches D1/A3)

| Source | Behaviour |
|---|---|
| `for i in 1 to 3` | `__cur`1: 1>3?no → i=1,`__cur`2; 2>3?no → i=2,`__cur`3; 3>3?no → i=3,`__cur`4; 4>3?yes → break. **i=1,2,3** ✓ |
| `for i in 1 to 6 by 2` | i=1(`__cur`3),3(5),5(7); 7>6?yes → break. **i=1,3,5** ✓ |
| `for i in 3 downto 1` | i=3(2),2(1),1(0); 0<1?yes → break. **i=3,2,1** ✓ |
| `for i in 5 to 1` | 5>1?yes → break. **0 iterations**, normal ✓ (A3) |
| `for i in 1 to 5 by 0` | step guard `0<=0` → **raise RangeError** before any iteration ✓ (D4) |
| `for i in 1 to 9 by 2` + `continue` | `continue` re-enters item 1 on advanced `__cur` → still advances, still terminates ✓ |
| `for i in 1 to 100 do[3] done` | `[n]` check (item 4) raises `MaxIterationsExceeded` after 3 body entries ✓ (bound composes) |

### 6.5 Evaluator (`src/agm/agl/eval/`)

**No changes.** Every node the desugar emits (`IrBind`, `IrLoad`, `IrAssign`, `IrBinary` int `+`/
`-`/comparison, `IrIf`, `IrBreak`, `IrRaise`, `IrMakeException`) is already handled. The range never
touches `IrIterInit`/`IteratorValue`.

## 7. Documentation

- `docs/agl/reference/control-flow.md` — document the range `for` (the `to`/`downto`/`by` forms,
  inclusivity, positive step, `int`-only, degenerate ⇒ zero iterations, evaluation-once timing).
  Implementation-free.
- `docs/agl/reference/lexical-structure.md` / `grammar.md` — add `to`/`downto`/`by` to the reserved
  words and the `for_clause` EBNF.
- `docs/agl/reference/exceptions.md` — add `RangeError` to the builtin exceptions.
- `docs/arch/agl/frontend.md` — note the range fields on `Loop`, the `field_name` keyword
  admissions, and the `int`-range typing.
- `docs/arch/agl/execution.md` — note the counter desugar (range `for` does **not** use the iterator
  ops) and `RangeError`; the AST→IR coverage table needs no new node row.

## 8. Testing strategy (TDD)

Per AgL area guidance, **e2e programs under `tests/agl/programs/` come first**, combined with other
features. Cover:

- Each form: `to`, `downto`, `to … by k`, `downto … by k`; bound by `until E`, `done`, omitted.
- Range `for` composed with: a `while` clause, a `[n]` bound, nesting (inner/outer ranges, range
  outside / collection inside and vice-versa), `break`/`continue` (innermost target; `continue`
  advancing the cursor), `if`/`case` in the body, and inside a `try` body.
- Inclusivity (`1 to 3` ⇒ 1,2,3; `3 downto 1` ⇒ 3,2,1); step overshoot (`1 to 6 by 2` ⇒ 1,3,5).
- Degenerate ranges (`5 to 1`, `1 downto 5`) ⇒ zero iterations, normal completion.
- `RangeError`: `by 0` and a negative dynamic `by` raise and are catchable; literal `by 0` rejected
  statically (if the optional static guard is implemented).
- Loop variable: `int`, visible in `while`/body/`until`, not after the loop, `:=` to it a static
  error; bounds/step cannot reference it.
- Bounds/step evaluated once (use a side-effecting start/step expression and assert single eval).
- Static errors: non-`int` start/`to`/`by`; `by` step on a *collection* `for` is a parse error
  (range tail only follows the bound expression); two `for`s / order rules unchanged.
- Field-name preservation: existing `tagged(by: …)` programs still parse/run (regression for D2).

Plus unit/golden tests: parser conflict-guard **0/0** with the new tokens + `field_name` admissions;
`Loop` range-field AST shape; scope/typecheck errors; **golden lowering IR** for each range desugar
(start/end/step pre-loop, the two body items, the step guard, `to` vs `downto` comparison/advance);
evaluator behaviour via the e2e suite. Maintain **100% `src/` coverage** and **100% command
coverage**; concurrency-robust; never run real agents.

## 9. Milestones (TDD; commit per milestone; `just check` green at each)

- **M1 — Grammar, lexer, AST, parser.** First task: grep corpus/tests for `to`/`downto`/`by`
  identifier (non-field) collisions. Add `KW_TO`/`KW_DOWNTO`/`KW_BY` + `KEYWORDS`; `%declare`
  `TO`/`DOWNTO`/`BY` and the `field_name` admissions; `for_clause` `range_tail` grammar;
  **conflict-guard test green (0/0)** — hard gate, stop if it fails; `Loop` range fields; transformer
  builds them; `walk`/`spans` updated. Parse tests incl. the existing `tagged(by: …)` regression.
- **M2 — Scope.** Resolve start/`to`/`by` in the enclosing scope; immutable `int` loop var; range
  composes with `while`/`[n]`/`break`/`continue` placement (unchanged rules). Scope tests.
- **M3 — Typecheck.** `int` start/`to`/`by`; loop var `int`; optional literal-`by ≤ 0` static guard;
  collection branch untouched. Type tests.
- **M4 — Lowering + `RangeError`.** Add `RangeError` to the prelude; counter desugar in `_lower_loop`
  (pre-loop cursor/end/step + step guard; body items 1–2 replaced; items 3–7 reused); golden lowering
  tests for `to`/`downto`/`by`/default-step and the step guard. No evaluator change; `validate.py`
  needs no new node.
- **M5 — Evaluator + e2e.** Confirm the existing evaluator runs every desugar; the
  `tests/agl/programs/` range suite green (all §8 e2e cases), including `RangeError` and single-eval.
- **M6 — Docs & cleanup.** Update reference + arch docs + grammar/lexical/exceptions; final
  `just check` at 100% `src/` and command coverage.

## 10. Resolved design points (no open risks)

- **LALR conflict-freeness — gated, not assumed.** New tokens + `field_name` admissions must report
  0/0 against the real `agl.lark` + `AglLexer`; the M1 conflict-guard is the hard gate.
- **`by` field-name collision — resolved by precedent.** `to`/`downto`/`by` admitted in `field_name`
  exactly as `agent` is; existing `tagged(by: …)` preserved, locked by a regression test (D2).
- **No iterator, no materialization — counter desugar.** Range `for` lowers to an `int` cursor; O(1)
  memory for any range size; no `IteratorValue`/`IterKind`/evaluator change (D5). Locked by golden
  lowering + e2e.
- **Inclusive bounds + positive step — comparison/advance encode it.** Strict `>`/`<` against `__end`
  gives inclusivity (D1); direction picks `+`/`-`; `by` validated `> 0` once at entry (D3/D4). Locked
  by the §6.4 traces as golden + e2e tests.
- **`continue` correctness.** Advance-at-top (item 2 reads `__cur` then steps it) means `continue`
  re-runs the termination check on the advanced cursor — never an infinite continue. Locked by an
  explicit test.
- **`RangeError` raise — fully desugared.** `IrRaise(IrMakeException(…))` with `AutoTraceField`, no
  dedicated node, reusing the `MaxIterationsExceeded` mechanism. Locked by golden lowering + e2e.
