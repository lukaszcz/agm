# AgL Unified Loop Syntax — Implementation Plan

Status: planned · Date: 2026-06-28 · **Every** design decision below is owner-approved.

This is the standalone, authoritative design and implementation plan for extending AgL with a
unified loop syntax (`for` / `while` / `do` / `until` / `done`) and the `break` / `continue`
control-flow expressions. It supersedes the current post-test-only `do … until` construct
([control-flow.md](../agl/reference/control-flow.md)) and the do-loop redesign that introduced
arbitrary `do[expr]` bounds. The settled decisions in §3 are authoritative.

## 1. Goal

Replace AgL's single post-test loop with **one uniform loop construct** that covers all loop
shapes, and add `break` / `continue`. The surface form is:

```
loop ::= ("for" NAME "in" expr)? ("while" expr)?  "do" ("[" expr "]")?  body  ("until" expr | "done")?
```

- A loop may begin with **at most one `for`** clause and **at most one `while`** clause, in the
  fixed order `for` then `while` (so a `while` guard can always reference the `for` variable).
- `do` is always present and may carry an optional iteration **bound** `[n]`.
- The body is an indented suite (multi-line) or an inline `;`-separated sequence.
- The body ends with `until E`, `done`, or — in **multi-line** form only — nothing (the body is
  delimited by indentation). `done` and an omitted terminator are **equivalent to `until false`**.

All existing variants are special cases of this one form:

```
do … until E                while E do … done            for x in l do … done
do … done                   while E do[n] … until E      for x in l while x < k do[n] … until E
do[n] … until E
```

In the **execution IR** the loop is simplified to a **single primitive loop kind**: `IrLoop(body)`
repeats `body` unconditionally, and `IrBreak` / `IrContinue` are the only control. Every richer
feature (`for` iteration, `while`/`until` guards, the `[n]` bound) is **desugared during lowering**
into the loop body using ordinary IR — the evaluator's loop handler stays trivial.

## 2. Non-goals

- No range type, comprehension, or `else`-on-loop. No labelled loops / labelled break.
- No `break`/`continue` carrying a value (loops are `unit`).
- No multi-`for` (cartesian product) or multi-`while`; exactly ≤1 of each (owner decision D1).
- No new user-facing builtins (`length`, `keys`, …); the iteration primitives added are
  **internal IR only** and never user-visible.
- No change to other control flow (`if`, `case`, `try`).

## 3. Settled design decisions (authoritative)

Each was settled one-by-one with the owner.

### D1 — Clause cardinality and order: **≤1 `for` and ≤1 `while`, `for` first**
A loop header is an optional `for` clause followed by an optional `while` clause. Two `for`s or two
`while`s are a static/parse error. The fixed order guarantees the `while` guard can reference the
`for` variable. The header clauses and `do` may be written inline or stacked on consecutive lines
at the **same indentation**. (This supersedes the multi-clause sketch in the original request.)

### D2 — Bound `[n]` semantics: **always a safety limit (current behaviour)**
`[n]` is an `int`-typed expression evaluated **once at loop entry, in the enclosing scope**. It
counts iterations (body entries); exceeding it raises `MaxIterationsExceeded`. `n ≤ 0` runs the
body **zero times** and completes normally. An omitted bound is unbounded and never raises. Because
`done ≡ until false`, `do[n] … done` runs the body `n` times and then **raises** (the bound is the
only "real" exit and it was reached). The only exception-free exits are `for` exhaustion, a false
`while`, a true `until`, and `break`.

### D3 — `for` iterables: **`list[T]`, `dict[K,V]` (keys), `text` (chars)**
- `for x in (l: list[T])` — `x : T`, each element in order.
- `for x in (d: dict[K,V])` — `x : K`, each **key** (no tuple/pair type exists; obtain the value
  with `d[x]`). Iteration order matches the dict's runtime key order.
- `for x in (s: text)` — `x : text`, each character as a length-1 `text`.

### D4 — `for` variable: **immutable, header/body/`until` scope**
The `for` variable is a fresh **immutable** binding each iteration (assigning it with `:=` is a
static error, via the existing binder-kind mechanism). It is in scope in the `while` guard, the
body, and the `until` condition. It is **not** in scope in the `for` collection expression or the
`[n]` bound (both evaluated once at entry in the enclosing scope), nor after the loop.

### D5 — `break` / `continue`: **`BottomType`, nullary, no lambda crossing**
`break` and `continue` are nullary keyword expressions of type `BottomType` (like `raise`, so they
are assignable to any expected type and usable in any branch position). They are valid only
**lexically inside a loop body within the same function/lambda**: a `break`/`continue` outside any
enclosing loop, or one that would cross a `fn`/`def` boundary into an outer loop, is a static error.
`break` exits the innermost enclosing loop; `continue` proceeds to that loop's next iteration
(re-running the header guards and the bound check).

### D6 — Single IR loop kind: **`IrLoop(body)` + `IrBreak`/`IrContinue`, full desugar**
The IR has one loop node, `IrLoop(body)`, that repeats `body` forever; the only exits are
`IrBreak` (leave the loop) and `IrContinue` (next iteration). `while` / `for` / `until` / `[n]` are
**all desugared in the lowerer** into the body (§6.3). Refinements (owner):

- The bound's **counter and raise machinery are emitted only when a `[n]` bound is present**; an
  unbounded loop produces no counter and no comparison.
- The bound expression `n` is **evaluated exactly once before the loop** (bound to a synthetic
  immutable symbol) — no semantic change from today.

### D7 — `for` iteration mechanism: **internal iterator value + three ops**
Lowering uses an internal-only **`IteratorValue`** plus `IrIterInit(kind, collection)`,
`IrIterHasNext(iter)`, and `IrIterNext(iter)`. One uniform mechanism serves list / dict-keys / text
(`IterKind ∈ {LIST, DICT_KEYS, TEXT}`). `IteratorValue` is a runtime value tag that is never
user-constructible or user-visible (the `for` variable binds the *elements*, never the iterator).

### Accepted assumptions (no objection raised)
- **A1.** A loop expression always has type `unit`.
- **A2.** `for`, `while`, `done`, `break`, `continue` become reserved keywords (matched as the
  existing lowercase keywords are; `in`, `do`, `until` already reserved).

## 4. Surface syntax & grammar

### 4.1 Grammar (replaces `do_expr` in `src/agm/agl/grammar/agl.lark`)

```lark
loop_expr  : for_clause? while_clause? "do" loop_bound? do_body loop_end

for_clause : "for" name "in" or_expr _NEWLINE?
while_clause : "while" or_expr _NEWLINE?

loop_bound : DO_LSQB or_expr RSQB           // first "[" after "do" is retagged DO_LSQB

do_body    : suite_expr                     // multi-line indented block
           | inline_seq                     // inline ;-separated sequence

loop_end   : "until" or_expr  -> loop_until // post-test exit condition
           | "done"           -> loop_done  // == until false
```

- `loop_expr` replaces `do_expr` in the `?expr` alternatives. `do_body`, `inline_seq` /
  `inline_item`, `suite_expr`, and `loop_bound` are reused unchanged from the current `do_expr`
  machinery.
- **The terminator (`loop_end`) is mandatory in the grammar** — structurally identical to today's
  `do_expr` (which always requires `until`). This is what keeps the grammar conflict-free (see
  below). The "terminator optional in multi-line" rule (D1) is realized **in the layout**, which
  injects a synthetic `done` token when a multi-line loop body dedents without an explicit
  `until`/`done` (§4.2). Inline loops therefore always carry an explicit terminator and need no
  injection.
- The optional `_NEWLINE` after each header clause lets `for` / `while` / `do` stack on separate
  same-indent lines; written inline, no `_NEWLINE` is present.

`break` / `continue` are added as nullary primary (atom-level) expressions:

```lark
?atom : break_expr | continue_expr | …
break_expr    : "break"
continue_expr : "continue"
```

**LALR conflict-freeness is verified, not assumed.** This grammar was spiked against the real
`agl.lark` built with Lark LALR + `AglLexer` (the same construction `tests/test_agl_parser.py`
checks): **0 shift/reduce, 0 reduce/reduce.** The earlier attempt that made the terminator optional
*after a suite* (`suite_expr loop_end?`) produced 2 shift/reduce conflicts on `UNTIL`/`DONE`: the
inline form (`inline_seq loop_end`) places `or_expr` immediately before the terminator, so
`UNTIL`/`DONE` enter `FOLLOW(or_expr)` and, via LALR state merging, the suite's optional-terminator
state. Keeping the terminator mandatory removes the competing reduce entirely. The conflict-guard
test remains the M1 gate.

### 4.2 Lexer / layout (`src/agm/agl/lexer/`)

- Add keyword tokens `KW_FOR`, `KW_WHILE`, `KW_DONE`, `KW_BREAK`, `KW_CONTINUE` (`tokens.py`), and
  the lowercase→uppercase remaps so the grammar's `"for"`/… string literals resolve. (`in`/`do`/
  `until` already exist; `DO_LSQB` retagging is unchanged.)
- Add `KW_DONE` to the **continuation-rule set** in `layout.py` (currently `PIPE`, `KW_ELSE`,
  `KW_CATCH`, `KW_UNTIL`) so an explicit `done` terminator may dedent-align with the loop opener
  exactly as `until` does today.
- `break`/`continue` follow the existing keyword-recognition path; per A2 they become reserved.

#### Synthetic-`done` injection (realizes D1's "terminator optional in multi-line")

Because the grammar requires a terminator, the layout supplies one for a multi-line loop body that
omits it. The mechanism is deterministic and local to `layout.py`:

- **Tag loop-body indent levels.** When the layout emits an `_INDENT` that opens a loop body, record
  the **column of the owning `do`** alongside that level on the indent stack. A loop-body `_INDENT`
  is one emitted while the most recent significant token is the loop header's `do` or the `]` that
  closes its bound. Detection: set `pending_do_col` to the `DO` token's column when `DO` is seen;
  carry it across an optional bound by tracking bracket kinds (`DO_LSQB` opens a "bound" bracket, its
  matching `RSQB` closes it, leaving `pending_do_col` set); if any other significant token appears on
  the same line before a newline, the body is **inline** → clear `pending_do_col` (no tagging, the
  explicit terminator is mandatory). On the next `_INDENT`, if `pending_do_col` is set, tag the new
  level with it and clear it.
- **Inject on close.** Whenever an indent level is popped (both the normal dedent path and the
  `until`/`done` continuation path), for each popped **loop-body** level: if the dedent is triggered
  by an `until`/`done` token whose column equals that level's recorded `do` column, the loop is
  explicitly terminated — do nothing. Otherwise emit a synthetic `DONE` token immediately **after**
  that level's `_DEDENT` (and before any following `_NEWLINE`/terminator). This yields the grammar's
  required `… _DEDENT DONE` for that loop.

This handles nesting correctly: a single `until`/`done` terminates only the loop at its column;
deeper popped loop bodies each receive their own synthetic `done`. Worked example — inner loop has
no terminator, outer ends with `until E`:

```
do            # col 0  (outer, do_col 0)
  do          # col 2  (inner, do_col 2)
    body      # col 4
until E       # col 0  → terminates OUTER
```

Token stream produced: `DO _INDENT DO _INDENT body _DEDENT DONE _DEDENT UNTIL E` — the inner loop
gets a synthetic `done` (its `do_col` 2 ≠ the `until` column 0), the outer consumes the explicit
`until`. Inline loops never tag a level, so they are unaffected.

## 5. Frontend passes

### 5.1 AST (`src/agm/agl/syntax/nodes.py`)

Replace the `Do` node with a generalized `Loop` node; add `Break` and `Continue`:

```python
@dataclass(frozen=True, slots=True)
class Loop:
    for_var: str | None        # for-clause variable name, or None
    for_iter: Expr | None      # for-clause collection (present iff for_var is not None)
    while_cond: Expr | None    # while-clause guard, or None
    bound: Expr | None         # [n] bound, or None (unbounded)
    body: Expr                 # usually a Block
    until_cond: Expr | None    # `until E`; None means `done`/omitted (== until false)
    span; node_id

@dataclass(frozen=True, slots=True)
class Break:
    span; node_id

@dataclass(frozen=True, slots=True)
class Continue:
    span; node_id
```

Update the `Expr` union, `syntax/visitor.py::walk`, `syntax/spans.py`, and the parser transformer
that builds AST nodes from the Lark tree. Renaming `Do` → `Loop` is an internal breaking change;
update every reference (parser, scope, typecheck, lower) in the same milestone that introduces it.

### 5.2 Scope / name resolution (`src/agm/agl/scope/`)

`Loop` resolution order and scoping (honouring D2/D4):

1. Resolve `bound` and `for_iter` **in the enclosing scope** (before the `for` variable exists).
2. Open a fresh loop scope; bind `for_var` as an **immutable** binding (new/existing `BinderKind`
   for the loop variable so a `:=` to it is rejected with a precise message).
3. Resolve `while_cond`, `body`, and `until_cond` in that scope (all see `for_var`).

`break`/`continue` validation (D5): the resolver tracks a **loop-nesting flag that resets at every
`fn`/`def` boundary**. A `Break`/`Continue` resolved with the flag false is a static error
("`break`/`continue` outside a loop"). This naturally forbids crossing a lambda into an outer loop.

### 5.3 Type checking (`src/agm/agl/typecheck/`)

- `Loop : unit` (A1). The body's value is discarded.
- `for_iter` must be `list[T]`, `dict[K,V]`, or `text`; `for_var`'s type is `T`, `K`, or `text`
  respectively (D3). Any other iterable type is a type error.
- `while_cond` and `until_cond` must be `bool`.
- `bound` must be `int`.
- `Break` / `Continue` yield `BottomType` (D5) — reuse the existing `BottomType` plumbing that
  `raise` uses (assignable to any target, absorbed by branch-join like `raise`).

## 6. Execution IR & lowering

### 6.1 New IR nodes (`src/agm/agl/ir/nodes.py`, `operations.py`)

- `IrLoop(location, body)` — unconditional repeat; replaces the current
  `IrLoop(limit, body, condition, condition_source)`.
- `IrBreak(location)`, `IrContinue(location)`.
- `IrIterInit(location, kind: IterKind, collection: IrExpr)`,
  `IrIterHasNext(location, iterator: IrExpr)`,
  `IrIterNext(location, iterator: IrExpr)`; `IterKind` enum `{LIST, DICT_KEYS, TEXT}` in
  `operations.py`.

Add all to the closed `IrExpr` union and the AST→IR coverage table; mypy exhaustiveness (D4 of the
execution-IR plan) forces matching cases in the lowerer, evaluator, and `validate.py`.

### 6.2 New runtime value (`src/agm/agl/values.py`)

`IteratorValue` — internal cursor over a materialized sequence (a `list` of element values for
LIST; a snapshot `list` of keys for DICT_KEYS; the backing `text` for TEXT) plus a position index.
Never rendered, hashed for user equality, serialized, or returned to user code. `validate.py` and
the evaluator's defensive tag checks treat it as a runtime-internal tag.

### 6.3 Lowering / desugaring (`src/agm/agl/lower/lowerer.py`)

`lower_loop(Loop)` emits an `IrBlock`:

**Pre-loop (enclosing frame, once, in source order):**
- if `for_iter`: `var __it = IrIterInit(kind, lower(for_iter))` (synthetic cell symbol).
- if `bound`: `let __n = lower(bound)` and `var __count = 0` (synthetic symbols; **only when a
  bound is present** — D6).

**`IrLoop(body = IrBlock[…])`** with these items, each included **only** for the clauses present,
in this exact order (the order is load-bearing — see the trace table below):

1. `for` exhaustion: `if not IrIterHasNext(__it) => IrBreak`   *(natural exit)*
2. `for` bind: `let <for_var> = IrIterNext(__it)`   *(immutable, into the frame slot)*
3. `while` guard: `if not lower(while_cond) => IrBreak`   *(natural exit)*
4. bound check: `if __count >= __n => (if __count == 0 => IrBreak else => «raise MaxIterations»)`
5. bound count: `__count := __count + 1`
6. body: `lower(body)`   *(value discarded)*
7. `until` guard: `if lower(until_cond) => IrBreak`   *(omitted when `until_cond is None`)*

`IrContinue` re-enters at item 1 (so it re-runs the `for` advance, the guards, and the bound check —
a continued iteration still advances the iterator and still counts toward the bound). `IrBreak`
leaves the loop, which yields `UnitValue`.

**MaxIterationsExceeded raise (item 4, exhaustion branch).** This fully desugars to an ordinary
`IrRaise(IrMakeException(...))` — **no dedicated node, no evaluator special-case**, so `IrLoop` stays
a pure repeat primitive. Confirmed against the code: the `IrMakeException` evaluator arm
(`ir_interpreter.py`) allocates one `trace_id` per construction via `AutoTraceField` slots and
evaluates every other field expression, exactly as for a user `raise Exc(...)`. The lowerer emits
`IrMakeException(nominal=NominalId(PRELUDE_ID, "MaxIterationsExceeded"), fields=…)` with the
`MaxIterationsExceeded` field set (`semantics/types.py`):

- `message` → `IrRenderTemplate(["Loop exhausted after ", IrLoad(__n), " iterations"])` (preserves
  the current dynamic count);
- `trace_id` → `AutoTraceField()` (auto-allocated at construction);
- `limit` → `IrLoad(__n)`;
- `condition` → `IrConstText(<until source slice, or "false" when the terminator is `done`/omitted>)`;
- `last_condition_value` → `IrConstBool(False)` (reachable only when no exit fired);
- `metadata` → `IrConstJsonNull`.

Every node used (`IrMakeException`, `AutoTraceField`, `IrRenderTemplate`/`IrTemplateText`/
`IrTemplateValue`, `IrLoad`, `IrConstText`, `IrConstBool`, `IrConstJsonNull`, `IrRaise`) already
exists in `ir/nodes.py`.

This desugaring proves the semantics. Traces (✓ = matches D2):

| Source | Behaviour |
|---|---|
| `do[1] … until E`, E false | item4 `0>=1?`no, count1, body, E false; next iter `1>=1?`yes,count≠0 → raise ✓ |
| `do[1] … until E`, E true after 1st | body, E true → break ✓ (1 iteration) |
| `do[0] …` | item4 `0>=0?`yes, count==0 → break ✓ (0 iterations, normal) |
| `do[2] … done` | run body twice; 3rd iter `2>=2?`,count≠0 → raise ✓ |
| `for x in [1,2] do[2] done` | iters bind 1,2 then `IrIterHasNext`→false → break ✓ (no raise: exhausted first) |
| `for x in [1,2,3] do[2] done` | bind 1,2; 3rd iter `2>=2?`,count≠0 → raise ✓ (exceeds bound) |
| `do … done` (no bound) | items 4/5 absent → infinite unless `break` ✓ |

### 6.4 Evaluator (`src/agm/agl/eval/`)

- `IrLoop`: `while True: try: eval(body) except _BreakSignal: return UnitValue() except
  _ContinueSignal: continue`. Internal control signals `_BreakSignal` / `_ContinueSignal` are
  Python exceptions raised by `IrBreak` / `IrContinue`.
- `IrIterInit`/`IrIterHasNext`/`IrIterNext`: build/advance the `IteratorValue` over list / dict
  keys / text.
- **Control signals bypass `catch`.** `try`/`catch` handles `AglRaise` (AgL exception values) only;
  `_BreakSignal`/`_ContinueSignal` propagate through `try` bodies to the enclosing `IrLoop`. A
  `break` inside a `try` inside a loop exits the loop. (Covered by an explicit test.)
- Remove the old bounded/unbounded `IrLoop` arm and its `condition_source` handling (now desugared).
- Per the execution-IR frame model (D5 there): no per-iteration frames; item 2's `let` rebinds the
  same frame slot each iteration.

## 7. Documentation

- `docs/agl/reference/control-flow.md` — rewrite the loop section for the unified
  `for`/`while`/`do`/`[n]`/`until`/`done` form, the iteration/bound/scope semantics (D1–D4), and
  `break`/`continue` (D5). Keep it implementation-free.
- `docs/agl/reference/grammar.md` — update the EBNF.
- `docs/agl/reference/expressions.md` / `index.md` — note `break`/`continue` and their `BottomType`.
- `docs/arch/agl/frontend.md` — `Loop`/`Break`/`Continue` AST, scope loop-context tracking,
  iterable typing.
- `docs/arch/agl/execution.md` — single `IrLoop` kind, `IrBreak`/`IrContinue`, the iterator ops +
  `IteratorValue`, and the desugaring contract; update the AST→IR coverage table.
- `docs/commands/agl.md` — only if it references loop syntax.

## 8. Testing strategy (TDD)

Per AgL area guidance, **e2e program examples under `tests/agl/programs/` come first**. Cover, in
combination with other features:

- Each variant: `while`-only, `for`-only over list/dict/text, `for`+`while`, `do`-only; each with
  and without `[n]`; terminated by `until E`, `done`, and (multi-line) omitted.
- `break` and `continue`: in each loop kind, in nested loops (innermost target), inside `if`/`case`
  branches, and inside a `try` body (exits the loop, not caught).
- Bound semantics: `MaxIterationsExceeded` raised with correct `limit`/`condition` fields; `n ≤ 0`
  zero iterations; `for` over an empty collection; `for` that exhausts before the bound (no raise);
  `do[n] done` raising after `n`.
- `for` variable: visible in `while`/body/`until`; not after the loop; `:=` to it is a static
  error.
- `continue` advances the `for` iterator and counts toward the bound.
- `BottomType`: e.g. `let v: int = if c => break else => 5` inside a loop type-checks.
- Static errors: `break`/`continue` outside a loop; across a lambda boundary; two `for`s / two
  `while`s; `while` before `for`; inline body with no terminator; non-`bool` guard; non-`int`
  bound; non-iterable `for`.

Plus unit/golden tests: parser conflict-guard stays 0/0; `Loop`/`Break`/`Continue` AST shape;
scope/typecheck errors; golden lowering IR for the desugar (per clause combination); evaluator
behaviour incl. iterator ops and control-signal/`catch` interaction. Maintain **100% `src/`
coverage** and **100% command coverage**; tests must be concurrency-robust and never run real
agents.

## 9. Milestones (TDD; commit per milestone; `just check` green at each)

- **M1 — Grammar, lexer, layout, AST.** New keywords + `KW_DONE` continuation rule; mandatory-
  terminator `loop_expr` grammar (§4.1) and `break`/`continue` atoms; **conflict-guard test green
  (0/0)** — verified by the §4.1 spike; layout loop-body tagging + synthetic-`done` injection
  (§4.2); parser transformer builds `Loop`/`Break`/`Continue`; replace `Do`; update
  `visitor`/`spans`. Parse + layout tests (incl. nested terminator-less loops).
- **M2 — Scope.** Loop-part resolution with D2/D4 scoping; immutable `for` variable (`:=` rejected);
  loop-context tracking with lambda-boundary reset; `break`/`continue` placement errors; ≤1-of-each
  and order enforced (parser or scope). Scope tests.
- **M3 — Typecheck.** `Loop : unit`; iterable typing (list/dict/text → element/key/char);
  `while`/`until : bool`; `bound : int`; `Break`/`Continue : BottomType`. Type tests.
- **M4 — IR + lowering.** `IrLoop`/`IrBreak`/`IrContinue`/`IrIterInit`/`IrIterHasNext`/`IrIterNext`
  nodes + `IterKind` + `IteratorValue`; desugar per §6.3 (counter only when bounded; `n` once);
  resolve the MaxIterations raise mechanism; remove old `IrLoop` fields; `validate.py` + golden
  lowering tests; AST→IR coverage table + exhaustiveness updated.
- **M5 — Evaluator.** Execute the new IR; `_BreakSignal`/`_ContinueSignal`; iterator ops;
  control-signal/`catch` bypass; remove the legacy `IrLoop` arm. Evaluator unit tests + the
  `tests/agl/programs/` e2e suite green.
- **M6 — Docs & cleanup.** Update reference + arch docs + grammar.md; delete dead `Do`/old-loop
  code; final `just check` at 100% `src/` and command coverage.

## 10. Resolved design points (no open risks)

Every point that could have been a risk is settled with a concrete design and a test that locks it.

- **LALR conflict-freeness — verified.** The §4.1 grammar (mandatory terminator + `for`/`while`
  clauses + `break`/`continue` atoms) was built against the real `agl.lark` with Lark LALR +
  `AglLexer` and reports **0 shift/reduce, 0 reduce/reduce**. The optional-terminator-after-suite
  formulation (2 conflicts) is explicitly rejected. Locked by the existing conflict-guard test.
- **Terminator-optional multi-line — layout, not grammar.** Realized by deterministic synthetic-
  `done` injection (§4.2) with per-level `do`-column matching for correct nesting. Locked by layout
  tests including nested terminator-less loops.
- **MaxIterationsExceeded raise — fully desugared.** `IrRaise(IrMakeException(…))` using only
  existing nodes and the existing `AutoTraceField` auto-population (§6.3); `IrLoop` stays a pure
  repeat primitive, no dedicated node. Locked by golden lowering + the `MaxIterationsExceeded` e2e
  tests.
- **Desugar item ordering (§6.3).** The for-exhaustion / bound / count / until order encodes the D2
  edge cases (exhaust-before-bound, `n ≤ 0`, `continue` counting). Locked by the §6.3 trace table as
  golden lowering + e2e tests.
- **Control signals vs `catch`.** `break`/`continue` use internal `_BreakSignal`/`_ContinueSignal`
  that bypass `catch` and unwind to the enclosing `IrLoop`. Locked by an explicit test.
- **Reserved-word breakage (A2).** `for`/`while`/`done`/`break`/`continue` become reserved; the
  corpus and tests are grepped for identifier collisions as the first task of M1.
- **Keyword `in` reuse.** `for x in e` reuses membership `in`; the for-`in` is grammatically fixed
  right after `for NAME`, and the conflict-guard (0/0) covers `bin_in` precedence.
