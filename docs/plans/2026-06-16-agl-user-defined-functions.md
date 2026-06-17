# Plan: User-defined functions + uniform call syntax (AgL v2)

## Overview

Today AgL has no user-defined functions. Repeated logic cannot be abstracted,
and the three call-like built-ins — `print`, `exec`, and named agent / `ask`
invocations — each have **bespoke syntax**: agent calls are juxtaposition with a
bracketed option cluster (`reviewer[on_parse_error: retry[2]] "…"`), `print` is a
dedicated statement (`print expr`), and `exec` is a contextual-keyword agent
call. The language is also statement-oriented: a closed `Stmt`/`Expr` split with
a bar-safe statement stratification baked into the grammar.

This plan introduces **statically-typed, first-class, recursive functions** and,
in doing so, reworks the language around a small set of uniform primitives. The
owner has chosen a direction that is best described as **AgL v2** — it is a
breaking redesign, not an additive feature:

1. **Expression-oriented core.** The syntactic category of *statements* is
   removed. Every former statement becomes an expression; `let x = A` is an
   expression that requires a continuation, so `let x = A; B` is a *single*
   expression (let-in). A block yields the value of its last expression.
2. **Uniform call syntax.** All calls are `callee(arg, …, name: val)`, with one
   sugar: when there is exactly one positional argument and no named arguments,
   the parentheses may be dropped (`ask "Hello?"`, `print review`). The argument
   list is conceptually a single (possibly labeled) tuple argument; there is **no
   first-class tuple type**.
3. **`print`, `exec`, `ask` become ordinary functions** (special only in their
   predefined semantics and a small built-in typing rule), invoked with the same
   syntax as user functions.
4. **Agents are first-class values** of a new `agent` type. The *only* agent
   invocation is `ask(prompt, agent: someAgent)`; bare named-agent calls
   (`reviewer "…"`) are retired. Omitting `agent:` uses the default agent.
5. **Functions are first-class and recursive.** Function values have positional
   types `(A, B) -> C`; named/optional arguments are available only when calling
   a *declared* name (a `def` or a built-in), and are erased from the value type.
6. **No generics yet**, but every interface is designed so parametric
   polymorphism can be added later without another redesign.
7. **A `unit` type** is added for side-effecting expressions (`print`, `set`, an
   `if` with no `else`, loops).

> This is a large undertaking touching every pipeline stage (lexer → grammar →
> AST → scope → typecheck → eval → runtime → REPL) plus the full reference docs
> and the entire `.agl` corpus. The Milestones section stages it so each step is
> independently green.

---

## Resolved owner decisions

These were decided directly with the owner and are **not** open for
re-litigation in review; they frame everything below.

| # | Decision |
|---|----------|
| R1 | **Expression-oriented.** Remove the statement category. All former statements are expressions. `let`/`var` bind over a continuation (`let x = A; B` is one expression). A block's value is its last expression. The program top level remains a sequence of declarations/expressions. |
| R2 | **Uniform parenthesized calls** `f(a, b, name: v)`, with single-positional-argument paren-drop sugar (`ask "Hi"`, `print x`). The sugar applies only when there are **no** named arguments. |
| R3 | **Argument list is a conceptual tuple (model "a")** — internal calling-convention only. **No first-class tuple type**: `(a, b)` appears only in argument position. |
| R4 | **Named/optional arguments live inside the parens** as `name: value` (same shape as constructors). The `[options]` bracket cluster is **retired**. |
| R5 | **`print` / `exec` / `ask` are ordinary functions** with predefined semantics and uniform call syntax. |
| R6 | **Agents are first-class values** of type `agent`. `ask(prompt, agent: a)` is the sole invocation form; bare named-agent calls are removed. `agent:` omitted ⇒ default agent. |
| R7 | **First-class, recursive functions.** Function value types are positional `(A, B) -> C`; named/optional args only at declared-name call sites, erased from the value type. |
| R8 | **No generics now**, but design for later addition. |
| R9 | **Add a `unit` type.** |

---

## Secondary decisions — resolved with the owner

Each is a real fork the headline decisions did not fully pin down. All were
walked through with the owner. **Every recommendation below was confirmed as
written except D10**, where the owner chose the structured-result-record form
for `exec` (see D10). Quick index of the confirmed choices:

| # | Resolution |
|---|------------|
| D1 | Uncurried application; currying deferred. |
| D2 | A block ending in a binder is a static error (binder needs a continuation). |
| D3 | `do…until` yields `unit`; the unit value is written `()`. |
| D4 | `def name(params) -> T = body` (expression body, return type required); lambda **`fn(p: T) (-> R)? => body`** — `fn` keyword required (keyword-free is not LALR(1)); lambda return type **optional, inferred from body**. Verified conflict-free. |
| D4b | `->` for return/function types, `=>` for lambda & branch bodies. |
| D5 | Optional arguments via **default values** only. |
| D6 | Built-in special typing rule; `ask`/`print` **not** bindable as values in v1. |
| D7 | `agent` is opaque: no equality, default agent stays implicit. |
| D8 | Top-level `def`s mutually recursive; no `let rec`; depth limit raises new `RecursionError`. |
| D9 | Rendering/encoding a function or agent value is a static error everywhere. |
| D10 | One context-typed `exec`; **`ExecResult` is the default target type** (no annotation → structured handle that does not raise on nonzero exit); a non-`ExecResult` target parses stdout into it and raises on failure (today's behavior). |
| D11 | Former call options become prelude types (`enum ParsePolicy \| Abort \| Retry(n)`, `bool`, codec-name `text`). |

The detailed alternatives/trade-offs that were presented are retained below for
the record.

### D1 — Currying

- **(a) Uncurried (recommended).** `f(a, b)` applies all arguments at once; no
  partial application. Follows directly from R3 (the argument is one tuple) and
  matches the `(A, B) -> C` value-type notation. Simplest checker/eval. The
  calling convention is designed so `f(a)(b)` currying can be layered in later.
- (b) Curried + partial application. Most FP-faithful, composes best with
  first-class functions, but subtle with optional args and most useful with
  generics (deferred). Conflicts with the tuple-argument model.

**Recommendation: (a).** Currying joins generics as an explicit "later" item.

### D2 — Block / sequencing semantics

A block (function body, branch body, or the program top level) is a sequence of
expressions separated by newlines or `;`. Binders scope over the remainder.

- **(a) Last-expression value; binder-without-continuation is an error
  (recommended).** `let`/`var`/`def` must be followed by at least one more
  expression in the block (the continuation); a block ending in a binder is a
  static error (*"a `let` must be followed by an expression"*). A block's type
  and value are those of its final expression. `;` and newline are equivalent
  separators inside a block.
- (b) Trailing binder yields `unit`. More permissive, but allows pointless
  `let x = e` blocks that compute and discard.

**Recommendation: (a)** — matches the owner's "let needs a continuation" framing.

### D3 — Former statements as expressions, and `unit`

Each former statement gets an expression type:

| Form | Value / type | Notes |
|------|--------------|-------|
| `let`/`var x = e; rest` | type of `rest` | binder over continuation |
| `set x = e` | `unit` | mutation; followed by a continuation in a block |
| `print(e)` | `unit` | |
| `if c => a` (no else) | `unit` | branch body must be `unit` |
| `if c => a \| else => b` | unified type of `a`,`b` | `int→decimal` widening, as `case` today |
| `case … of …` | unified branch type | unchanged from current `case` expression |
| `do … until c` | `unit` | a loop produces no useful value (recommended) |
| `try b catch …` | unified type of body + handlers | |
| `raise e` | bottom — assignable to any expected type | diverges; never yields |
| `pass` | `unit` | |

- **Recommendation:** introduce `unit` as a primitive type with a single value;
  the surface literal is `()` (also the empty argument list of a zero-arg call —
  consistent, since the argument tuple of a no-arg call is the empty tuple). An
  `if` whose branches disagree unless one is `unit` is a static error, exactly
  like `case` today. A `do…until` yields `unit`.
- Alternative for loop value: yield the last body value. Rejected — loops are run
  for effect and the bounded-iteration contract makes "last value" fragile.

### D4 — Function declaration and anonymous-function syntax

- **Declarations (recommended):**
  `def name(p1: T1, p2: T2 = default, …) -> RetType = body`
  where `body` is an expression (which may be a block). The `-> RetType` is
  **required** (full static typing; no return-type inference in v1). Reuses `=`
  (as `type X = …` does). `def` is a **top-level declaration**.
  - Alternative: `def name(...) -> T: <indented block>` (colon+suite like
    Python). Rejected for consistency with the expression-bodied `=` forms and
    because a block is already a valid expression after `=`.
- **Anonymous functions / lambdas — `fn(p: T, …) (-> R)? => body`** (owner-
  confirmed, revised after the M1 spike). The originally-confirmed keyword-free
  form `(p: T, …) -> R => body` is **not LALR(1)** (`( name` is ambiguous at one
  token of lookahead). Owner chose the **`fn`** keyword. Example:
  `let dbl = fn(x: int) -> int => x * 2`.
  - **The lambda return type is optional** (owner decision): when `-> R` is
    omitted, the checker **infers** it from the body (or from an expected
    function type propagated into the binding). `fn(x: int) => x * 2`,
    `fn() => 1`, `fn(x: int, y: int) => x + y` are all valid. This is a limited,
    bottom-up inference — safe because lambdas are non-recursive (D8), so a body
    never depends on its own result type. **`def` return types remain required**
    (full annotation for named, possibly-recursive declarations).
  - Lambda parameter types are **required** in v1 (no parameter inference).
  - A lambda is an ordinary expression and is how a local (block-scoped) function
    is created. As a *call argument* it works directly (`map(fn(x: int) => x*2,
    xs)`); in juxtaposition/operator position it must be parenthesized.
  - Both refinements verified conflict-free in the M1 prototype.
- **No `return` keyword** (R1): a body's value is its last expression.

### D5 — Optional arguments and defaults

- **Recommendation:** optionality is expressed by a **default value**:
  `def f(x: int, y: int = 0) -> …`. A parameter with a default may be omitted at
  a (declared-name) call site; when omitted the default expression is evaluated
  in the function's definition scope. Defaulted parameters must follow required
  ones in the declaration. At a call site, a defaulted parameter is supplied by
  name (`f(1, y: 2)`); required parameters are positional. This is what lets
  `ask(prompt, agent: …, format: …, on_parse_error: …)` use one uniform
  mechanism.
  - Alternative: nullable/absent optionals (`y: int?`) distinct from defaults.
    Rejected — adds an option/nullable type and a second optionality concept;
    defaults are sufficient and already needed for `ask`.
- **Named-argument rules** (mirror constructors): a named argument names a
  declared parameter; unknown names, duplicates, and supplying a required
  positional both positionally and by name are static errors.

### D6 — Typing `print` / `exec` / `ask` without generics

`ask`'s result type is the *expected* type from context (the established
target-type propagation), and `print` accepts a value of *any* type. Neither is
expressible by a monomorphic signature, and generics are deferred (R8).

- **(a) Built-in polymorphic-signature mechanism (recommended).** Model
  built-ins as `BuiltinFn` entries the checker knows special typing rules for —
  reusing the *existing* machinery: `ask` and `exec` take their target type from
  expected-type propagation and build an `OutputContractSpec` (`exec` adds the
  `ExecResult` default-target special case, D10); `print` is special-cased to
  accept any type and yield `unit`. Syntactically and at the
  call site they are ordinary calls (same grammar, scope, eval dispatch as user
  `def`s). The "signature" is internal; the design leaves a clear seam so these
  become ordinary *generic* signatures (`ask[T](prompt, …) -> T`,
  `print[T](T) -> unit`) once generics land — at which point the special cases
  collapse into the generic checker.
- (b) Generics now. Rejected per R8.
- (c) Keep them fully special forms outside the call grammar. Rejected — defeats
  R5 (they would not be uniform with user calls).

**Recommendation: (a).** Note the consequence: because they are not yet generic,
`print`/`ask`/`exec` **cannot be bound to a function-typed value** in v1 (`let f
= print` is a static error: their type is not yet expressible). User `def`s
(monomorphic) are fully first-class. This restriction lifts automatically when
generics arrive. Flag if you'd rather make them first-class now via a minimal
internal polymorphic value type.

### D7 — The `agent` type and the default agent

- Agents now live in the **value namespace**: an `agent NAME [= "runner"]`
  declaration introduces an immutable binding `NAME : agent`. Agent values are
  passed to `ask` via the `agent:` parameter and may be stored in `let`
  bindings, passed to functions, and held in `list[agent]`.
- The `agent` type is **opaque**: no fields, no operators except equality (open
  sub-point: support `=` on agents? recommend **no** equality in v1 — agents are
  capability handles), not JSON-shaped, not renderable (see D9).
- The default agent: `ask(prompt)` with no `agent:` uses the host default agent,
  exactly as `ask "…"` does today. **Recommendation:** keep the default implicit
  (no surface name) rather than exposing a magic `default_agent` value, to avoid
  promising a portable handle the host may not have.
- `agent` declarations stay root-only, with the same duplicate/reserved-name
  rules as the existing agent-declarations feature. Reserved call name `ask`
  cannot be declared.

### D8 — Recursion and termination

R7 allows recursion. AgL otherwise guarantees bounded termination (`do[N]`,
`MaxIterationsExceeded`).

- **Recommendation:** top-level `def`s are **mutually recursive** — collected in
  a pre-pass so every `def` is in scope for every other (and itself), like a
  letrec. Local functions (lambda bound by `let`) are **not** self-recursive in
  v1 (the binding is not in scope inside its own initializer); recommend `let`
  remains non-recursive and recursion is expressed via top-level `def`. Flag if
  you want `let rec`.
- **Termination guard:** add a runtime **call-depth limit** (configurable,
  default e.g. 256). Exceeding it raises a catchable exception. Recommend a new
  `RecursionError` (`message`, `trace_id`, `limit`) rather than overloading
  `MaxIterationsExceeded` (whose schema is loop-specific). Flag if you'd prefer
  to reuse `MaxIterationsExceeded`.

### D9 — Function and agent values: opacity

Function values and agent values are **not JSON-shaped** (like records inside a
`json` slot). Recommendation: interpolating one into a prompt/template, encoding
it via the JSON codec, or `print`ing it is a **static error** with a targeted
message (*"a function/agent value has no rendering"*). They can be bound,
passed, compared only structurally if equality is added (D7 recommends no
equality). This keeps the value model closed without inventing serialization for
callables.

### D10 — `exec` and `print` exact signatures

- `print` — accepts one positional argument of any type, yields `unit`. Keeps
  current console+trace behavior. (Named render override like `${x as bullets}`
  stays a template feature, unaffected.)
- `exec` — **one context-typed built-in with two effective forms, selected by
  the target type** (owner decision). `exec(command: text, …) -> T`, `T` from
  context exactly like `ask`, with a built-in prelude record `ExecResult`
  (`stdout: text`, `exit_code: int`, likely `stderr: text`, `timed_out: bool`)
  acting as the **default** target type. The two forms:
  - **Structured form — target is `ExecResult`** (the default when there is no
    expected type, or an explicit `: ExecResult`). Returns the record; **a
    nonzero exit does NOT raise** — the caller branches on `r.exit_code`. This is
    the new capability.
  - **Parsed form — target is any non-`ExecResult` type `T`.** Parses stdout into
    `T` (a `text` target binds stdout verbatim), honoring `format` /
    `on_parse_error` / `strict_json`, and **raises `ExecError` on a nonzero
    exit** (and a parse error on bad output) — i.e. today's behavior, unchanged.
  - This reuses the existing target-type propagation and `OutputContractSpec`
    machinery; `ExecResult` is simply special-cased in the built-in typing rule
    to mean "hand back the structured handle, do not parse, do not raise on
    nonzero." `ExecError` is retained for both the parsed form and transport
    failures (spawn failure, timeout) in either form.
  - Define `ExecResult` in the prelude alongside the D11 types; the existing
    `ExecError` schema (`command`, `exit_code`, `stdout`, `stderr`, `timed_out`)
    is the natural shape to mirror for its fields.
  - **Open sub-point to confirm during implementation:** what `exec`'s *no
    expected type* default should be. Recommended: `ExecResult` (so a bare `let r
    = exec "…"` gives the structured handle). This changes today's untyped-`exec`
    default (which was `text`); flag at implementation if the owner prefers the
    untyped default to stay `text`.
- `ask` — `ask(prompt: text, agent: agent = «default», format: «codec» = «auto»,
  strict_json: bool = «host default», on_parse_error: ParsePolicy = Abort) -> T`,
  `T` from context. The former `CallOptions` become named parameters typed via
  the D11 prelude types. `ask` remains the context-typed polymorphic built-in
  (D6).

### D11 — Types of the option-valued arguments

The former call options had ad-hoc value grammars (`format: json`,
`on_parse_error: retry[2]`). As named arguments they need types.

- **Recommendation:** introduce small built-in **enums** in a prelude:
  `enum ParsePolicy | Abort | Retry(n: int)` and represent `format` as `text`
  (codec name, validated against registered codecs by the built-in typing rule,
  as today). `strict_json: bool`. This reuses the existing enum machinery and
  keeps `ask` a normal typed signature. `retry[2]` becomes `Retry(n: 2)` or the
  prelude exposes `retry` as a constructor.
  - Alternative: keep bespoke literal grammars for these arguments. Rejected —
    reintroduces special syntax that R4/R5 remove.
- Flag: this is the least-settled corner; an alternative is to keep `format`/
  `on_parse_error`/`strict_json` as host-recognized keyword arguments with
  validation but no first-class enum. Recommend the prelude-enum approach for
  uniformity.

---

## Target language design (concrete)

### Syntax examples

```agl
record Issue
  title: text
  severity: int

enum Review
  | Pass
  | Fail(issues: list[text])

agent reviewer
agent planner = "claude -p %{PROMPT_FILE}"

# user function — expression body
def classify(n: int) -> text =
  if n > 0 => "pos"
  | n < 0  => "neg"
  | else   => "zero"

# block body: let-continuation, last expression is the result
def summarize(doc: text, limit: int = 3) -> text =
  let head = ask "Summarize: ${doc}"
  let tagged = "[${limit}] ${head}"
  tagged

# recursion (top-level def, mutually recursive set)
def fact(n: int) -> int =
  if n <= 1 => 1 | else => n * fact(n - 1)

# uniform calls
print review                                  # one arg, sugar
print(classify(-4))                           # compound arg ⇒ parens
let s = ask "Hello?"                           # default agent
let r: Review = ask("Review ${a}", agent: reviewer, on_parse_error: Retry(n: 2))
let res = exec "ls -la"                        # res : ExecResult (default; never raises)
print(res.stdout)
if res.exit_code != 0 => print("command failed")
let out: text = exec "ls -la"                  # parsed form: stdout verbatim; raises on nonzero
let g: (int) -> text = classify               # first-class function value
let label = g(7)                               # positional call of a value
```

### Grammar shape — proven by the Milestone 1 spike

The expression-oriented grammar is **de-risked**: a complete prototype lives at
`prototypes/agl_v2/` (grammar `agl_v2.lark`, conflict+corpus harness `check.py`,
findings `README.md`) and builds under Lark `parser="lalr"` with **0 shift/reduce
and 0 reduce/reduce conflicts**, parsing a 54-program positive corpus and
rejecting the negative cases (`f a b`, `ask[opts] "…"`). It is the reference for
porting into production. Proven structure:

- `start: block`; `block` is a `_NEWLINE`/`;`-separated sequence of *items*
  (declarations, binders, expressions); binders scope over the following items;
  the block's value is its last item.
- `?expr` puts the "greedy" open forms (`if`/`case`/`do`/`try`/`raise`/lambda) at
  the top, then the operator chain `or → and → not → comparison → additive →
  multiplicative → unary(-) → juxt → postfix → atom`.
- **Uniform calls.** `postfix` carries paren-calls `f(args)` and field access
  `f.field` (left-assoc). A **separate, non-chaining** `juxt: postfix juxt_arg`
  is the single-arg sugar, where `juxt_arg` is a *restricted atom* that **excludes
  `(`-led forms** (so `f(x)` is always the paren-call) and **excludes unary `-`**
  (so `f -1` is subtraction). Application binds tighter than operators (`print x +
  1` ≡ `(print x) + 1`). `juxt` must be a *concrete* (non-`?`) rule — making it
  transparent cascades S/R conflicts into every operator rule.
- `arg_list: arg ("," arg)* ","?`; `arg: named_arg | pos_arg`, reusing the
  constructor named-arg shape. `()` is both the unit literal and the empty arg
  list (`paren_expr_or_unit: LPAR RPAR -> unit_lit | LPAR expr RPAR -> paren_expr`).
- New type forms: `func_type: "(" type_list? ")" "->" type_expr` and the `unit`
  primitive (non-reserved type keyword).
- **Bar-stratification fully removed.** The 3-tier `closed_stmt`/`open_stmt`/
  `bar_*` system is gone; branch/loop/condition bodies use `or_expr` directly. A
  *named* alias (`closed_expr: or_expr`) reintroduces conflicts via a FOLLOW-set
  cycle, so branch positions must reference `or_expr` directly (matching the old
  transparent `?bar_expr`). This is a real simplification win.
- `print_stmt`, the `agent_call` option-cluster, and `pass` are removed (`pass`'s
  role is taken by `()`). `print`/`pass` are no longer reserved; `def`/`fn` are
  added as keywords, plus a `->` token distinct from `=>`.

**Design points the spike settled** (details in §"Milestone 1 results"): lambdas
use the `fn` keyword with an optional (inferred) return type; the single-arg
sugar covers simple atoms *and* field-access paths (`print res.stdout` works),
but a *call* result as the lone argument still needs parens
(`print(classify(x))`). Both verified conflict-free.

### Type system additions

- `UnitType` (kind `"unit"`), with a single value.
- `FunctionType(params: tuple[Type, …], result: Type)` — positional only.
  Assignability is by exact structure (no variance in v1). Designed to later
  carry type parameters.
- `AgentType` (kind `"agent"`), opaque.
- Built-in function table consumed by the checker (D6): name → typing rule.
  `ask`/`exec` reuse the existing target-type + `OutputContractSpec` path;
  `print` is the any→unit rule.
- None of `FunctionType`/`AgentType`/`UnitType` are JSON-shaped (D9); the
  `is_json_shaped`/codec selection code rejects them.

### Evaluation model

- A `Closure` value: captured definition environment + parameter list (with
  default expressions) + body expression. Top-level `def`s are installed into the
  root environment in a pre-pass (mutual recursion).
- Call evaluation: evaluate positional args, bind named/defaulted params
  (defaults evaluated in the closure's captured scope), open a call scope, eval
  the body, return its value. Enforce the call-depth limit (D8).
- Built-ins (`print`/`exec`/`ask`) dispatch to the current interpreter routines,
  now reached through the unified call node rather than `PrintStmt`/`AgentCall`.
- Agent values resolve to the host agent registry entry; `ask` with no `agent:`
  uses the default agent. The runtime reconciliation (declared vs backed agents)
  carries over from the agent-declarations feature.

---

## Per-component implementation impact

1. **Lexer (`lexer/`).** Add `def` keyword (reserved). Add `unit` as a
   non-reserved type keyword (like `text`). Add `->` … already `ARROW` is `=>`;
   need a distinct token for the function-type/return arrow. **Decision needed
   (fold into D4):** reuse `=>` for both lambda bodies and `->` return types, or
   add a `->` token. Recommend adding `->` (`THIN_ARROW`) for return/function
   types and keeping `=>` for branch/lambda bodies — they read differently and
   avoid ambiguity. Retire the `LOOP_BOUND` `[N]` lexer merge only if `do[N]`
   syntax changes (it need not). The `[options]` disambiguation for agent calls
   is removed.
2. **Grammar (`grammar/agl.lark`).** The largest change: expression-oriented
   rewrite (see sketch). Re-verify 0 shift/reduce, 0 reduce/reduce.
3. **AST (`syntax/nodes.py`).** Add `FuncDef`, `Lambda`, `Param` (name, type,
   default), `Call` (callee: Expr, args: tuple of positional Expr + named
   `(name, Expr)`), `UnitLit`. Remove/repurpose `PrintStmt` and `AgentCall`
   (folded into `Call` + built-in resolution) and `CallOptions` (named args
   subsume it). Collapse the `Stmt` union into `Expr` (R1): `LetDecl` etc. gain a
   `body`/continuation field or blocks are modeled as `Block(items)` with binder
   scoping. Add `FunctionT`, `UnitT`, `AgentT` to `syntax/types.py`.
4. **Scope (`scope/`).** Functions and agents are value bindings. Pre-pass
   collects top-level `def`s (mutual recursion) and `agent` decls. Resolve
   `Call.callee` as an ordinary expression; classify built-ins (`print`/`exec`/
   `ask`) by name in call position (extend `CallKind`, or a new built-in table).
   Parameter names bind in the function body scope. `let`-continuation scoping
   replaces statement-sequence scoping.
5. **Typecheck (`typecheck/`).** Add `FunctionType`/`UnitType`/`AgentType` and
   assignability. Check function declarations (param types, default-value types,
   body type vs `-> RetType`), calls (arity, positional/named matching, defaults,
   widening), and the built-in typing rules (D6) reusing the existing
   target-type/`OutputContractSpec` logic for `ask`/`exec`. Expected-type
   propagation extends to call arguments (positional → param type; named → param
   type) and to function bodies (`-> RetType` propagates in).
6. **Eval (`eval/`).** Add `Closure`/`UnitValue`/agent values; unify call
   dispatch; install top-level `def`s; enforce call-depth limit; route
   `print`/`exec`/`ask` through the unified node. `do_until`/`if`/`try` now
   produce values.
7. **Runtime (`runtime/`).** Agent reconciliation stays; `ask`/`exec` contract
   building is reached via the new call path. Trace records for `print` and agent
   calls unchanged in content. Prelude (`ParsePolicy` enum etc., D11) registered
   as built-in types.
8. **REPL (`repl/`).** A bare trailing expression still echoes its value
   (already supported via `EchoInterpreter`); `print`/agent calls flow through
   the new call node. Session promotion/rollback semantics unchanged.

---

## Documentation updates

All under `docs/agl/reference/` plus `docs/agl-grammar.md`, `docs/arch/agl.md`,
`README.md`, `docs/commands.md`:

- **New page `functions.md`** — `def`, lambdas, optional args/defaults, recursion,
  function values & types, the depth limit.
- **`agent-calls.md`** — rewrite around `ask(prompt, agent: …, …)`; agents as
  values; remove bracket-cluster and bare named-agent syntax.
- **`expressions.md` / `program-structure.md` / `control-flow.md`** — the
  expression-oriented model, blocks/let-continuation, `unit`, every former
  statement's value, uniform call syntax + single-arg sugar.
- **`types.md`** — `unit`, function types, `agent` type, opacity rules.
- **`shell-execution.md`** — `exec` as a function.
- **`bindings-and-scope.md`** — agents/functions as value bindings; mutual
  recursion; lambda scoping.
- **`lexical-structure.md` / `docs/agl-grammar.md`** — new tokens, precedence
  (application binds tightest), removed option-cluster/bar stratification.
- **`docs/arch/agl.md`** — updated pipeline notes (statement category removed;
  built-in call table; closure model).

(The reference must not mention the implementation — keep it language-only.)

## Test plan (TDD)

Write failing tests first; add regression tests; hold 100% `src/` coverage.

- **Lexer:** `def`/`unit`/`->` tokens; `def` reserved; `unit` usable as a type.
- **Parser:** `def` with required/defaulted params; lambda; `Call` with
  positional + named args; single-arg sugar (`ask "x"`, `print y`); paren-grouped
  compound args; function types; `let`-continuation blocks; the conflict-guard
  regression (0/0).
- **Scope:** functions/agents as value bindings; mutual recursion in scope;
  unknown callee; param shadowing; agent passed as value; built-in name
  classification.
- **Typecheck:** arity/named/default checking; widening in args; body vs
  `-> RetType`; function-value assignability; `unit` propagation; `ask` target
  typing via context; `print` any→unit; rejection of `let f = print` (D6);
  opacity rejections (D9); option-arg enum typing (D11).
- **Eval:** closures, recursion + depth-limit exception; defaults evaluated in
  def scope; `if`-no-else/`do`/`set`/`raise` values; `ask`/`exec`/`print` through
  the unified path; agent value dispatch + default agent.
- **Runtime/REPL:** agent reconciliation unchanged; trace content stable; REPL
  echo of expression values.
- **E2E (`tests/agl/`):** migrate the entire corpus (below); add multi-scenario
  programs exercising recursion, defaults, function values, and `ask(agent:)`
  with multiple input/mock combinations; new rejection programs for each new
  static error.

## Migration impact

- **Every `.agl` program** that uses agent calls migrates from `reviewer "…"` /
  `ask "…" [opts]` to `ask("…", agent: reviewer, …)` (single-arg `ask "…"`
  survives unchanged). `print x` survives via the sugar.
- The `Stmt`-oriented grammar/AST/passes are substantially rewritten; this is the
  bulk of the work and the main source of risk.
- The reference docs (≈12 files) and `docs/arch/agl.md` are rewritten/extended.
- Host API: `CallOptions` removed; built-in call table added; `agent` becomes a
  value type. The agent-declarations reconciliation logic is preserved.

## Milestone 1 results (de-risking spike)

A standalone grammar spike (`prototypes/agl_v2/` — kept as the reference for the
production port, not scratch) **proved the central feasibility question**: an
expression-oriented AgL grammar with uniform parenthesized calls and single-arg
juxtaposition sugar builds under Lark `parser="lalr"` with **0 shift/reduce and 0
reduce/reduce conflicts**, parses a 54-program positive corpus, and rejects the
intended negatives. Reproduce: `uv run python prototypes/agl_v2/check.py`.

**What was confirmed feasible (no change to the plan):**
- Single-arg juxtaposition (`ask "hi"`, `print x`, `f 5`) coexisting with
  paren-calls (`f(a, b)`), field access, and the full operator chain — with
  application binding tightest (`print x + 1` ≡ `(print x) + 1`, `f -1` =
  subtraction, `f (x+1)` = a call).
- Dropping the entire bar-safe statement stratification — a real simplification.
- `()` serving as both the unit literal and the empty argument list.

**Two grammar changes — both resolved with the owner and re-verified
conflict-free in the prototype:**

1. **Lambdas use the `fn` keyword (overrides D4).** Keyword-free `(p: T) -> R =>
   body` is not LALR(1) (`( name` ambiguous). Owner chose `fn`. The lambda return
   type is **optional** (`fn(x: int) => x*2`), inferred from the body; `def`
   return types stay required. `lambda_expr: "fn" "(" params? ")" ("->" type)?
   "=>" expr` — verified 0/0.
2. **Single-arg sugar covers simple atoms *and* field-access paths.** `print x`,
   `ask "…"`, and now `print res.stdout` / `print a.b.c` all parse bare (`.` binds
   tighter than juxtaposition; `DOT ∉ FOLLOW(juxt)`, so no conflict). A **call**
   result as the lone argument still needs parens — `print(classify(x))` — and
   `print classify(x)` is a parse error (a deliberate negative case). Verified 0/0
   with `print res.stdout` parsing as `juxt_call(print, field_access(res, stdout))`.

**Carried into the production port (known, low-risk):**
- The standalone prototype handles only single-line `if`/`case` branches; the
  production custom lexer already suppresses `_NEWLINE` before `|`/`catch`/`until`,
  so multi-line branches work once ported.
- The prototype models strings as a plain terminal; template interpolation is
  orthogonal and re-attached from the existing template sub-scanner unchanged.
- `LOOP_BOUND` (`[N]`) stays a single lexer token, as in production today.

## Risks & mitigations

- **Scope creep / "two features at once."** Expression-orientation and functions
  are entangled (functions need value-producing bodies). *Mitigation:* stage via
  Milestones; land the expression-oriented core (M2) green before functions (M4).
- **LALR(1) conflicts from juxtaposition application.** *Resolved by the M1
  spike* (`prototypes/agl_v2/`, 0/0 conflicts): the sugar's argument is a
  restricted atom (no `(`-led, no unary `-`) and `juxt` is a concrete
  non-chaining rule. Residual: the standalone prototype only handles *single-line*
  `if`/`case` branches; multi-line branches rely on the production custom lexer's
  existing `|`/`catch`/`until`-continuation `_NEWLINE` suppression, which the
  production port inherits for free.
- **Corpus + docs churn.** *Mitigation:* mechanical migration sweep in one
  milestone behind green passes; keep the e2e gate green throughout.
- **`ask`/`print` typing without generics feels special-cased.** *Mitigation:*
  D6's built-in table is explicitly the pre-generics seam; document the future
  collapse into generic signatures.
- **Unbounded recursion.** *Mitigation:* runtime depth limit (D8).

## Milestones

Each milestone ends green (`just check`) and is committed (per repo policy).

1. **Grammar/lexer prototype** — ✅ **DE-RISKED** (see "Milestone 1 results"
   below and `prototypes/agl_v2/`). Conflict-free expression-oriented grammar with
   uniform calls proven. Remaining M1 work to land in production: port the proven
   grammar to the `%declare` + custom-lexer setup (adding `def`/`fn`/`->` tokens,
   `|`-continuation already handled, template interpolation re-attached); build the
   v2 AST nodes + transformer; parser tests + conflict guard. No semantics yet.
2. **Expression-oriented core** — scope + typecheck + eval reworked so existing
   constructs are expressions (let-continuation, `unit`, value-producing
   `if`/`case`/`do`/`try`); migrate corpus to the new core *without* functions
   (agent calls still in transitional form if needed). Green.
3. **Uniform calls + agents as values** — `Call` node end-to-end; `print`/`exec`/
   `ask` as built-ins; `agent` type + `ask(agent:)`; retire bracket cluster and
   bare agent calls; migrate corpus agent calls. Green.
4. **User functions** — `def`/lambda, params/defaults, function values & types,
   typecheck + closures + dispatch. Green.
5. **Recursion + depth limit + prelude** — mutual recursion, `RecursionError`,
   `ParsePolicy`/option-arg types (D11). Green.
6. **Docs + final corpus/e2e sweep** — full reference rewrite, `docs/arch`,
   `README`, `commands.md`; new rejection programs; multi-scenario e2e. `just
   check` green.
```
