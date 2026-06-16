# AgL v2 Grammar — De-Risking Spike

**Status:** 0/0 LALR(1) conflicts. Full corpus passes. Exit 0.

**Refinements applied (2026-06-16):**
- Refinement 1: Optional lambda return type — `fn(x: int) => x * 2` now valid.
- Refinement 2: Field-access in juxt sugar — `print res.stdout` now valid.

**Reproduce:**
```
uv run python prototypes/agl_v2/check.py
```

---

## Goal

Prove that an expression-oriented grammar with uniform function-call syntax is
LALR(1) conflict-free in Lark, and that it can parse a representative corpus of
AgL v2 programs. This is a standalone prototype — no production code was modified.

---

## Conflict Status

**0 shift/reduce, 0 reduce/reduce conflicts.**

The grammar `agl_v2.lark` builds with `parser="lalr"`, `lexer="contextual"` and the
standard `Indenter` postlex. Lark's DEBUG log is captured and asserted clean in
`check.py` (same mechanism as `TestConflictGuard` in `tests/test_agl_parser.py`).

---

## Whether Bar-Stratification Was Dropped

**Partially dropped: 3-tier → 1-tier.**

The production grammar uses a 3-tier stratification:
- `closed_stmt` — self-contained statements
- `open_stmt` — if/case/try (extends open to the right)
- `bar_closed_stmt` — bar-safe twin of closed_stmt for branch bodies

For v2, all constructs are expressions, so the statement taxonomy is gone.
The residual split is:

| Position | Type used | Why |
|---|---|---|
| Branch bodies (if/case/try), loop condition, catch body | `or_expr` | Cannot consume `\|` or `=>` or `until` |
| Binder RHS (`let x = ...`) | `expr` | Can be open (if/case/do/try/lambda) |
| Func body inline | `expr` | Can be open |
| Func body suite | `suite_expr` | Indented block |
| Param default value | `or_expr` | Inside `(...)`, COMMA terminates |

The key insight: **do NOT introduce a named `closed_expr: or_expr` alias**.
That creates a FOLLOW-set cycle:

1. `or_expr: or_expr KW_OR and_expr` → `KW_OR ∈ FOLLOW(or_expr)`
2. `closed_expr: or_expr` → `KW_OR ∈ FOLLOW(closed_expr)` (propagated)
3. `if_branch: closed_expr FAT_ARROW closed_expr` → `KW_OR ∈ FOLLOW(closed_expr)`
   (it already is, but now the parser has *two* closed_expr items, each with KW_OR in FOLLOW)
4. After reducing `or_expr → closed_expr`, seeing `KW_OR` is ambiguous:
   - Shift KW_OR to continue `or_expr KW_OR and_expr`
   - Reduce `closed_expr` (KW_OR ∈ FOLLOW(closed_expr))
   → S/R conflict on every operator-chain rule

The production grammar avoids this with `?bar_expr: or_expr` (transparent `?` prefix),
which means `bar_expr` is inlined everywhere and has no separate FOLLOW set.
For v2 we do the same: use `or_expr` directly in branch positions (no alias).

**Result: the bar-stratification is dropped from 3 tiers to zero named tiers.
Branch bodies use `or_expr` directly (not a named intermediate).**

---

## Whether Keyword-Free Lambda Worked

**No. `fn` keyword is required.**

The design requested a keyword-free lambda `(params) -> R => body`. The
investigation confirms this is NOT LALR(1)-compatible:

After `(name`, the parser cannot distinguish:
- `(name: type ...)` — a parameter list for a lambda
- `(name op ...)` — a parenthesized expression

with only 1 token of lookahead (`:` vs operator — but `=` is both an operator
and a type annotation in different contexts).

**Resolution: `fn(params) -> R => body`** using a distinct keyword `fn`.
The `fn` keyword provides the 1-token lookahead needed to commit to lambda mode.

---

## Refinement 1 — Optional Lambda Return Type

**Status: LANDED, 0/0 conflicts.**

The `-> RetType` part of `fn` lambdas is now optional:

```
lambda_expr: KW_FN LPAR param_list? RPAR (THIN_ARROW type_expr)? FAT_ARROW expr
```

Both forms are legal:
```
let dbl = fn(x: int) -> int => x * 2   // explicit return type
let dbl = fn(x: int) => x * 2           // return type omitted — inferred later
let f   = fn() => 1                     // zero-param, return type omitted
let add = fn(x: int, y: int) => x + y  // multi-param, return type omitted
```

**NOTE: `def` return types remain REQUIRED.** Only `fn` lambdas may omit the
return type. `def f(x: int) -> int = ...` cannot omit the `-> int`.

**No LALR(1) conflict:** when `-> type` is absent, `FAT_ARROW` immediately
follows `RPAR`. After reducing `(param_list? RPAR)`, the lookahead `FAT_ARROW`
vs `THIN_ARROW` is a single-token distinguisher — no shift/reduce conflict arises.

---

## Refinement 2 — Field Access in Juxt Sugar

**Status: LANDED, 0/0 conflicts.**

`juxt_arg` now supports an optional dotted field-access tail, so a single-dot
or multi-dot path can serve as the lone juxtaposed argument:

```
print res.stdout    // juxt_call( var_ref('print'), juxt_field_access(res, stdout) )
print a.b.c         // juxt_call( var_ref('print'), juxt_field_access(a, b, c) )
print x             // juxt_call( var_ref('print'), juxt_bare(var_ref('x')) )    — still works
```

Call results are still **excluded** from the sugar:
```
print classify(x)   // PARSE ERROR — must write print(classify(x))
```

**Grammar:**
```
juxt_arg: juxt_atom (DOT field_name)+ -> juxt_field_access
        | juxt_atom                   -> juxt_bare

?juxt_atom: lit_int | lit_decimal | lit_true | lit_false | lit_null
          | lit_string | var_ref | constructor | lit_list | lit_dict
```

**Why it stays conflict-free:** DOT is not in FOLLOW(juxt). FOLLOW(juxt)
consists of binary operator tokens (`+`, `-`, `*`, `/`, comparisons, `KW_OR`,
`KW_AND`, `KW_IS`, `KW_IN`, `FAT_ARROW`, `PIPE`, `_NEWLINE`, `SEMICOLON`,
`RPAR`, `RSQB`, `RBRACE`, etc.) — none of which is DOT. So after reducing a
`juxt_atom` inside a `juxt_arg`, seeing DOT unambiguously means "shift to
extend the dotted path" — no shift/reduce conflict.

---

## Lambda Position: Top-Level, Not Atom

The most important LALR(1) constraint discovered in this spike:

**`lambda_expr` must be at the TOP of the `expr` chain, not in `atom`.**

If `lambda_expr` is placed in `atom` (sub-expression position), its body expression
creates a FOLLOW-set conflict. Example: after parsing `fn(x: T) -> T => x`, the
parser has an `or_expr` for the body. It sees `KW_OR` and must decide:
- Shift (to continue `or_expr KW_OR and_expr` — make `x or y` the body)
- Reduce (to complete lambda_expr — `x` is the body, `or y` follows outside)

Since `lambda_expr` in `atom` can be followed by binary operators (as left
operand of `or`, `+`, etc.), `KW_OR ∈ FOLLOW(lambda body)`, creating an
irresolvable S/R conflict.

**Fix:** `lambda_expr` is a top-level `expr` alternative alongside `if_expr`,
`case_expr`, etc. Once `KW_FN` is shifted, the parser commits to lambda mode.
The body is `expr` (full expression), so lambda bodies can contain if/case/do/try.

**Consequence:** A lambda used as a function argument must be parenthesized:
```
// OK: lambda as a let binding
let dbl = fn(x: int) -> int => x * 2

// OK: lambda passed as arg (parenthesized)
map(fn(x: int) -> int => x * 2, lst)
```

This is the same constraint as ML/Haskell `fun`/`\` expressions.

---

## Juxtaposition (Single-Arg Sugar): Design and Constraints

**`juxt` must be a CONCRETE (non-transparent) non-terminal.**

Making `juxt` transparent (`?juxt`) cascades S/R conflicts into all operator-chain rules.
Root cause: a transparent `juxt` is inlined into `mul`, `additive`, etc., and the
FIRST sets of `juxt_arg` tokens appear directly in the operator-chain FOLLOW contexts,
creating irresolvable shifts.

**Resolution:** `juxt` is concrete. The grammar is:
```
juxt: postfix juxt_arg -> juxt_call
    | postfix
```

The `postfix`-only alternative produces a `juxt` wrapper node even for a single
`postfix` (no applied argument). This means `bin_sub(juxt(var_ref('f')), ...)` is
the shape for `f - 1`, not `bin_sub(var_ref('f'), ...)`. The check script's shape
assertion accounts for this.

**Juxt restrictions:**
- `juxt_arg` excludes LPAR-led forms → `f(x)` is always a paren-call, not juxt
- `juxt_arg` excludes unary-minus → `f -1` is subtraction `f - 1`, not `f(-1)`
- Does NOT chain → `f a b` is a parse error (correctly rejected)

**Multi-line if/case branches:** When if/case branches span multiple lines inside an
indented suite, each `_NEWLINE` is consumed as a block separator, breaking the branch
chain. The prototype does not implement the production's "|-continuation layout rule"
(which suppresses `_NEWLINE`/`_INDENT`/`_DEDENT` before `|`). For the spike corpus,
multi-line if/case branches are written on a single line.

---

## loop_bound and lit_int Conflict

`loop_bound: LSQB INT RSQB` conflicted with `lit_int: INT` because after `do [N`,
the parser saw RSQB and had to decide:
- Reduce `INT` to `lit_int` (as if starting a list literal `[N, ...]`)
- Shift RSQB to complete `loop_bound`

**Fix:** `LOOP_BOUND: /\[[0-9]+\]/` — a single terminal matching `[N]` as one token,
so it never competes with `INT`. The production grammar handles this via the custom
lexer's `DO LSQB INT RSQB → DO LOOP_BOUND` merge.

---

## Forced Syntax Adjustments Summary

| Feature | Original Spec | Actual Grammar | Reason |
|---|---|---|---|
| Lambda | `(params) -> R => body` (no keyword) | `fn(params) -> R => body` | LALR(1) impossible without leading keyword |
| Lambda position | Atom/sub-expression | Top-level `expr` | FOLLOW-set cycle in operator chain |
| `juxt` non-terminal | Transparent (`?`) | Concrete (no `?`) | Transparent cascades S/R conflicts |
| Bar-stratification | 3-tier | 0 named tiers (use `or_expr` directly) | Named alias creates FOLLOW cycle |
| Multi-line if/case branches | Natural indentation | Single-line only (in prototype) | `_NEWLINE` suppression needs custom lexer |
| `loop_bound: [N]` | `LSQB INT RSQB` | `LOOP_BOUND: /\[[0-9]+\]/` single token | S/R conflict with `lit_int: INT` |
| `print res.stdout` (juxt) | Field access as juxt arg | Now supported: `print res.stdout` is `juxt_call(print, juxt_field_access(res, stdout))` | DOT ∉ FOLLOW(juxt) — shift always wins |
| `pass` keyword | Removed (use `()`) | `()` is `unit_lit` | As specified |

---

## Simplifications (Noted)

1. **String literals**: simple regex `/"[^"]*"/` — no interpolation (`${...}`).
2. **Indentation**: Lark's standard `Indenter` postlex (not the production custom lexer).
3. **`do [N]` loop bound**: single `LOOP_BOUND` terminal — no multi-digit or complex form.
4. **Multi-line if/case branches**: require single-line in indented suites (no `_NEWLINE` suppression before `|`).
5. **Agent calls**: `ask "prompt"` works as juxtaposition; complex forms use paren-calls.

---

## Residual Risks for v2 Implementation

1. **Multi-line if/case branch continuation**: the production custom lexer needs to
   suppress `_NEWLINE`/`_INDENT`/`_DEDENT` before `|` continuation lines.
   This is non-trivial but the production lexer already does it for `enum_def`.
   The grammar is ready; the lexer extension is the work remaining.

2. **String interpolation templates**: the prototype uses plain strings. The production
   grammar's `TEMPLATE_START/INTERP_START/etc.` token approach is orthogonal to the
   expression structure and can be ported directly.

3. **Lambda as argument**: requiring parens for lambda-as-argument may surprise users
   who expect `map(fn(x: int) -> int => x, lst)` to work naturally — it does
   (parens around the lambda body are not required; parens around the whole lambda
   expression are required when it appears as a sub-expression).

4. **Type inference for lambdas**: the grammar now supports omitting the return type
   on `fn` lambdas (`fn(x: int) => ...`). The omitted return type is left for the
   type checker to infer. `def` return types remain required.

5. **`func_body: suite_expr | expr`**: the choice relies on `suite_expr` starting with
   `_NEWLINE` and `expr` not starting with `_NEWLINE`. This is currently safe but
   if any `expr` form ever starts with `_NEWLINE`, a conflict would arise.

---

## Grammar File

See `agl_v2.lark` in this directory.

Key structural choices:
- Keywords as explicit terminal strings (`KW_LET: "let"` etc.)
- `VAR_NAME` regex excludes all keywords via negative lookahead
- `DECIMAL` before `INT` in terminal order (prefix shadowing)
- `LOOP_BOUND: /\[[0-9]+\]/` single token for `do [N]`
- `juxt: postfix juxt_arg -> juxt_call | postfix` — concrete, not transparent
- `juxt_arg: juxt_atom (DOT field_name)+ -> juxt_field_access | juxt_atom -> juxt_bare`
  (field-access tail allowed; paren-call and unary-minus still excluded)
- `lambda_expr: KW_FN LPAR param_list? RPAR (THIN_ARROW type_expr)? FAT_ARROW expr`
  at top of `expr` chain; `-> type` is optional for `fn` lambdas
- Branch bodies: `or_expr` directly (no named alias)
