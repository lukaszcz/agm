# AgL Python FFI (`extern def`) — Implementation Plan

Status: planned · Date: 2026-07-04 · **Every** design decision below is owner-approved.

This is the standalone, authoritative design and implementation plan for an FFI from AgL to
Python. An AgL module may declare body-less functions with the `extern` keyword; their
implementations live in a co-located Python file (same path as the `.agl` module, `.py`
extension). The settled decisions in §3 are authoritative.

## 1. Goal

```agl
# mylib.agl
extern def f(x: int) -> int
extern def reverse[T](xs: list[T]) -> list[T]
```

```python
# mylib.py  (co-located companion)
def f(x: int) -> int: return x + 3
def reverse(xs): return list(reversed(xs))
```

An `extern def` declares a normal, fully typed, first-class AgL function whose invocation
crosses into the companion Python module. The boundary converts values in both directions
under a fixed type mapping (§4), validates everything Python returns, and surfaces Python
failures as a catchable AgL exception.

## 2. Non-goals

- No callbacks: function-typed and agent-typed values cannot cross the boundary (D4).
- No Python-side API for raising specific AgL exceptions (only `ExternError`, D11); no
  auto-mapping of Python builtin exceptions onto AgL builtins.
- No name-mapping/mangling for AgL names that are not Python identifiers (operator names,
  `do-it!`-style names) — such externs are a static error for now (D10).
- No generated Python stubs/dataclasses for AgL types; Python sees JSON-shaped data (D2).
- No sandboxing of the companion Python code: it runs in-process with full privileges,
  gated only by the host capability (D13). This is a documented trust boundary, like `exec`.
- No `Option[T]` special case at the boundary (D3).
- No change to agent invocation, `exec`, or the codec pipeline.

## 3. Settled design decisions (authoritative)

Each was settled one-by-one with the owner.

### D1 — Core data mapping: **lossless; `decimal` ↔ `decimal.Decimal`**
`int`↔`int`, `bool`↔`bool`, `text`↔`str`, `unit`↔`None`, `list[T]`↔`list`,
`dict[T]`↔`dict[str, …]`, `json`↔JSON-shaped object (`dict`/`list`/`str`/`int`/`Decimal`/
`bool`/`None`). `decimal` maps to `decimal.Decimal` in both directions and is **never routed
through `float`**, preserving the runtime-wide exactness invariant (`runtime/serialize.py`).
AgL values are immutable, so the boundary deep-copies in both directions.

### D2 — Nominal types cross as their canonical JSON shape
Records, enums, and exceptions are allowed in extern signatures. Python receives the shape
already produced by `value_to_json_obj`: record → dict of fields, enum →
`{"$case": <variant>, …fields}`, exception → dict of fields. Returns are strictly decoded
back into typed values via the existing decode-walk machinery.

### D3 — `Option[T]`: **uniform tagged dict, no special case**
Option is an ordinary enum at the boundary: `{"$case": "None"}` / `{"$case": "Some",
"value": …}`. (A Pythonic `None`/value mapping was rejected: ambiguous for
`Option[Option[T]]`/`Option[json]` and a second rule for one type.)

### D4 — Function and agent types are banned in extern signatures
Any occurrence of a function or agent type in an extern's parameter or return types is a
static error. The FFI is a pure data boundary; callbacks/agent proxies can be lifted
later without breaking anything. (Note: a *type variable* instantiated at a function type by
a caller is fine — sealing (D5) keeps such values opaque.)

### D5 — Generics: **dynamic sealing at all type-variable positions**
Extern defs may carry type parameters. Every value at a type-variable position crosses the
boundary as a **sealed opaque handle**; seal keys are minted **per dynamic call, per type
variable**. Python may rearrange, count, and compare handles (`==`/`hash`/`repr` delegate to
the wrapped AgL value) but cannot inspect or forge them. Return positions typed at a type
variable require a handle carrying that call's seal for that variable — a stale handle
(stashed from a previous call) or a cross-variable handle fails validation. Consequences:

- One boundary contract per extern **declaration** (compiled from the declared signature);
  no per-call-site contract machinery and no monomorphization.
- Extern calls work at rigid type variables, i.e. inside other polymorphic functions — no
  restriction on call contexts.
- This dynamically enforces parametricity: a Python implementation that peeks fails the same
  way at every call site.

(Rejected alternatives: per-call-site instantiated contracts — rejects calls at rigid
variables; hybrid concrete-contracts+sealing — two mechanisms, instantiation-dependent
behavior; runtime type reps — changes the calling convention of all generic functions.)

### D6 — Signature surface: **full parity with `def`; return type required**
Extern signatures support everything a regular `def` signature does — type params, param
kinds/zones (`/`, `*`, `@…`), named-only params, and AgL default expressions (evaluated on
the AgL side before crossing). The `-> type` annotation is **mandatory**, exactly as for
`builtin def`. Owner directive: the grammar/transformer/checker handling of the body-less
signature must be **analogous to `builtin def` and reuse its code**, not copy it.

### D7 — Return validation: **strict, plus the single `int → decimal` coercion**
Returned values are validated structurally with the strict decode walk (the cast pipeline,
never the lenient agent codec). The sole coercion is accepting a Python `int` where
`decimal` is declared, mirroring AgL's own assignability. Python `bool` must be rejected
where `int`/`decimal` is declared (`bool` is a subclass of `int`). Wrong type, wrong
variant, missing/extra fields, wrong/missing seal → runtime `ExternError`.

### D8 — Companion loading: **eager at program load, fail-fast**
`foo/bar.agl` requires sibling `foo/bar.py` iff the module contains at least one extern
def. The companion is imported (executing its top-level code) during program setup, before
any evaluation; every extern is resolved to its callable up front. Missing `.py`, missing
attribute, or non-callable attribute produce a load-time diagnostic naming the module and
function — never a mid-program surprise.

### D9 — Placement: **any file-backed module**
Library modules and a file-backed entry program (`agm exec file.agl`) may declare externs.
Inline source (`-c`) and direct REPL entries reject `extern def` with a clear diagnostic
(no backing file → no companion). REPL `import` of extern-bearing modules works normally.

### D10 — Python lookup: **same name, positional call**
The companion must define a function with **exactly the AgL name**; extern names must
therefore be valid Python identifiers and not Python keywords (static error otherwise).
Arguments are passed **positionally in declaration order** — the checker already resolves
every call (named args, defaults, zones) into a complete positional vector, so Python
parameter names are unconstrained.

### D11 — Errors: **one new builtin exception, `ExternError`**
All three runtime failure classes — the Python callable raising, return-contract violations
(including seal violations), and argument-conversion failures — raise the new builtin
`ExternError`, catchable with `try` like `ExecError`. Fields: inherited `message` +
`trace_id`, plus `function: text` (the extern's name) and `python_type: text` (the Python
exception class name; empty for contract violations). Load-time problems are diagnostics
(D8), not exceptions.

### D12 — Extern functions are **fully first-class**
An extern binds a closure value like any top-level `def`: it can be passed, stored,
returned, and invoked through the indirect-call path. Extern-ness is an implementation
detail of the defining module. (The `builtin def` model — inlined host-ops, call-only — was
rejected as breaking functions-as-values uniformity.)

### D13 — Gating: **`HostCapabilities` flag, default on**
A `supports_extern` capability mirrors `supports_shell_exec`: enabled for `agm exec` /
`agm repl`, switchable off by any embedding. A disabled host rejects extern-bearing
programs at load with a clear diagnostic. No user-facing config knob.

## 4. Type mapping specification

### 4.1 AgL → Python (arguments)

| AgL type | Python argument |
|---|---|
| `int` | `int` |
| `decimal` | `decimal.Decimal` (never `float`) |
| `bool` | `bool` |
| `text` | `str` |
| `unit` | `None` |
| `json` | JSON-shaped object (`dict`/`list`/`str`/`int`/`Decimal`/`bool`/`None`) |
| `list[T]` | `list` of mapped `T` |
| `dict[T]` | `dict[str, mapped T]` |
| record `R` | `dict` of mapped fields (declaration order) |
| enum `E` | `{"$case": <variant>, …mapped fields}` |
| exception | `dict` of mapped fields |
| type variable | sealed opaque handle (§4.3) |
| function / agent | **static error in extern signatures** (D4) |

### 4.2 Python → AgL (returns) — strict decode against the declared return type

Same table read right-to-left, with exactly these tolerances and no others:

- `int` accepted where `decimal` is declared (converted exactly); `float` is **never**
  accepted anywhere.
- `bool` rejected where `int`/`decimal` is declared (subclass guard).
- Records/enums/exceptions: exact field/variant match — missing, extra, or misnamed fields
  and unknown `$case` values are errors. Decoded into proper nominal values (fields
  normalized to declaration order, as construction always does).
- `json` positions accept any JSON-shaped object (including `Decimal`); anything outside
  the closed JSON-shape domain (arbitrary objects, sealed handles, `float`) is an error.
- Type-variable positions accept only a sealed handle with this call's seal for that
  variable (§4.3).
- `unit` return: the callable's result must be `None`.

Every violation raises `ExternError` (D7, D11).

### 4.3 Sealed handles

A small runtime class (boundary-only; **not** part of the AgL `Value` union) wrapping an
AgL `Value` plus a seal token `(call instance, type-var name)`:

- `__eq__` — structural equality delegating to the wrapped values' own equality (a handle
  never equals a non-handle).
- `__hash__` — consistent with `__eq__`: a canonical structural hash for data values
  (derived from the canonical serialized form), falling back to the value's own
  identity-based hash for values whose equality is identity (closures). Implementation
  detail; must satisfy the eq/hash contract so handles work in sets/dicts.
- `__repr__` — shows the rendered AgL value (debugging aid; nothing extractable).
- No other public surface. The wrapped value is not reachable through documented API;
  reaching into private attributes is documented as undefined behavior.

Sealing composes with containers: `list[T]` crosses as a real Python `list` whose elements
are handles. Seal tokens are minted fresh per extern invocation, so handles cannot be
smuggled between calls or between type variables of one call.

## 5. Surface syntax, AST, and static passes

### 5.1 Lexer

`extern` becomes a fully reserved keyword: add `KW_EXTERN` to `lexer/tokens.py` constants
and the `KEYWORDS` set (the `GRAMMAR_TOKEN_REMAP` convention exposes it to Lark as `EXTERN`
automatically). Same reservation model as `builtin`.

### 5.2 Grammar (`grammar/agl.lark`) — mirrors `builtin_func_def`

```lark
extern_func_def: "extern" _NEWLINE? "def" name type_params? LPAR param_list? RPAR THIN_ARROW type_expr
```

Registered as a `?declaration` alternative; `private` composes the same way it does for
`func_def`/`builtin_func_def`. No body, no `EQ` — the required `THIN_ARROW type_expr` is
the grammar-level enforcement of D6.

### 5.3 AST and transformer

`FuncDef` gains `is_extern: bool = False` alongside `is_builtin`. Invariants: `is_extern`
implies `body is None` and `return_type is not None`; `is_builtin` and `is_extern` are
mutually exclusive. The transformer handler is shared with `builtin_func_def` — refactor
the existing body-less signature construction into one helper used by both (owner
directive, D6).

### 5.4 Scope

Externs are collected by the same top-level function pre-pass as ordinary `def`s (they
participate in mutual recursion and export maps; `private extern def` stays
module-private). Resolution of calls, bare-name references, and first-class use is
unchanged — an extern is just a declared function whose body list is empty.

New placement check (D9): `extern def` in a module with no backing file (inline `-c`
source, direct REPL entry) is a resolution-time error. The check reads the module's
origin (the loader records the canonical path; the REPL/`-c` paths have none).

### 5.5 Typecheck

- Signature checking reuses the `def`/`builtin def` signature path (params, kinds,
  defaults, type params as rigid vars); there is no body to check.
- New extern-specific checks:
  - name is a valid Python identifier and not a Python keyword (D10);
  - no function/agent type occurs anywhere in the parameter or return types (walk the
    checked types; type variables are permitted — D4/D5);
  - `builtin`-name collision guard applies as for ordinary defs.
- Calls to externs type exactly like calls to ordinary functions (same
  `bind_arguments`/`ArgumentBindings` machinery, same generic inference). No call-site
  restrictions (D5).
- Extern call sites are recorded like `ask`/`exec` call sites so they appear in the
  dry-run inventory (`--dry-run` lists them with the extern name and return-type label).

## 6. Companion resolution and loading

### 6.1 Resolution

The companion path is derived, not searched: the canonical path of the `.agl` file with the
suffix replaced by `.py`. Module-root ambiguity rules do not apply (the `.agl` file is
already unique). `modules/loader.py` records the companion path on `LoadedModule` for
extern-bearing modules and verifies the file exists during graph loading (early half of
D8's fail-fast).

### 6.2 Import and callable resolution — `runtime/externs.py` (new)

A new eval-free runtime service, `ExternRegistry`:

- Imports each companion exactly once via `importlib.util.spec_from_file_location`, under a
  synthetic unique module name derived from the AgL module id and path (registered in
  `sys.modules` for the import's duration; cached per canonical path thereafter). No
  `sys.path` manipulation — the companion may import installed packages absolutely.
- Resolves every declared extern to `getattr(module, name)`, requiring a callable; failures
  are load diagnostics naming the AgL module and function (second half of D8).
- Owns the per-call boundary crossing: mint seal tokens, encode arguments, invoke
  positionally, translate exceptions, decode/validate the result.

The registry is built during host-environment assembly in the pipeline — **after** all
static passes succeed (so static errors precede any Python top-level side effects) and
**before** evaluation starts. The REPL session holds one registry across entries, so a
companion imports once per session.

### 6.3 Capability gate (D13)

`capabilities.py` gains `supports_extern` (default `True`, mirroring
`supports_shell_exec`). The pipeline rejects a linked program containing externs when the
host disables it, with a diagnostic before evaluation.

## 7. IR, lowering, and evaluation

### 7.1 IR

`ExecutableProgram` gains an extern-function table keyed by `FunctionId`:
`ExternFunctionDescriptor(function_id, function_symbol, module_id, name, params
(IrFunctionParam — carries default IrExprs), param_labels, result_label, contract,
companion_path)`. The **extern boundary contract** is a new typeless artifact in
`ir/contracts.py`: an encode recipe per parameter and a strict decode walk for the return
type, both compiled from checker types at lowering. Type-variable leaves compile to
`SEAL(var)` / `UNSEAL(var)` recipe nodes; all other leaves reuse the existing conversion
vocabulary. IR validation covers the new table.

### 7.2 Lowering

An extern lowers to a descriptor + contract (no `FunctionDescriptor`, no body). Its
top-level binding is initialized exactly like other top-level functions: a closure value
referencing its `function_id`, created in the closure-initialization phase, so forward
references, exports, and first-class use all work unchanged (D12).

### 7.3 Evaluation

The direct- and indirect-call paths resolve a `function_id`; when it lands in the extern
table (instead of the ordinary function table) the interpreter delegates to the effects
layer (`eval/effects.py`), which:

1. evaluates unfilled parameter defaults (AgL expressions, in a fresh frame chained to
   module scope, as for ordinary functions);
2. calls `ExternRegistry.invoke(descriptor, args)` — encode per contract (sealing at
   type-var leaves with this call's tokens), positional Python call, strict decode of the
   result (unsealing with the same tokens);
3. wraps any Python exception, conversion failure, or contract violation in
   `AglRaise(make_builtin_exception("ExternError", …, function=…, python_type=…))`,
   following the `ExecError` pattern.

`ExternError` is added to `BUILTIN_EXCEPTIONS` in `semantics/types.py` (extending
`Exception` with `function: text`, `python_type: text`) and to the frozen name list.

## 8. Documentation

- New user reference page `docs/agl/reference/ffi.md` (declaration syntax, the §4 mapping
  table, sealed-handle contract for Python authors, error model, companion loading rules,
  trust caveat); cross-linked from `functions.md`, `modules.md`, and the sidebar/index.
- `lexical-structure.md` (new reserved keyword) and `grammar.md` (new rule).
- Architecture docs: `docs/arch/agl/frontend.md` (extern in the signature/typecheck story),
  `execution.md` (extern table, boundary contract, effects dispatch), `modules.md`
  (companion resolution), `index.md` (runtime package note); `docs/arch/index.md` only if
  the system shape summary needs it.
- `agm exec`/`agm repl` command docs: brief FFI mention + `--dry-run` inventory addition.

## 9. Testing plan (TDD; tests first per milestone)

Real companion `.py` fixtures are used freely (Python is not an agent). Multiple scenarios
per behavior; no exact-message assertions. New files, named by topic:

- `tests/test_agl_extern_syntax.py` — lexer keyword, grammar (arrow required, body
  rejected, `private`/type-params compose), transformer flags/invariants, scope collection
  and export, placement rejection for `-c` and direct REPL entries.
- `tests/test_agl_extern_typecheck.py` — signature parity (kinds/zones/defaults), required
  return type, Python-identifier/keyword name check, function/agent-type rejection (incl.
  nested occurrences), generic signatures accepted, call typing incl. inference, dry-run
  call-site recording.
- `tests/test_agl_extern_loading.py` — companion path derivation; missing `.py` / missing
  attribute / non-callable diagnostics at load, before evaluation and before Python side
  effects where applicable; import-once caching (incl. across REPL entries);
  `supports_extern=False` rejection.
- `tests/test_agl_extern_runtime.py` — round-trips for every §4 row (scalars incl. Decimal
  exactness and bool-where-int rejection, unit/None, json passthrough, lists/dicts,
  records/enums incl. `$case`, Option as plain enum); strict return validation matrix;
  defaults and named-arg binding into positional Python calls; first-class use (stored,
  passed to higher-order fn, returned, called indirectly); `ExternError` fields and
  catchability via `try`; recursion between AgL and extern-adjacent code paths.
- Sealing (in `test_agl_extern_runtime.py` or its own file if large) — parametric utilities
  (reverse/len/merge) at several instantiations incl. records and closures; handle
  `==`/`hash`/`repr`; forged value at a `T` return position rejected; cross-call stashing
  rejected; cross-type-var swap rejected; nested `list[T]`/`dict[T]` sealing.
- Multi-file fixtures under `tests/agl/` (extern-bearing library imported by entry;
  qualified/open imports; `private extern`); e2e `agm exec` runs of extern programs and
  REPL import flow.
- Dependency-contract test (`test_agl_dependencies.py`) extended for `runtime/externs.py`
  (eval-free) if the layering table enumerates modules.

Coverage gates unchanged: 100 % of `src/`, `just check` green.

## 10. Milestones

1. **Syntax + scope** — keyword, grammar, shared body-less transformer helper, AST flag,
   scope collection/exports, placement checks. (`test_agl_extern_syntax.py`)
2. **Typecheck + contracts** — extern signature checks, call typing, dry-run call sites;
   boundary-contract compilation (encode/decode/seal recipes) in lowering-facing form.
   (`test_agl_extern_typecheck.py`)
3. **Loading + registry** — loader companion resolution, `runtime/externs.py` import and
   callable resolution, capability gate, pipeline wiring. (`test_agl_extern_loading.py`)
4. **IR + lowering** — extern table, descriptors, closure initialization, IR validation.
5. **Evaluation** — call-path dispatch, boundary crossing, sealing, `ExternError`;
   end-to-end single-file and graph runs. (`test_agl_extern_runtime.py`)
6. **REPL + docs + polish** — REPL session registry, e2e/multi-file fixtures, all §8 docs,
   final `just check`.

Commit after each milestone passes the gates.

## 11. Deferred extensions (explicitly out of scope, non-breaking to add later)

- Explicit Python-name mapping clause for operator/non-identifier extern names (D10).
- A public Python helper API for raising specific AgL exceptions from companions (D11).
- Function/agent proxies (callbacks) across the boundary (D4).
- Generated typed Python stubs for AgL nominal types (D2).
