# Plan: AgL-native value rendering (display data structures as they are defined)

## Overview

AgL currently renders every structured value (record, enum, exception, `list`,
`dict`, `json`) as **pretty-printed JSON** when it is printed, interpolated into
a template, or cast to `text`. For example:

```agl
record R
  x: int

let r: R = R(x: 1)
print(r)              # today → {"x": 1}
```

We want structured values to render **in the AgL syntax used to define them**:

```agl
print(r)              # new  → R(x: 1)
print(r as json)      # JSON still reachable via explicit cast → pretty multi-line {"x": 1}
```

JSON output becomes an explicit opt-in through `as json`: casting any data
value, including a record, enum, or exception, to `json` structurally serializes
it into a `json` value, which still renders as JSON. This deliberately expands
the cast matrix in `docs/plans/2026-06-19-agl-casts.md`, where nominal values are
currently rejected as `json` sources. The resulting rule is universal: default
rendering uses AgL syntax, while `as json` requests structural JSON.

This change is centered on the runtime renderer
(`src/agm/agl/runtime/render.py`) and threads a read-only type lookup through
the call sites that render values (the interpreter, the cast converter, and the
REPL echo). It also updates the language-reference docs and the e2e suite.

## Resolved Owner Decisions

These were confirmed with the owner one-by-one and frame the implementation.

| # | Decision |
|---|----------|
| D1 | **Global scope.** The single renderer `render_value` becomes AgL-native at *every* output site: `print`, REPL echo, template/`${…}` interpolation (agent prompts **and** `exec` commands), and the `as text` cast. JSON is reachable only via `as json`. |
| D2 | **Single-line / compact layout.** AgL-form values render on one line: `Issue(title: "x", severity: 3, author: Author(name: "Ada"))`; lists and dicts inline too. No injected newlines. |
| D3 | **Enums render qualified.** `Outcome.Partial(left: 2)`; nullary variant as `Outcome.Done` (no parens). Keeps the type name always visible (parity with records) and is unambiguous under variant-name collisions. |
| D4 | **Dict keys are always quoted.** `{"origin": "static", "retries": 3}`. This is always valid AgL, avoids coupling the runtime renderer to lexer/keyword rules, and applies only to the AgL `dict` type (`json` retains JSON rendering). |
| D5 | **Exceptions render record-style with *all* fields, including `trace_id`.** `CastError(message: "…", trace_id: "…")`. No special-casing in the renderer; fully faithful to runtime state. |
| D6 | **Text: top-level verbatim, nested quoted.** `print("hi")` → `hi`; `print(R(t: "hi"))` → `R(t: "hi")`. Nested `text` is emitted as a fully-escaped AgL string literal (JSON escape set **plus `\$`** so `${` cannot read as interpolation). The REPL echo additionally quotes top-level `text` (preserving today's `render_value_repl` behavior). |
| D7 | **Fields render in declaration order**, not construction-argument order, for records, enums, and exceptions — canonical, deterministic output independent of how the value was constructed. Nominal rendering requires an authoritative read-only type lookup. An unknown type, wrong nominal kind/variant, or runtime/declaration field-set mismatch is an internal invariant error; the renderer never falls back or silently omits fields. |
| D8 | **JSON layout is unchanged** (the `as json` path stays 2-space pretty-printed, multi-line). This feature only *adds* the AgL form; the existing JSON serializer and its tests are left intact. |
| D9 | **A nested `json`-typed value renders compact** (single-line) so the enclosing AgL value stays single-line (D2); a *top-level* `json` value stays pretty-printed (D8). This top-level-vs-nested split mirrors the text rule in D6. |
| D10 | **Explicit nominal-to-JSON casts are total.** Records, enums, and exceptions join the sources accepted by `as json`; `value_to_json_obj` supplies their structural encoding. This is an explicit conversion only and does not make nominal values JSON-shaped or implicitly assignable to `json`. Exceptions are also accepted by `as text`, consistent with D1. |
| D11 | **Rendering depends on a read-only type-lookup protocol, not `TypeEnvironment` itself.** `TypeEnvironment` satisfies the protocol structurally for checked programs. The REPL exposes a read-only facade over its persistent environment so presentation code cannot mutate session typing state. |

## Rendering specification

The renderer is a single recursive function over the runtime `Value` union
(`src/agm/agl/eval/values.py`). Two boolean axes drive the leaf cases:

- **top-level vs. nested** — set `top_level=True` for the value passed in by a
  caller; every recursive child call uses `top_level=False`.
- **interpolation vs. REPL echo** — only affects top-level `text`.

Per-kind rules:

| Value kind | Rendering |
|---|---|
| `text` (top-level, interpolation) | verbatim, no quotes — `hi` |
| `text` (top-level, REPL echo) | quoted AgL string literal — `"hi"` |
| `text` (nested) | quoted AgL string literal (escapes incl. `\$`) — `"hi"` |
| `int` / `decimal` / `bool` | plain scalar text via existing `_scalar_text` (`1`, `1.5`, `true`) — unchanged, same at any depth |
| `unit` | `()` — unchanged (non-data; only appears top-level) |
| `agent` | `<agent NAME>` — unchanged (non-data; REPL/echo only) |
| `function` (`Closure`) | `<function/N -> T>` — unchanged (REPL/echo only) |
| `json` (top-level) | pretty JSON, 2-space indent (D8) |
| `json` (nested) | compact JSON, single-line (D9) |
| `list` | `[e1, e2, …]`, children nested; empty → `[]` |
| `dict` | `{"k1": v1, "k2": v2}`, keys always quoted with `_quote_text`, values nested; empty → `{}` |
| record | `TypeName(f1: v1, f2: v2)`, fields in declaration order (D7), values nested; no fields → `TypeName()` |
| enum | `TypeName.Variant(f1: v1, …)`; nullary → `TypeName.Variant`; fields in declared variant order |
| exception | `TypeName(f1: v1, …)` with *all* runtime fields incl. `trace_id`, in declaration order (D5/D7) |

Note that AgL containers (`list`, `dict`, record, enum, exception) render the
same whether top-level or nested — they are always single-line AgL form. Only
`text` and `json` differ by depth.

Worked examples (with `record Author { name: text, active: bool }` and
`record Issue { title: text, severity: int, tags: list[text], author: Author }`):

```
Issue(title: "Missing tests", severity: 3, tags: ["tests", "coverage"], author: Author(name: "Ada", active: true))
Outcome.Partial(left: 2)
Outcome.Done
{"origin": "static", "two words": 1}
CastError(message: "cannot parse \"x\" as int", trace_id: "evt-7", source_type: "text", target_type: "int", raw: "x")
```

### D6 — string escaping detail

`render.py` already has `_quote_text` producing a double-quoted surface form
with the JSON escape set and `\uXXXX` for control chars. It will be extended to
also escape `$` as `\$`, and reused for **both** nested `text` and the
top-level REPL-echo case so the two never diverge. (This slightly changes the
REPL echo of a string containing `$`, e.g. `"a${b}"` now echoes `"a\${b}"`;
that is the correct, round-trippable form and the corresponding REPL test will
be updated.)

### D7/D11 — declaration-order field lookup and invariants

The runtime `RecordValue` / `EnumValue` / `ExceptionValue` carry a `type_name`
(and `variant`) string plus a `fields` dict in *construction* order
(`interpreter.py:1262-1284`). To emit declaration order the renderer resolves
the definition through a small read-only `TypeLookup` protocol whose only
operation is `get_type(name) -> Type | None`. `TypeEnvironment` already has this
shape, but the renderer does not depend on its mutation API.

For every nominal value, the renderer:

1. Requires a type lookup and resolves `value.type_name`.
2. Requires the resolved definition to have the matching nominal kind; for an
   enum, it also requires `value.variant` to exist.
3. Compares the declared and runtime field-name sets for exact equality.
4. Emits the runtime values in declaration order.

Failure of any requirement is a `RuntimeError` identifying the type and the
violated invariant. Such a mismatch cannot be produced by a valid AgL program;
it represents a bug in the runtime value representation and must never be
hidden by construction-order fallback or field omission.

`RecordValue.type_name`, `EnumValue.type_name`, and `ExceptionValue.type_name`
are populated from the semantic type's `.name`, which is the same name used by
`TypeEnvironment.get_type`. This is therefore an implementation invariant, not
an open module-name normalization question.

### D10 — nominal `as json` semantics

The existing cast matrix accepts only JSON-shaped scalar/container sources for
`as json`, even though `value_to_json_obj` already defines structural encodings
for records, enums, and exceptions. This feature expands `cast_classification`:

- `record`, `enum`, or exception → `json` is `TOTAL_JSON`;
- exception → `text` is `TOTAL_RENDER`, matching other rendered data values;
- implicit assignability to `json` remains unchanged, so nominal values require
  the explicit cast;
- record JSON is its field object, enum JSON retains the existing `"$case"`
  tag, and exception JSON contains all fields including `trace_id`.

The casts plan and language reference must be updated in the same change so
they no longer describe `record as json` as a static error.

## Architecture & threading

`render_value` is currently parameterless beyond the value and is called from:

- `interpreter.py:558` — template interpolation (`_eval_template`)
- `interpreter.py:838` — `_eval_to_text` (exec command / ask prompt text)
- `interpreter.py:845` — `_eval_print_call`
- `convert.py:420` — `as text` cast result **and** `CastError.raw` diagnostic
- `repl/render.py` — `render_value_repl` via `format_typed_value` / `_render_echo`

All interpreter call sites have `self._checked.type_env` in scope. That object
satisfies `TypeLookup` and is passed to `render_value`. `convert_value`
(`convert.py:384`) gains a `type_lookup` parameter and forwards it to
`render_value`; its two callers (`interpreter.py:1406,1425`) pass
`self._checked.type_env`.

The REPL has no persistent `CheckedProgram`; it owns a persistent `_type_env`.
`ReplSession` therefore exposes a read-only `TypeLookup` facade backed by that
environment. The console echo path, `:load`, `:bindings`, and `:params` pass
this facade through `render_entry_result`, `_render_echo`, and
`format_typed_value` to `render_value_repl`. This enumerated flow is required so
all REPL displays use the same canonical declaration order without exposing a
mutable `TypeEnvironment` to presentation code.

No runtime-to-lexer dependency is introduced. `runtime/render.py` depends only
on semantic types and the read-only lookup protocol; always-quoted dict keys do
not require duplicating or importing lexical identifier rules.

## Implementation milestones

Following TDD: write failing tests first, then implement.

### M1 — Core renderer rewrite (`runtime/render.py`)

- Replace the JSON-by-default `render_value` body with the recursive
  AgL-native renderer per the spec table. Keep `_scalar_text` and
  `_closure_surface` as-is.
- Extend `_quote_text` to escape `$` → `\$`; reuse it for nested text and
  REPL-echo top-level text.
- Quote every AgL dict key with `_quote_text`; do not inspect lexer tokens or
  keywords (D4).
- Define the read-only `TypeLookup` protocol and add a declared-field-order
  helper that resolves `RecordType` / `EnumType` / `ExceptionType`, validates
  nominal kind, enum variant, and exact field-set equality, then yields ordered
  `(field_name, value)` pairs (D7/D11).
- `json` rendering: nested → `dumps_exact(value_to_json_obj(v), indent=None)`
  (compact, D9); top-level → `dumps_exact(..., indent=2)` (pretty, D8).
  `serialize.dumps_exact` already supports `indent=None`.
- `render_value(value, type_lookup=None)` and
  `render_value_repl(value, type_lookup=None)` keep their names. A lookup is
  optional for scalar and structural container values but mandatory when a
  nominal value is encountered; absence is an internal invariant error.
  `render_value_repl` differs only by quoting top-level `text`.
- Rewrite the module docstring (currently describes JSON-by-default).
- Tests: rewrite `TestRenderValue` in `tests/test_agl_runtime.py` to assert the
  AgL forms (records, qualified enums incl. nullary, exceptions with
  `trace_id`, lists, always-quoted dict keys, nested text escaping incl.
  `\$`, top-level vs nested `text`, top-level pretty vs nested compact `json`,
  declaration-order normalization including out-of-order construction, empty
  record/list/dict). Add regression tests proving nominal rendering raises an
  internal error for a missing lookup, unknown/wrong type, unknown enum variant,
  and missing or extra runtime fields. Prefer behavior tests through
  `render_value`; do not create a lexer-facing dict-key helper.

### M2 — Interpreter & cast wiring

- `cast_classification`: classify record/enum/exception → `json` as
  `TOTAL_JSON`, and exception → `text` as `TOTAL_RENDER`; keep implicit
  `is_assignable(..., JsonType)` rules unchanged.
- `convert_value` (`convert.py`): add `type_lookup` parameter and pass it to the
  `render_value` call whose result supplies both `as text` and
  `CastError.raw`.
- Interpreter: pass `self._checked.type_env` at `interpreter.py:558, 838, 845`
  and to `convert_value` at `interpreter.py:1406, 1425`.
- Tests: update `tests/test_agl_convert.py` `→ text` expectations (records,
  enums, lists, dicts, exceptions now produce AgL form, not JSON); add total
  record/enum/exception → `json` cases and `as?` coverage; confirm their exact
  existing structural JSON encodings. Add interpreter-level tests that `print`,
  template interpolation, and `exec`/prompt interpolation all emit AgL form,
  and that `${x as json}` emits JSON for every structured data kind.

### M3 — REPL echo wiring (`repl/render.py`)

- Add a read-only lookup facade over `ReplSession._type_env`.
- Thread it through console echo and the `:load`, `:bindings`, and `:params`
  meta-command paths, including `render_entry_result`, `format_typed_value`,
  and `_render_echo`; pass it to `render_value_repl`.
- Tests: direct entry echo, `:load`, `:bindings`, and `:params` render nominal
  values in AgL form and declaration order; top-level `text` stays quoted;
  update the `$`-containing string echo expectation.

### M4 — Docs sweep + e2e

- Sweep the repository for every description or example of rendering, pretty
  JSON, `as text`, `as json`, interpolation, and structured `print` output.
  Update every affected language-reference page, including at minimum
  `strings-and-interpolation.md`, `expressions.md`, `types.md`,
  `exceptions.md`, `shell-execution.md`, and relevant agent-call/host pages.
- Update `docs/plans/2026-06-19-agl-casts.md` and its conversion matrix to
  permit explicit record/enum/exception → `json` and exception → `text`, while
  preserving the distinction from implicit JSON-shaped assignability.
- `docs/arch/agl.md`: update the rendering architecture note (single recursive
  AgL-native renderer; read-only type lookup and strict nominal invariants).
- Update affected README, command documentation/help examples, plan documents,
  and test-program comments/fixtures found by the sweep; do not leave a second
  contradictory rendering specification elsewhere in the repository.
- e2e programs under `tests/agl/programs/`: add a `rendering/` group (or extend
  `types/`) exercising native rendering of records/enums/exceptions/lists/dicts,
  nesting, `as json` round-trips, and interpolation — combined with other
  language features per the AgL area testing guidance. Keep existing programs'
  expected output in sync where they print structures.

### M5 — Verify

- `just check` (lint + tests + strict mypy) green.
- Confirm 100% coverage of `src/` and 100% e2e command coverage maintained.
- No `type: ignore` / `noqa` / `fmt:` suppressions; if any static-analysis
  obstacle arises, stop and ask the owner.

## Explicitly unchanged

- JSON serialization keeps its existing structural shapes and field order;
  declaration-order normalization applies to AgL-native rendering only.
- Nominal values do not become implicitly assignable to `json`; structural JSON
  conversion requires `as json`.
- Existing prompts or shell commands that require JSON must opt in with
  `${value as json}`; M2/M4 update affected tests, fixtures, and examples.

## Risks

- **Wide output change.** Many tests and e2e expected-output fixtures assert
  the old JSON form. Expect broad, mechanical fixture updates; the risk is
  missing one, caught by `just check` and the 100%-coverage requirement.
- **REPL `$`-escaping behavior change.** Unifying `_quote_text` to escape `$`
  changes the echo of strings containing `$`. Intentional and round-trippable;
  the affected REPL test will be updated.
- **Cast semantic expansion.** Allowing explicit nominal → `json` must be
  reflected consistently in type checking, reference docs, the older casts
  plan, and e2e coverage; changing only the renderer would leave the motivating
  `print(r as json)` example statically invalid.
- **Type-lookup threading.** Canonical nominal output intentionally has no
  construction-order fallback. Missing lookup propagation now fails loudly, so
  interpreter, conversion, console, `:load`, `:bindings`, and `:params` paths
  must all be covered by tests.
