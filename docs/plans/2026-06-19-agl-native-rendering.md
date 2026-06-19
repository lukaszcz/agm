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
print(r as json)      # JSON is still reachable, via an explicit cast → {"x": 1}
```

JSON output becomes an explicit opt-in through the existing `as json` cast (see
the casts plan, `docs/plans/2026-06-19-agl-casts.md`): casting a value to `json`
produces a `json` value, which still renders as JSON. This makes "the default
rendering is AgL; `as json` gives JSON" universally true.

This change is centered on the runtime renderer
(`src/agm/agl/runtime/render.py`) and threads a type-namespace handle through
the call sites that render values (the interpreter, the cast converter, and the
REPL echo). It also updates the language-reference docs and the e2e suite.

## Resolved Owner Decisions

These were confirmed with the owner one-by-one and frame the implementation.

| # | Decision |
|---|----------|
| D1 | **Global scope.** The single renderer `render_value` becomes AgL-native at *every* output site: `print`, REPL echo, template/`${…}` interpolation (agent prompts **and** `exec` commands), and the `as text` cast. JSON is reachable only via `as json`. |
| D2 | **Single-line / compact layout.** AgL-form values render on one line: `Issue(title: "x", severity: 3, author: Author(name: "Ada"))`; lists and dicts inline too. No injected newlines. |
| D3 | **Enums render qualified.** `Outcome.Partial(left: 2)`; nullary variant as `Outcome.Done` (no parens). Keeps the type name always visible (parity with records) and is unambiguous under variant-name collisions. |
| D4 | **Dict keys: shorthand when a valid identifier, else quoted.** `{origin: "static", retries: 3}` but `{"two words": 1}`. (The `json` type always renders as JSON regardless — this rule is for the AgL `dict` type only.) |
| D5 | **Exceptions render record-style with *all* fields, including `trace_id`.** `CastError(message: "…", trace_id: "…")`. No special-casing in the renderer; fully faithful to runtime state. |
| D6 | **Text: top-level verbatim, nested quoted.** `print("hi")` → `hi`; `print(R(t: "hi"))` → `R(t: "hi")`. Nested `text` is emitted as a fully-escaped AgL string literal (JSON escape set **plus `\$`** so `${` cannot read as interpolation). The REPL echo additionally quotes top-level `text` (preserving today's `render_value_repl` behavior). |
| D7 | **Fields render in declaration order**, not construction-argument order, for records, enums, and exceptions — canonical, deterministic output independent of how the value was constructed. Requires the renderer to consult the type namespace; falls back to stored (construction) order when a type definition is not resolvable. |
| D8 | **JSON layout is unchanged** (the `as json` path stays 2-space pretty-printed, multi-line). This feature only *adds* the AgL form; the existing JSON serializer and its tests are left intact. |
| D9 | **A nested `json`-typed value renders compact** (single-line) so the enclosing AgL value stays single-line (D2); a *top-level* `json` value stays pretty-printed (D8). This top-level-vs-nested split mirrors the text rule in D6. |

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
| `dict` | `{k1: v1, k2: v2}`, keys per D4, values nested; empty → `{}` |
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
{origin: "static", "two words": 1}
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

### D7 — declaration-order field lookup

The runtime `RecordValue` / `EnumValue` / `ExceptionValue` carry a `type_name`
(and `variant`) string plus a `fields` dict in *construction* order
(`interpreter.py:1262-1284`). To emit declaration order the renderer needs the
type definition, which lives in `CheckedProgram.type_env`
(`src/agm/agl/typecheck/env.py`): `TypeEnvironment.get_type(name)` returns the
`RecordType` (`.fields`, ordered), `EnumType` (`.variants[variant]`, ordered),
or `ExceptionType` (`.fields`, ordered). The renderer:

1. Looks up `type_env.get_type(value.type_name)`.
2. If found, iterates its declared field names; for each, emits the matching
   entry from `value.fields`.
3. If not found (no `type_env` passed, or name unresolvable — e.g. certain
   module-qualified edge cases), falls back to `value.fields` iteration order.

This makes `type_env` an **optional** renderer parameter: callers that have it
(interpreter, cast converter, REPL session) pass it for canonical output;
the fallback keeps the function callable without it.

> Open point (verify during M1): confirm whether `RecordValue.type_name`
> stores the simple name or a module dot-path, and that `type_env.get_type`
> is keyed consistently. If they differ under the module system, add the
> appropriate normalization (the construction-order fallback covers the gap
> safely in the meantime). See the module-system memory.

## Architecture & threading

`render_value` is currently parameterless beyond the value and is called from:

- `interpreter.py:558` — template interpolation (`_eval_template`)
- `interpreter.py:838` — `_eval_to_text` (exec command / ask prompt text)
- `interpreter.py:845` — `_eval_print_call`
- `convert.py:420` — `as text` cast result **and** `CastError.raw` diagnostic
- `repl/render.py` — `render_value_repl` via `format_typed_value` / `_render_echo`

All interpreter and convert call sites already have `self._checked.type_env`
in scope. `convert_value` (`convert.py:384`) gains a `type_env` parameter and
forwards it to `render_value`; its two callers (`interpreter.py:1406,1425`)
pass `self._checked.type_env`. The REPL renderer threads `type_env` from the
session's checked program into `format_typed_value` and `_render_echo`
(`repl/render.py`); the source of the checked program in the REPL session will
be wired in M3.

No new module-dependency direction is introduced: `runtime/convert.py` already
imports from `typecheck.types`, so `runtime/render.py` consulting a
`TypeEnvironment` is consistent with existing layering.

## Implementation milestones

Following TDD: write failing tests first, then implement.

### M1 — Core renderer rewrite (`runtime/render.py`)

- Replace the JSON-by-default `render_value` body with the recursive
  AgL-native renderer per the spec table. Keep `_scalar_text` and
  `_closure_surface` as-is.
- Extend `_quote_text` to escape `$` → `\$`; reuse it for nested text and
  REPL-echo top-level text.
- Add a dict-key helper: shorthand when the key matches the grammar `NAME`
  identifier rule, else `_quote_text` (D4). Reuse the lexer/grammar identifier
  definition rather than hand-rolling a regex.
- Add a declared-field-order helper that resolves a `RecordType` / `EnumType`
  / `ExceptionType` via the optional `type_env` and yields ordered
  `(field_name, value)` pairs, with construction-order fallback (D7).
- `json` rendering: nested → `dumps_exact(value_to_json_obj(v), indent=None)`
  (compact, D9); top-level → `dumps_exact(..., indent=2)` (pretty, D8).
  `serialize.dumps_exact` already supports `indent=None`.
- `render_value(value, type_env=None)` and
  `render_value_repl(value, type_env=None)` keep their names; `render_value_repl`
  differs only by quoting top-level `text`.
- Rewrite the module docstring (currently describes JSON-by-default).
- Tests: rewrite `TestRenderValue` in `tests/test_agl_runtime.py` to assert the
  AgL forms (records, qualified enums incl. nullary, exceptions with
  `trace_id`, lists, dicts with both key styles, nested text escaping incl.
  `\$`, top-level vs nested `text`, top-level pretty vs nested compact `json`,
  declaration-order normalization including out-of-order construction, empty
  record/list/dict). Add unit tests for the dict-key shorthand helper and the
  declared-order helper (incl. fallback when `type_env` is `None`).

### M2 — Interpreter & cast wiring

- `convert_value` (`convert.py`): add `type_env` parameter; forward to both
  `render_value` calls (the `as text` result and the `CastError.raw` field).
- Interpreter: pass `self._checked.type_env` at `interpreter.py:558, 838, 845`
  and to `convert_value` at `interpreter.py:1406, 1425`.
- Tests: update `tests/test_agl_convert.py` `→ text` expectations (records,
  enums, lists, dicts, exceptions now produce AgL form, not JSON); confirm
  `→ json` is unchanged. Add interpreter-level tests that `print`, template
  interpolation, and `exec`/prompt interpolation all emit AgL form, and that
  `${x as json}` still emits JSON.

### M3 — REPL echo wiring (`repl/render.py`)

- Thread `type_env` (from the REPL session's checked program) into
  `format_typed_value` and `_render_echo`; pass to `render_value_repl`.
- Tests: REPL echo of records/enums/dicts shows AgL form; top-level `text`
  stays quoted; update the `$`-containing string echo expectation.

### M4 — Docs + e2e

- `docs/agl/reference/strings-and-interpolation.md`: rewrite the interpolation
  rendering rules (AgL form for structures, top-level verbatim text, `as json`
  escape hatch).
- `docs/agl/reference/expressions.md` (casts) and `types.md`: note that the
  default rendering of data structures is AgL syntax and JSON is obtained via
  `as json`; cross-reference.
- `docs/arch/agl.md`: update the rendering architecture note (single recursive
  AgL-native renderer; `type_env` threading for declaration order).
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

## Open points to confirm during implementation

1. **Module-qualified type names (D7).** Whether `RecordValue.type_name` is a
   simple name or dot-path and whether `type_env.get_type` is keyed to match
   (see memory on the AgL module system). Construction-order fallback makes
   this non-blocking but it should be resolved for canonical output.
2. **JSON field order under `as json`.** D8 keeps JSON pretty-printed; it also
   keeps the existing serializer's *construction* field order for records (the
   `value_to_json_obj` path is untouched). If declaration-order JSON is also
   wanted later, that is a separate, additive change — flagged, not done here.
3. **`exec`/prompt blast radius (from D1).** Any existing e2e program or
   docs example that relied on a structure interpolating as JSON into a prompt
   or shell command must switch to `${value as json}`. M2/M4 will sweep these.

## Risks

- **Wide output change.** Many tests and e2e expected-output fixtures assert
  the old JSON form. Expect broad, mechanical fixture updates; the risk is
  missing one, caught by `just check` and the 100%-coverage requirement.
- **REPL `$`-escaping behavior change.** Unifying `_quote_text` to escape `$`
  changes the echo of strings containing `$`. Intentional and round-trippable;
  the affected REPL test will be updated.
- **Renderer ↔ type-namespace coupling.** Threading `type_env` touches several
  call sites. The optional-parameter-with-fallback design contains the blast
  radius and keeps the renderer usable without a type environment.
