# AgL Raw-Tail Builtins (`exec!`, `ask!`) and `%{}` Interpolation — Implementation Plan

Status: planned · Date: 2026-07-23 · **Every** design decision below is owner-approved.

This is the standalone, authoritative design and implementation plan for the raw-tail syntax
(`exec!` for shell commands, `ask!` for agent prompts) and the accompanying global
interpolation-character change. It records the settled owner decisions, the language
specification, the implementation design, the migration inventory, and the acceptance criteria;
no external design note is required to interpret it.

## 1. Goal

Make shell execution and agent prompting ergonomic. Today a shell command or a prompt must be
spelled as a string argument (`exec("ls -lh")`, `ask("Summarize ${topic}")`). This plan
introduces **raw-tail forms** — `exec!` and `ask!` — whose payload text extends to the end of the
line or over a following indented block, verbatim except for AgL interpolation:

```agl
exec! ls -lh

exec!
  ls -lh

exec! ls -lh "My File Name With Spaces"

let file = "path/to/file"
exec! ls -lh $PATH %{file}

ask! Summarize the changes in %{file}

let r: Review = ask!::[Review]
  Review %{file} and report:
  - correctness
  - style
```

Because shell's `$` must pass through verbatim, AgL's interpolation trigger changes **globally**
from `${expr}` to `%{expr}` — in ordinary strings and in raw-tail text alike, with identical
semantics everywhere.

The existing `exec` and `ask` builtins are **not changed in any way**: the call forms, the
juxtaposition sugar (`exec "cmd"`, `ask "prompt"`), type args, contextual typing, and the named
arguments (`agent`, `format`, `strict_json`, `on_parse_error`) all keep their current syntax and
semantics. `exec!` and `ask!` are pure surface sugar that desugars to `exec(<template>)` /
`ask(<template>)` calls.

## 2. Non-goals (owner-approved)

- No change to `exec`/`ask` runtime semantics: `sh -c`, `ExecResult`, agent dispatch, parsed/unit
  result forms, retry loops, `ExecError`/`AgentParseError`, and tracing are untouched.
- No change to the `exec`/`ask` call forms' syntax or typing. No migration of existing call sites.
- No `print!` and no raw form for `ask-request` (a niche introspection builtin — call form only).
  Future raw-tail adopters each require a separate owner decision.
- No named arguments in the raw forms — `agent`, `format`, `strict_json`, and `on_parse_error`
  remain reachable only through the call forms. In particular, `ask!` always dispatches to the
  **default agent** (D8); a dedicated agent slot (e.g. `ask!@reviewer`) is a possible later
  extension in the prefix-slot seam, not part of this change.
- No new escape processing in raw-tail text beyond the single `\%{` rule (D3). In particular, no
  `\n`/`\uXXXX`-style escapes and no shell-side quoting/word-splitting performed by AgL.

## 3. Settled owner decisions

- **D1 — Interpolation trigger is `%{expr}` globally.** Strings and raw-tail text share one
  interpolation syntax and one set of semantics. A lone `%` not followed by `{` remains literal
  (mirroring today's lone-`$` rule), so `printf '%s'`, `date +%Y`, and `100%` are unaffected. In
  ordinary strings the escape `\%` yields a literal `%` (replacing `\$`); `\$` ceases to be a valid
  escape (a bare `$` needs no escape). Rationale: shell text must be verbatim for **all** dollar
  forms (`$VAR`, `"${var}"`, `$(...)`, `$1`), and `%{` is rare in shell, prose, and prompts; the
  collision moves to a sequence that almost never occurs and fails loudly (unknown-name scope
  error) when it does. Rejected: `$$` doubling (taxes every shell dollar, Make-style pain, diverges
  from string escape rules) and keeping `${` with lone-`$` literal (silently captures
  shellcheck-style `"${var}"`).
- **D2 — Raw-tail forms are distinct `<name>!` keywords; the base builtins are untouched.** The
  two forms are never mixed: after `exec!`/`ask!` (and an optional `::[T]` type-arg group)
  *everything* is verbatim payload — parentheses, quotes, `#`, `;`, `$` included — so there is no
  disambiguation rule at all, and subshell-first (`exec! (cd x && ls)`) or parenthesis-first
  prompts work inline. Rationale: overloading the base names required token-lookahead rules with
  silent or surprising edge cases in both directions; distinct names remove the ambiguity class
  entirely and cost nothing since `exec!`/`ask!` already lex as single NAMEs (`!` is an
  identifier-continuation character).
- **D3 — Escape in raw-tail text: the three-character sequence `\%{` yields a literal `%{`.**
  Every other backslash is verbatim. This mirrors the string-side `\%` escape — one suppression
  idea across the language — and collides with no common shell idiom (`date +\%Y`,
  `printf '100%%'`, `sed 's/x/\n/'` all pass through). The always-available fallback for
  pathological cases is interpolation itself: `%{"%{"}` splices a literal `%{` (spliced values are
  never re-scanned).
- **D4 — Full typing parity.** A raw-tail form is sugar for the base builtin applied to a
  template: explicit type args (`exec!::[json] cat data.json`, `ask!::[Review] ...`), contextual
  expected-type, the per-builtin defaults (`ExecResult` for `exec`, `text` for `ask`), and exec's
  discard-to-`unit` rule all behave exactly as in the call forms. Raw forms differ from call forms
  only in how the payload string is written.
- **D5 — Line-final positions only.** Raw-tail forms are legal exactly where the grammar
  guarantees the expression extends to the end of its line: block-item (statement) position,
  `let`/`var`/assignment RHS, `return` operand, `=` function bodies, and last-argument
  juxtaposition position. Inside brackets, or anywhere further syntax could follow on the line, it
  is a hard compile-time error with a hint to use the call form. Rationale: the alternative ("eats
  to EOL wherever it appears") silently executes surrounding AgL syntax as payload.
- **D6 — Block form is one payload.** A raw-tail keyword followed by end-of-line and an indented
  block denotes a single payload: the block's lines are dedented to the block margin (relative
  indentation and blank lines preserved — right for both shell scripts and markdown prompts),
  joined with newlines, and passed as one command (`sh -c`) or one prompt. A raw-tail keyword with
  no payload (end-of-line and no indented block) is an error.
- **D7 — The raw-tail mechanism is generic; `exec!` and `ask!` are the adopters.** The scanner
  mode, tokens, grammar production, and desugaring are keyword-agnostic: a registry maps raw-tail
  names to their underlying builtin (`{"exec!": "exec", "ask!": "ask"}`). The `<name>!` suffix is
  the reserved convention for any future raw-tail forms, each adopted by a later owner decision.
- **D8 — `ask!` uses the default agent; agent selection is call-form only.** Rejected: a
  parenthesized prefix option group (`ask!(agent = r) ...` — reintroduces the lookahead-ambiguity
  class rejected in D2; prompts legitimately start with `(`) and a dedicated `@agent` slot
  (novel syntax, names-only, no `exec!` counterpart). If explicit-agent ergonomics prove painful,
  a slot in the prefix seam can be added by a later owner decision without breaking anything.

## 4. Language specification

### 4.1 Interpolation: `%{expr}`

- In string templates (single- and triple-quoted, both quote styles) and in raw-tail text,
  `%{expr}` embeds an expression. Semantics (expression checking, uniform rendering, single-line
  restriction inside the braces) are unchanged from today's `${expr}` — only the trigger character
  changes.
- A `%` not followed by `{` is literal content everywhere.
- In strings: `\%` is a valid escape producing `%`; `\$` is removed from the escape table (unknown
  escapes remain lexical errors, so stale `\$` fails loudly at the right location).
- Rendering: when nested text is quoted during uniform rendering, the interpolation trigger is
  escaped as `\%` (today `\$`), keeping rendered text re-parseable.

### 4.2 Raw-tail inline form

```agl
exec! <shell text to end of line>
exec!::[T] <shell text to end of line>
ask! <prompt text to end of line>
ask!::[T] <prompt text to end of line>
```

- `exec!` and `ask!` are reserved names; they cannot be declared, referenced bare, or used as
  agent/program names (same reservation class as `exec`/`ask`).
- After the keyword and optional type-arg group, payload text starts at the first non-space
  character and extends to the end of the line. Trailing whitespace is trimmed; interior content
  is verbatim.
- Verbatim means verbatim: `"`/`'` quotes, `(`/`)`, `#` (payload text, not an AgL comment), `;`
  (no second AgL statement can share the line), `$` in all forms, and backslashes all pass through
  untouched. The only special sequences are `%{` (interpolation) and `\%{` (literal `%{`).
- An empty payload (end-of-line immediately after the keyword with no indented block, or only
  whitespace) is an error.
- `ask!` dispatches to the default agent with default parse-shaping options; `exec!` runs with
  default options. Named arguments require the call forms (D8).

### 4.3 Raw-tail block form

```agl
exec!
  <shell line>
  ...

ask!::[Review]
  <prompt line>
  ...
```

- Trigger: the keyword (plus optional `::[T]`) followed by end-of-line and a more-indented block.
- The block consists of the following lines indented deeper than the keyword's line, up to the
  first line at or below its indentation. The margin is the indentation of the first content line;
  every non-blank line must be indented at least to the margin (a lexical error otherwise). The
  margin is stripped, relative indentation and blank lines are preserved, and the lines are joined
  with newlines into one payload.
- The result is a single command — one `sh -c` invocation — or a single prompt, with the same
  typing as the inline form. Shell control-flow keywords inside an `exec!` block (`done`, `else`,
  `if`, `for`, ...) are plain shell text and never interact with AgL layout; likewise any prompt
  content in an `ask!` block.
- Interpolation works across all block lines; a `%{...}` hole must still be contained in one line.

### 4.4 Positions

Raw-tail forms may appear exactly in line-final expression positions:

- as a block item (statement position),
- as the RHS of `let` / `var` / assignment,
- as a `return` operand,
- as an `=` function body (`def today() = exec! date +%F`),
- as the final juxtaposition argument (`print exec! date`).

Anywhere else — inside parentheses/brackets/braces, inside an interpolation hole, or in an inline
position where more syntax could follow — is a compile-time error whose message points to the call
form and the block form. The criterion for future grammar extensions is fixed: a position
qualifies iff the grammar guarantees nothing else can follow on the same line.

### 4.5 Typing and desugaring

`exec! <text>` / `ask! <text>` desugar in the parser to the same AST as `exec(<template>)` /
`ask(<template>)`, where `<template>` is the template built from the payload's fragments and
interpolation holes (collapsing to a plain string literal when there are no holes). Scope
classification (`BuiltinKind.EXEC`/`ASK`), type checking (target-type selection, per-builtin
defaults, exec's unit-discard, obligation finalization), lowering, and evaluation are shared with
the call forms with **zero changes** below the parser. Named arguments are not expressible in the
raw forms by design (D2/D8); the call forms remain the home for `agent`, `format`, `strict_json`,
and `on_parse_error`.

## 5. Implementation design

### 5.1 Lexer (`src/agm/agl/lexer/`)

**Interpolation switch** (`scanner.py`): the template sub-scanners' trigger check changes from
`$`+`{` to `%`+`{` (both the single- and triple-quoted paths); the escape table replaces `\$` with
`\%`. The `INTERP_START` token text becomes `%{`.

**Raw-tail mode** (new, generic):

- Registry: a small mapping of raw-tail names to underlying builtins,
  `{"exec!": "exec", "ask!": "ask"}`, owned by the lexer module (the generic seam of D7). All
  raw-tail behavior below is parameterized by this registry only — nothing exec- or ask-specific.
- Trigger: the scanner already produces `exec!`/`ask!` as single NAME tokens (`!` is an
  identifier-continuation character). When a scanned NAME is in the registry, the previous
  significant token is not a field-access `.` (or module qualifier), and the bracket depth is
  zero, the scanner emits a dedicated `RAW_TAIL_NAME` token and enters raw-tail mode. At nonzero
  bracket depth this is a lexical error with the "not allowed inside brackets; use the call form"
  message (brackets suppress newlines, so "to end of line" is meaningless there).
- After the trigger, an optional `::[...]` type-arg group is scanned with ordinary code tokens
  (bracket-balanced, as today), then the mode consumes either the rest of the line (inline) or the
  indented block (block form), emitting `RAW_TAIL_START`, `RAW_FRAGMENT` text pieces, standard
  `INTERP_START expr INTERP_END` hole token runs (reusing the existing interpolation-code
  sub-scanner, including its no-newline-in-hole error), and `RAW_TAIL_END`.
- The block form is consumed entirely inside the scanner, like triple-quoted templates: the layout
  pass never sees the interior lines, so shell `done`/`else`/`until` lines (or any prompt content)
  cannot trip the layout continuation or synthetic-`done` rules. Dedenting adapts the existing
  triple-quoted dedent helper (margin from the first content line; under-indented non-blank lines
  are lexical errors). After the block, the scanner resumes normal scanning and emits the usual
  `_NEWLINE` so layout stays consistent.
- `\%{` in raw fragments emits a literal `%{`; all other backslashes are fragment text.

### 5.2 Grammar and parser (`grammar/agl.lark`, `parser/transform.py`)

- `%declare` the new tokens. Productions:
  `raw_tail: RAW_TAIL_START (RAW_FRAGMENT | interp)* RAW_TAIL_END` and
  `raw_call: RAW_TAIL_NAME type_args? raw_tail`.
- A `raw_call` alternative is added at exactly the line-final positions of §4.4 (block item, binder
  RHS, `return`, `=` bodies, final juxtaposition argument). Because `RAW_TAIL_NAME` never begins an
  ordinary expression, this stays LALR-unambiguous, mirroring the `suite_expr` invariant.
- The AST builder desugars `raw_call` to a `Call` whose callee is a synthesized `VarRef` for the
  underlying builtin name from the registry (span on the keyword token) and whose sole positional
  argument is the built `Template` (collapsing to `StringLit` when hole-free, via the existing
  helper). Type args pass through unchanged. No new AST node kinds.

### 5.3 Downstream passes

No changes to scope, typecheck, lower, IR, or eval: the desugared nodes are indistinguishable from
call-form `exec`/`ask`. The plan's tests assert this equivalence directly (§7). The only
shared-runtime change is the rendering quote-escape switch (`runtime/render.py`, `$` → `%`).

### 5.4 Reservation and diagnostics

- `exec!` and `ask!` join the reserved-name checks that today cover `exec`/`ask` (declaration
  sites, agent names, reserved program names), with messages naming the raw forms.
- New diagnostics: raw form inside brackets (lex), empty payload (lex/parse), under-indented block
  line (lex), raw form in a non-line-final position (parse; hint: call form or block form).

### 5.5 REPL (`src/agm/agl/repl/`)

The console's multi-line continuation must treat a raw-tail keyword + end-of-line as an open block
(like other suite openers) so the block form is enterable interactively; entry evaluation then
flows through the ordinary pipeline. Covered by REPL e2e tests.

## 6. Migration inventory

All migration is mechanical and lands with stage S1 (interpolation) — the raw-tail forms
themselves require no migration since `exec` and `ask` are untouched.

- `${` → `%{` in `tests/**/*.agl` (~266 occurrences across ~74 files) and in AgL snippets embedded
  in `tests/**/*.py` (~235 occurrences across ~25 files — each verified to be AgL source, not
  Python/shell, before rewriting).
- `\$` → `$` (escape no longer valid, `$` needs none) wherever it appears in AgL sources and in
  expected-output fixtures; rendering-related expectations change from `\$` to `\%` quoting.
- Docs examples under `docs/agl/` (exercised by `test_agl_doc_snippets.py`, so covered by tests).
- Pre-check: repository-wide grep confirms **zero** existing `%{` sequences in `.agl` sources,
  stdlib, or docs, so no content needs `\%{`-escaping during migration.
- `stdlib/` needs no changes (no interpolation occurrences).

## 7. Testing plan (TDD)

Write failing tests first at each stage; group by behavior, not by plan stage. Agent calls in all
`ask!` tests are mocked (never real agents), via the existing agent-mock infrastructure.

- **Lexer unit tests**: `%{` triggering and `\%` escape in single/triple templates; `\$` now a
  lexical error; raw-tail token streams for inline and block forms (both keywords — the registry
  is exercised generically, not per-builtin); `\%{` suppression; verbatim `#`, `;`, quotes,
  backslashes, `$` forms; type-arg prefix; block dedent, relative indentation, blank lines;
  under-indent error; empty-payload error; bracket-refusal error; no trigger after `.`; `exec!x` /
  `ask!x` (single NAMEs, no trigger); shell `done`/`else` lines inert in blocks; newline-in-hole
  error.
- **Parser/AST equivalence**: `exec! ls -lh` produces an AST equivalent to `exec("ls -lh")`, and
  `ask! Summarize %{f}` to `ask(<template>)`; interpolated and block forms produce the equivalent
  call-form AST; type args carried over.
- **Rejection tests** (`tests/agl/rejections/`): raw forms inside brackets, in inline-composite
  positions, empty payload, declaring/referencing `exec!`/`ask!`, misaligned block.
- **Typecheck parity**: `::[T]`, contextual typing, per-builtin defaults (`ExecResult` / `text`),
  exec's unit-discard — each asserted equal in raw and call forms; named arguments (including
  `agent`) remain call-form-only.
- **E2e programs** (`tests/agl/programs/exec/` and agent-call program dirs, multi-scenario per
  program): inline commands with `$`-heavy shell and quoting; interpolation of variables and
  expressions; block scripts with shell loops/conditionals (`done` inside); `\%{` (curl-style
  format strings); typed results; failure/exit-code scenarios; `ask!` prompts inline and as
  markdown blocks with typed (`::[Review]`) and text results against mocked agents, alongside
  call-form `ask` with explicit agents in the same programs; existing `exec`/`ask` programs
  unchanged and passing.
- **REPL e2e**: inline raw-tail entries; block-form continuation.
- Maintain 100 % coverage of `src/` and 100 % e2e command coverage.

## 8. Implementation stages

Each stage follows TDD, ends with `just check` green, and is committed separately.

1. **S1 — interpolation switch + migration.** Scanner trigger and escape-table change, rendering
   quote-escape change, full `${`→`%{` / `\$` migration of tests and docs, reference-doc updates
   for interpolation (§9).
2. **S2 — raw-tail lexer mode.** Registry (both keywords), tokens, trigger rules, inline and block
   scanning, dedent, escapes, lexical diagnostics.
3. **S3 — grammar, parser, desugaring.** Productions at line-final positions, AST desugar to the
   call forms, reservation of `exec!`/`ask!`, parse diagnostics, AST-equivalence and rejection
   tests.
4. **S4 — parity, e2e, REPL.** Typing-parity tests, e2e shell-block and prompt-block programs
   (mocked agents), REPL block continuation, diagnostics polish.
5. **S5 — documentation sweep.** Remaining reference and architecture docs (§9).

## 9. Documentation updates

- `docs/agl/reference/strings-and-interpolation.md` — `%{expr}`, `\%`, rendering quote rule (S1).
- `docs/agl/reference/lexical-structure.md`, `docs/agl/reference/grammar.md` — trigger change
  (S1); raw-tail tokens, `exec!`/`ask!` forms, line-final positions (S5).
- `docs/agl/reference/shell-execution.md` — `exec!` inline/block forms, verbatim and `\%{` rules,
  positions, typing parity, when to use the call form (S5).
- `docs/agl/reference/agent-calls.md` — `ask!` inline/block forms, default-agent rule, when the
  call form is required (`agent =`, parse-shaping options) (S5).
- `docs/arch/agl/frontend/*.md` — raw-tail scanner mode, registry seam, new tokens; succinct,
  architecture-level only (S5).
- Update any command help text or `docs/commands/*.md` that quotes interpolation examples (S1).

## 10. Acceptance criteria

- All motivating examples work: `exec! ls -lh`; the equivalent block form; verbatim
  `exec! ls -lh "My File Name With Spaces"`; `exec! ls -lh $PATH %{file}` with `$PATH` reaching
  the shell and `%{file}` interpolated by AgL; `ask! Summarize the changes in %{file}` against the
  default agent; a typed `ask!::[Review]` markdown block prompt.
- `exec` and `ask` call-form programs (including juxtaposition sugar and named arguments) behave
  byte-for-byte as before.
- Raw/call parity: for any payload text, the raw form and the call form applied to the same
  template produce identical results and identical types under identical contexts.
- Every misuse in §5.4 fails at compile time with a targeted message; no input silently
  reinterprets AgL syntax as payload or vice versa.
- `just check` passes: lint, full test suite with 100 % coverage, strict mypy.

## 11. Risks and edge cases

- **Layout interaction**: shell keywords (`done`, `else`) or arbitrary prompt prose at the start
  of block lines must never reach the layout pass — guaranteed structurally by scanning blocks
  wholly inside the raw-tail mode; covered by dedicated lexer tests.
- **`tests/**/*.py` migration precision**: `${` occurrences in Python test files must be verified
  as AgL snippets (not Python f-string braces or shell fixtures) before rewriting; done file by
  file, backed by the full suite.
- **Rendering round-trip**: the render quote-escape switch (`\$`→`\%`) changes expected outputs in
  rendering tests and any goldens; S1 updates them together with the scanner change so the
  round-trip property (rendered text re-parses) never regresses.
- **REPL continuation**: block-form detection in the console is the only piece outside the
  lexer/parser; it is isolated in the REPL layer and covered by e2e tests.
- **Registry purity**: the raw-tail seam must stay free of builtin-specific behavior (trigger,
  dedent, and desugaring parameterized by builtin name only), so future adopters — or a later
  `ask!` agent slot in the prefix seam (D8) — are opt-ins plus their own design decisions, not
  refactors.
