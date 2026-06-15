# Plan: Require explicit agent declarations in AgL

## Overview

Today AgL has **no agent-declaration construct**. An agent call (`reviewer "…"`)
stores the callee as a bare string (`AgentCall.agent`), the scope pass never
validates it, and the typecheck pass only rejects an unknown name when the host
advertises `has_fallback_agent = False`. Because `agm exec` installs a
runner-backed *default* agent, it sets `has_fallback_agent = True`, so **any**
agent name resolves implicitly with no declaration anywhere — in the source or
in config.

This plan inverts ownership of the agent name set: **the source program declares
which agents exist**, an undeclared agent name is a **static binding error**, and
the host (config / library registration) only supplies *backings* for
already-declared names. A declaration may optionally carry a runner-command hint,
resolved exactly like config runner strings (`%%` / `%{PROMPT_FILE}`
placeholders), which config can override.

## Decisions

### Resolved with the owner

1. **Source owns the name set; host only backs.** Every called named agent MUST
   be declared in the source with an `agent` declaration. Calling an undeclared
   name is a static binding error. `register_agent(name, fn)` / `[exec.agents.name]`
   merely supply a *backing* for an already-declared name; registering a name
   that the program never declares is a host error.

2. **`prompt` and `exec` stay built-in contextual keywords** — never declared,
   never reserved as agent declarations. `prompt` resolves to the default agent;
   `exec` to shell execution, exactly as today.

3. **A declaration may optionally specify a runner string** that resolves like a
   config runner command (supports `%%` and `%{PROMPT_FILE}` placeholders;
   substituted with the rendered prompt-file path by the host). Bare declaration
   ⇒ no source backing.

4. **Config overrides source.** When both a config `[exec.agents.<name>]` entry
   and a source runner string exist for the same agent, config wins. Full
   precedence (high → low):

   ```
   [exec.agents.<name>]   (config, per-agent)
   source `agent` runner string
   --runner flag
   [exec] runner          (config)
   [loop] runner          (config)
   claude -p              (built-in default)
   ```

### Minor decisions taken with recommended defaults (flag if you disagree)

5. **Syntax:** `agent NAME` (bare) and `agent NAME = "runner string"`. Reuses the
   `=` form already used by `type X = …`. *(Alternative: `:` like `input`; or an
   indented attribute block for future model/timeout knobs. Chosen `=` for the
   minimal grammar change that satisfies the requirement.)*

6. **Runner string is a static string literal.** It is parsed as an AgL string
   but must contain **no `${…}` interpolation holes** — interpolation in a runner
   declaration is a static error. It is an opaque host hint: scope/typecheck do
   not interpret it; only the `exec` host consumes it (placeholder substitution,
   no env-var expansion — matching today's `[exec.agents]` behavior).

7. **Root-level only.** Like `input`, `agent` declarations must appear at program
   root (not nested in `if`/`do`/`try`). Declaring one inside a block is a static
   error (new rejection test, mirroring `input_not_root.agl`).

8. **Duplicate declaration of the same agent name is a static error** (mirrors
   `input_redeclared`). Declaring `agent prompt` / `agent exec` is a static error
   (reserved names).

9. **Agent names occupy a separate namespace from variables.** Agent calls are
   syntactically distinct (`name template`), so an `agent impl` declaration does
   not collide with a `let impl`/`input impl` binding. *(We keep them separate
   rather than reserving the name across both namespaces, to avoid breaking
   programs that legitimately reuse an identifier. Flag if you prefer a single
   shared namespace.)*

10. **Declared-but-uncalled agent ⇒ warning, not error** (consistent with the
    host's existing "useless construct" warnings; never blocks execution).

11. **A declared agent with no resolvable backing is a pre-execution host error.**
    Under `agm exec` every declared agent always resolves (default runner is the
    floor), so this only bites the library/test host: a declared agent that is
    neither `register_agent`-ed nor covered by a `default_agent` is reported
    before any statement runs (consistent with "host config errors execute
    nothing"). Flag if you'd rather defer this to a catchable runtime
    `AgentCallError` at the call site.

## Current architecture (what changes)

```
source → lexer → parser → AST(AgentCall.agent: str)
  → scope:  CallKind only; agent name NOT validated          ← ADD: declared-agent table + binding check
  → typecheck: reject unknown name iff !has_fallback_agent    ← REMOVE name-existence check (scope owns it)
  → host caps: agent_names + has_fallback_agent(=has default) ← RETIRE has_fallback_agent for name validity
  → runtime dispatch: named → default fallback → KeyError     ← fallback now backs only declared names
  → agm exec: default_agent = runner factory (any name)       ← register each declared agent w/ resolved runner
```

## Target design

### 1. Lexer (`src/agm/agl/lexer/tokens.py`)

- Add `KW_AGENT = "agent"` and include `"agent"` in the `KEYWORDS` frozenset.
- `agent` becomes a **reserved keyword** (cannot be a variable/type name).
- Verify no existing test program/identifier uses `agent` as a name (grep; none
  expected). The lexer remap (`lexer.py`) auto-uppercases the terminal for Lark.

### 2. Grammar (`src/agm/agl/grammar/agl.lark`)

- Add to `closed_stmt`: `agent_decl`.
- Rule: `agent_decl: "agent" VAR_NAME (EQ template)?`
  - Reuse `template` for the runner string so the existing string lexing/quoting
    applies. The transformer rejects a template with interpolation holes.
  - *(If we later want attributes, this rule is the extension point.)*

### 3. AST (`src/agm/agl/syntax/nodes.py`)

- New frozen dataclass:

  ```python
  @dataclass(frozen=True, slots=True)
  class AgentDecl:
      name: str
      runner: str | None          # static runner string; None = bare declaration
      span: SourceSpan = dc_field(compare=False)
      node_id: int = dc_field(compare=False)
  ```

- Add `AgentDecl` to the `Stmt` union.
- Transformer (`parser/transform.py`): new `agent_decl()` method. Extract
  `VAR_NAME` → `name`. If a `template` is present, require it to be a static
  string (no `TemplateHole`); raise a parse/transform diagnostic otherwise; store
  its literal text as `runner`.

### 4. Scope pass (`src/agm/agl/scope/resolver.py`)

This is where the **binding error** lives.

- **Pre-pass to collect declarations.** Like `input`, agent declarations are
  root-level. Add a first pass over `program.body` that collects `AgentDecl`s
  into a `declared_agents: dict[str, AgentDecl]` (name → decl).
  - Duplicate name ⇒ `AglScopeError` ("agent 'X' is already declared").
  - `name in {"prompt","exec"}` ⇒ `AglScopeError` ("'prompt'/'exec' is built-in
    and cannot be declared as an agent").
  - `AgentDecl` appearing in a non-root scope ⇒ `AglScopeError` (root-only).
- **Validate every agent call.** In `_resolve_agent_call`, when
  `kind == CallKind.agent` and `node.agent not in declared_agents` ⇒
  `AglScopeError` ("unknown agent 'X'; declare it with `agent X`"). `prompt`/`exec`
  branches unchanged.
- Surface the declared set on `ResolvedProgram` (e.g.
  `declared_agents: frozenset[str]` or the full decl map with runner hints) so
  typecheck/host can consume it without re-walking.
- Emit a **warning** for any declared agent never referenced by a call
  (non-fatal), alongside existing warnings.

### 5. Typecheck pass (`src/agm/agl/typecheck/checker.py`)

- **Remove** the `has_fallback_agent` / `agent_names` existence check for
  `CallKind.agent` (lines ~794-803) — scope now guarantees the name is declared.
- **Keep** the `CallKind.default_agent` (`prompt`) check: still needs a default
  or fallback agent or it cannot run.
- The runner-hint string is opaque to typecheck (no validation beyond
  "is a static string", enforced at transform time).

### 6. Host capabilities (`src/agm/agl/capabilities.py`)

- `has_fallback_agent` no longer governs **name validity** (scope does). Two
  options; recommend (a):
  - **(a) Retire `has_fallback_agent`** from `HostCapabilities`. Keep
    `has_default_agent` (drives the `prompt` check) and `agent_names` (the set of
    host-supplied backings, used for the "register undeclared name" host-error
    check and for the static inventory). Update the registry's capability
    construction and all references.
  - (b) Keep the field but document it as backing-only; riskier (dead lever).
- `agent_names` stays: the set of names the host can *back*. The runtime cross-
  checks this against the source-declared set (see §8).

### 7. Runtime registry & dispatch (`src/agm/agl/runtime/agents.py`, `runtime.py`)

- `register_agent` unchanged in mechanics (reserved-name + duplicate guards stay).
- `AgentRegistry.dispatch`: the default-agent fallback remains, but it now only
  ever fires for **declared** names (scope guarantees no undeclared call reaches
  runtime). Reframe its docstring: the fallback is the documented backing for a
  *declared* agent that has no dedicated registration, not an implicit name
  resolver.
- `WorkflowRuntime.run(...)`: after the static pipeline produces the declared
  agent set, enforce the host/source contract (see §8) before execution.

### 8. Source↔host reconciliation (in `WorkflowRuntime.run`)

After parse+scope yields `declared_agents` and before execution:

- **Registered-but-undeclared ⇒ host error.** Any `register_agent` name not in
  `declared_agents` is a pre-execution host error ("agent 'ghost' is registered
  but never declared in the program").
- **Declared-but-unbacked ⇒ host error** *(per decision 11)*: a declared agent
  with no registration and no `default_agent` is reported pre-execution.
- Both are reported on the same channel as input-validation/host-config errors:
  diagnostics set, `result.ok = False`, nothing executes.

To expose the declared set to the `exec` host **without re-running the whole
pipeline twice**, add a lightweight API:

```python
class WorkflowRuntime:
    def declared_agents(self, source: str) -> tuple[AgentDeclInfo, ...]:
        """Parse + scope only; return declared agents with optional runner hints."""
```

(`AgentDeclInfo` = name + runner hint + span.) This also feeds the existing
"static call inventory" dry-run feature.

### 9. `agm exec` integration (`src/agm/commands/exec.py`, `config/general.py`)

- Resolve the default runner as today (`--runner` > `[exec] runner` >
  `[loop] runner` > `claude -p`).
- Call `runtime.declared_agents(source)` to get declared names + source runner
  hints.
- For each declared agent, compute its backing command by precedence
  (decision 4): `config.agents.get(name)` → source runner hint → default runner.
- **Register each declared agent explicitly** with a `runner_backed_agent_factory`
  bound to its resolved command (or build a single factory whose `per_agent_cmds`
  is the merged map `{**source_hints, **config.agents}` so config wins, and
  register each declared name against it).
- Keep `default_agent` = runner-backed factory on the default runner, to back
  `prompt`.
- Net effect: no implicit fallback for unknown names; `exec` still "just works"
  for any declared agent, with or without config, because the default runner is
  the floor.
- `command_with_prompt_target` already substitutes `%%` / `%{PROMPT_FILE}`, so
  source runner hints get identical placeholder handling for free.

### 10. Config & templates

- No schema change to `[exec.agents]`; it remains a per-agent override map and now
  sits **above** source runner hints in precedence.
- Update the config template comment in `config/` and `docs/commands.md` exec
  section to document the new precedence and that agents must be declared in
  source.

## Documentation updates

- `docs/agl/reference/host-environment.md` — rewrite the **Agents** section:
  named agents are declared in source; the host supplies backings; no implicit
  fallback; describe the runner-hint precedence.
- `docs/agl/reference/program-structure.md` — add `agent` declarations to the
  top-level declaration forms and the execution-pipeline name-resolution step.
- `docs/agl/reference/bindings-and-scope.md` — document the agent namespace,
  root-only rule, duplicate/reserved errors, undeclared-call binding error.
- `docs/agl/reference/agent-calls.md` — show declarations preceding calls;
  `prompt`/`exec` need none.
- `docs/agl-grammar.md` — add the `agent_decl` production.
- `docs/commands.md` — exec precedence table + "agents must be declared".
- `README.md` — only if it mentions agent usage at a high level (keep brief).

## Test plan (TDD — write failing tests first)

Per repo policy, write red tests before implementing, and add a regression test
for the binding-error behavior. 100% coverage of `src/` must hold.

1. **Lexer** (`tests/test_agl_lexer.py`): `agent` tokenizes as a keyword; cannot
   be used as an identifier.
2. **Parser** (`tests/test_agl_parser.py`): `agent reviewer` → `AgentDecl(name,
   runner=None)`; `agent impl = "claude -p %{PROMPT_FILE}"` → runner set;
   interpolation hole in runner string → rejected.
3. **Scope** (`tests/test_agl_scope.py`): undeclared agent call → `AglScopeError`;
   duplicate `agent` → error; `agent prompt`/`agent exec` → error; non-root
   declaration → error; declared-but-unused → warning; `prompt`/`exec` calls need
   no declaration.
4. **Typecheck** (`tests/test_agl_typecheck.py`): declared agent call type-checks;
   `prompt` still requires a default/fallback agent.
5. **Runtime** (`tests/test_agl_runtime.py`): `register_agent` for an undeclared
   name → host error; declared-but-unbacked (no default agent) → host error;
   declared + registered → runs; `declared_agents()` API returns names + hints.
6. **Exec command** (`tests/test_exec_command.py`): precedence — config beats
   source hint beats default runner; bare declaration uses default runner;
   `%{PROMPT_FILE}` substitution in a source hint; undeclared call exits with the
   pre-execution code (1).
7. **E2E** (`tests/agl/`):
   - New `rejections/scope/undeclared_agent.agl` + `.expect.json`,
     `agent_redeclared.agl`, `agent_reserved_name.agl`, `agent_not_root.agl`.
   - Migrate **all** valid programs that call named agents to add `agent`
     declarations (≈24 scenario-bearing programs; e.g. `basics/bindings.agl` gains
     `agent impl`). Scenario `agents` map keys must match declared names.
   - Add a multi-scenario program exercising source runner hint vs config
     override (multiple input/mock combinations per the e2e harness convention).

## Migration impact

- **~24 `.agl` programs** with `agents` scenario maps need `agent` declarations
  added (mechanical: one `agent NAME` per scenario agent key, excluding `prompt`).
- The `agm exec` agent-wiring is restructured (explicit per-declared registration
  instead of a blanket fallback).
- `has_fallback_agent` removed from `HostCapabilities` and its construction.
- Public docs reference set (7 files) updated.

## Risks & mitigations

- **Portability concern.** Embedding a host runner string (`claude -p`,
  `%{PROMPT_FILE}`) in otherwise host-independent source couples the program to a
  host. *Mitigation:* the runner string is an **optional opaque hint** that
  non-`exec` hosts ignore (they back via `register_agent`); config always
  overrides it; core semantics (declaration + binding) stay host-independent.
- **Large test migration.** Touching every named-agent program risks churn.
  *Mitigation:* land grammar/AST/scope first behind red tests, migrate programs in
  one mechanical sweep, keep the e2e suite green (it is part of the standing gate).
- **Double parse in `exec`** (declared_agents + run). *Mitigation:* `declared_agents`
  runs only parse+scope (cheap vs. agent calls); revisit with a cached-parse
  entrypoint only if profiling warrants.

## Milestones

1. **Lexer + grammar + AST + transformer** for `agent_decl` (parser tests green).
2. **Scope**: declaration table, binding error, root-only/duplicate/reserved,
   unused warning; expose declared set on `ResolvedProgram` (scope tests green).
3. **Typecheck + capabilities**: drop name-existence check, retire
   `has_fallback_agent` (typecheck tests green).
4. **Runtime**: `declared_agents()` API + source↔host reconciliation errors
   (runtime tests green).
5. **`agm exec` + config**: per-declared registration, precedence, placeholder
   handling (exec tests green).
6. **E2E migration + new rejections + docs**; `just check` green; commit per
   milestone.
