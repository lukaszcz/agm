# AgL Module System — Implementation Plan

Status: planned · Date: 2026-06-19 · **Every** design decision below is owner-approved.

## 1. Goal

Add a **file-based module system** to AgL: one module per file, identified by a dotted logical
name resolved against configurable roots. Support qualified access (`module::name`), selective
imports with renaming, import-hiding, and namespace-wildcard imports (`module.*`). The design is
principled and general — names behave uniformly regardless of which module declares them.

This plan was produced after settling **every** architectural decision with the owner (no point
left "to confirm"). Section 2 is the authoritative spec; the rest is the implementation
breakdown. Section 3 lists explicitly-approved non-goals.

## 2. Settled design (authoritative spec)

### D1 — Module identity & location
- A module is identified by a **dotted logical name** (`foo.bar.baz`) mapped to a file:
  `foo.bar.baz` → `<root>/foo/bar/baz.agl`. The **only** accepted extension is `.agl`.
- **Module dot-paths form one global namespace.** A module id must resolve to **exactly one**
  file across **all** roots. If the same id exists under more than one root, that is an
  **ambiguity error** — there is *no* first-root-wins shadowing.
- Consequently **roots are an unordered search set**, not an ordered precedence list (order is
  irrelevant: an id is found in at most one root, else error/not-found).
- Unordered semantics do not imply unstable output: canonical roots are sorted in search-list
  diagnostics; wildcard matches are sorted by logical `ModuleId` (then canonical path when
  reporting conflicts); ambiguity candidates are sorted by displayed qualifier. Graph traversal
  order has no semantic effect.
- Roots:
  1. `agm exec -c <code>`: the **current working directory**.
  2. `agm exec <file>`: the **file's directory**.
  3. A **global library root**, default `~/.agm/lib`, configurable.
  4. Additional roots via AGM config **and** a repeatable `agm exec -I/--module-path DIR` flag
     (durable settings in config; ad-hoc roots on the command line — also how e2e/fixture tests
     point at test roots).
- CLI module paths are resolved relative to the invocation cwd. A configured relative path is
  resolved relative to the config file that declares it, so config loading must retain each
  path's origin until resolution. All roots are user-expanded, made absolute, canonicalized, and
  deduplicated before use.
- Candidate module files are canonicalized before identity comparison. The same canonical file
  reached through duplicate, nested, or symlinked roots counts once; one module id resolving to
  different canonical files is ambiguous. An import resolving to the entry file's canonical
  identity is rejected as an attempt to import the entry.
- "Module not found" lists every root searched.

### D2 — Qualifier syntax
- `::` separates the **module boundary**; `.` is used **inside** module paths and for member
  access / enum-variant qualification (unchanged).
- Examples: `foo.bar.baz::thing`, `foo.bar::Color.Red`, `foo.bar::thing.field`.
- `::` (DCOLON) already exists in the lexer. `.` continues to terminate identifiers; module
  paths are assembled by the grammar from `name (DOT name)*`, never lexed as one token.

### D3 — Import declarations

Grammar shape:
```
import MODPATH[.*] [qualified] [as ALIAS] [ using ITEM (, ITEM)*  |  hiding NAME (, NAME)* ]
ITEM = NAME [as NEWNAME]
```
Single-module forms (owner-confirmed):
```
import foo.bar                      # all exports unqualified; also qualified foo.bar::x
import foo.bar as A                 # all exports unqualified; qualified ONLY via A::x
import foo.bar qualified            # nothing unqualified; qualified via foo.bar::x
import foo.bar qualified as A       # nothing unqualified; qualified via A::x
import foo.bar using x, y           # only x, y (unqualified + qualified)
import foo.bar hiding x, y          # all except x, y (unqualified + qualified)
import foo.bar using x as X, y      # import x renamed to X (unqualified X, qualified A/foo.bar::X)
import foo.bar qualified using x, y
import foo.bar qualified as A using x, y
import foo.bar qualified hiding x, y
```

Core semantics:
- **Open by default.** Bare `import foo.bar` brings *all* exported names in unqualified and keeps
  the module reachable qualified.
- **Imported set S** = `using` list / complement of `hiding` list / all exports. **S bounds
  qualified access too** — with `using x, y`, `foo.bar::z` is an error for an unimported `z`.
- **`qualified`** suppresses the unqualified injection; names in S remain reachable via the
  qualifier.
- **`as A` replaces the qualifier** (only `A::x`; `foo.bar::x` no longer works). **Aliases may be
  any case** (`as A` or `as fb`).
- **Renames are canonical.** `using x as X` makes `X` the name everywhere: unqualified `X`,
  qualified `A::X` / `foo.bar::X`.
- **No clash at import; clash only on *use*.** Two imports exporting the same `x` are legal; an
  *unqualified* reference to an ambiguous `x` is an error **at the reference site**, suggesting
  the disambiguating qualifiers. Qualification always disambiguates.

Namespace-wildcard imports (`import foo.*`):
- **`import foo.*`** imports **every module in the `foo` subtree**: the module `foo` itself
  (`foo.agl`, if present) and every module whose dot-path is under `foo.` recursively
  (`foo.bar` → `foo/bar.agl`, `foo.bar.baz` → `foo/bar/baz.agl`, …).
- A wildcard prefix that matches no module is a located **module-prefix-not-found** error, not an
  empty import.
- **Spans all roots.** Because dot-paths are one global namespace (D1), `foo.*` collects every
  matching module wherever its file physically lives, unioned (no conflict possible — duplicate
  ids are already an error).
- **Subtree expansion.** `import foo.*` expands over every matched module: open → each module's
  names unqualified (clash-on-use) plus full-path qualified; `qualified` → full-path qualified
  only. `using` selects each listed name from every matched module that exports it; `hiding`
  removes each listed name wherever it is exported. For either clause, every listed source name
  must be exported by at least one matched module, otherwise it is a located non-exported-name
  error. A matched module lacking a listed name is not itself an error.
- **`as A` on a wildcard re-roots the matched prefix.** `import foo.bar.* as A` replaces prefix
  `foo.bar` with `A`: `foo.bar` → `A::x`, `foo.bar.baz` → `A.baz::y`. This introduces
  **alias-rooted multi-segment qualifiers** (`A.baz::y`).

Merge & conflict policy:
- **Lenient merge.** Multiple `import` declarations for the **same module** union their effects
  by normalizing each declaration into unqualified exposed-name → target-binding entries and
  qualifier-handle + exposed-name → target-binding entries. All aliases, handles, and renames
  remain valid; unqualified injection comes from every open declaration. Identical mappings are
  idempotent. An unqualified exposed name with multiple distinct targets remains a deferred
  use-site ambiguity; a qualified key mapping to different targets is a static conflict, preserving
  the rule that qualification always disambiguates.
- **Static errors:**
  - a single-module alias binds its complete qualifier to one module and cannot be rebound to a
    different module (`import foo as A` + `import bar as A`); a wildcard alias instead binds a
    namespace prefix, with descendant qualifier paths derived by prefix replacement; two imports
    conflict only if they map the same complete qualifier path to different modules; overlapping
    alias prefixes are otherwise legal, and repeated mappings from the same complete qualifier to
    the same module are idempotent;
  - an **alias name equal to a module-path root segment used elsewhere** (`import foo` +
    `import bar.baz as foo` → ambiguous `foo::x`).

### D4 — What a module may contain
- Imports are **top-level, header-only, and module-wide**. A module header contains imports and,
  in the entry module only, `config` pragmas; these may be interleaved. An import after the first
  ordinary declaration/executable item, or inside a block, is a scope error.
- After its header, an **importable module is declaration-only**: `def`, `record`, `enum`, and
  `type` (with optional `private`). Top-level bare expressions, `let`/`var`,
  `print`/`exec`/`ask`, and agent calls are a **scope error** inside an imported module.
- Imports therefore **never execute code** — no side effects, no cost, order-independent.
- The **entry file** keeps today's mixed freedom (declarations + executable statements).

### D5 — Visibility
- **Public by default + `private`**, two levels only (no export lists, no `protected`).
- `private` declarations are usable within their module but **invisible** to other modules — not
  importable, not qualifiable, excluded from `using`/`hiding`/open/wildcard import, absent from
  the export set. Applies to `def`/`record`/`enum`/`type`.

### D6 — Whole-program declarations
- `program`, `config`, `param`, and `agent` are **entry-file-only**; each is a scope error inside
  an imported module. Libraries parameterize via **function arguments**, including agent-typed
  arguments when they need to invoke a program-owned agent.

### D7 — Agent ownership
- Agents belong to the entry program, not importable modules. `agent` declarations in imported
  modules are a scope error; agents are not exported, imported, qualified, renamed, or `private`.
- The host agent registry and `[exec.agents]` configuration retain their existing unqualified
  program-owned keys. Runner specs remain at the entry declaration site (`agent NAME = "..."`).

### D8 — Cyclic imports
- **Allowed** (= cross-file mutual recursion; safe because modules are declaration-only).
- Loader uses **load-all-reachable-then-resolve**, with whole-program resolve/typecheck pre-passes.
  (Survey-confirmed: cycles are clean in whole-unit, no-import-execution models —
  Java/C#/Rust-intra-crate — and hazardous only under import-time execution (Python) or separate
  compilation (Go/Haskell-boot), neither of which applies here.)

### D9 — Entry identity & self-reference
- **`::name`** (empty module prefix) refers to `name` in the **current** module, for any name.
  A reference is `[MODQUAL] :: name`, `MODQUAL` optional (omitted = this module). This is the
  uniform self-disambiguation tool in every module.
- The **entry program is a non-importable program root** (internal reserved `main` identity, used
  only for keying — never spelled by users). `exec <file>` and `exec -c` behave identically.

### D10 — Name space for import/qualification
- **One unified namespace.** Any exported top-level value *or* type may be listed in
  `using`/`hiding` and qualified via `::`, identified by its plain identifier (case distinguishes
  type from value). Entry-program agents remain first-class values but are not module exports.
  **Enum variants travel with their enum** — not
  separately importable; reached as `foo::Color.Red` (or bare `Color.Red`/`Red` under existing
  constructor-resolution rules).

## 3. Non-goals (v1) — explicitly approved

All confirmed with the owner as out of scope for v1 (each is additive / can come later):
- **Re-exports** (a module republishing imported names). Deferred — additive (`public import …`
  later) with no breakage.
- **Conditional / dynamic (runtime) imports.** Imports are static, literal, load-time only.
- **Directory-index modules.** A directory is never itself a single module; the `import foo.*`
  wildcard (D3) covers "import the whole subtree."
- **Package management / versioning** for the lib root. The lib root is a plain directory AgM
  only resolves from.
- **Visibility beyond public/`private`** (no export lists, no `protected`).

## 4. Affected code map

Grounded in `src/agm/agl/`:
- `lexer/` — soft keywords `import`, `qualified`, `using`, `hiding`, `private` (`as` exists);
  ensure `::` usable in reference position; `.*` wildcard tail. (`tokens.py`, `scanner.py`,
  `lexer.py`)
- `grammar/agl.lark`, `parser/transform.py` — import decl (incl. `.*` and any-case alias),
  qualified refs, `::name`, alias-rooted multi-segment qualifiers, `private`, module-qualified
  type/constructor heads.
- `syntax/nodes.py`, `syntax/types.py`, `syntax/spans.py` — `ImportDecl` (with
  `wildcard: bool`), qualifier on refs/constructors/types, `is_private` flags, source-aware spans,
  module-unit / module-graph nodes.
- **New** `modules/` package — `ids.py` (ModuleId), `roots.py` (root set assembly), `resolver.py`
  (ModuleId→path; global-uniqueness check; wildcard glob expansion across roots), `loader.py`
  (parse graph, dedup, cycles, node-id seeding).
- `scope/resolver.py`, `scope/symbols.py` — per-module symbol tables, exports/`private`, import
  semantics (single + wildcard + merge), open-by-default, clash-on-use, `::name`, declaration-only
  & entry-only enforcement, whole-program pre-passes; `BindingRef` gains owning ModuleId;
  `ModuleResolution` → graph form.
- `typecheck/checker.py`, `typecheck/env.py`, `typecheck/types.py` — module-qualified type
  identity, whole-program type pre-pass, qualified type refs.
- `eval/interpreter.py`, `eval/scope.py` — per-module top-level frames, statically-resolved
  cross-module references, mutual recursion across files.
- `runtime/agents.py`, `runtime/runtime.py` — program-owned agent handling across the program-level
  runtime; new load-and-prepare entrypoint (entry source + entry path + roots).
- `repl/session.py` — imports in the REPL (load from cwd root); `::name` = REPL main module.
- `src/agm/config/` — lib-root + extra-roots configuration.
- `src/agm/commands/` (exec), `src/agm/cli.py`, `src/agm/completion.py` — `-I/--module-path`
  option + completion.
- Docs: `docs/agl/reference/` for user-facing language semantics, `docs/arch/agl.md` for the
  program-level implementation architecture, `docs/agl-grammar.md`, `README.md` (brief),
  `docs/commands.md` (full), and help texts. Update the relevant reference and architecture docs
  in the milestone that changes them; M6 performs the final consistency review.

## 5. Syntax & AST details

### 5.1 Lexer
- Add the five soft keywords as **contextual** keywords (avoid breaking existing identifiers);
  `as` already exists (cast operator — same token, different position).
- No change to `.` handling. Module paths and the `.*` tail are assembled at the grammar level.
- Confirm `::` accepted as a leading token (`::name`) and between qualifier and name.

### 5.2 Grammar (agl.lark)
```
import_decl  : "import" import_path ["qualified"] ["as" alias] [import_clause]
import_path  : module_path ["." "*"]              // ".*" => wildcard over the subtree
module_path  : VAR_NAME ("." VAR_NAME)*
alias        : VAR_NAME | TYPE_NAME                // any-case alias
import_clause: "using" import_item ("," import_item)*
             | "hiding" ref_name ("," ref_name)*
import_item  : ref_name ["as" ref_name]

// qualifier before "::": unresolved raw segments, OR empty (current module).
// TYPE_NAME may lead an any-case alias; actual alias-vs-module-path classification is semantic.
mod_qualifier: (VAR_NAME | TYPE_NAME) ("." VAR_NAME)*
qualified_ref: [mod_qualifier] DCOLON ref_name
ref_name     : VAR_NAME | TYPE_NAME
```
- Integrate `qualified_ref` as a primary expression and as a type/constructor head so
  `foo.bar::Color.Red`, `foo.bar::MyType`, `A.baz::y`, and `::name` parse, composing with the
  existing `Color.Red` and postfix `.field`.
- Alias-vs-module-path before `::` is resolved at **resolve time** (alias table lookup); the
  alias/module-root collision (D3) is a static error, so this is unambiguous.
- `private` is a leading modifier on the four importable declaration rules.
- Keep LALR(1) conflict-free (currently 0/0). The optional `mod_qualifier`, optional `as alias`,
  and `.*` tail are the risk spots — adjust factoring as needed.

### 5.3 AST (nodes.py)
- `ImportDecl(module_path: tuple[str, ...], wildcard: bool, qualified: bool, alias: str | None,
  mode: ImportMode (ALL|USING|HIDING), items: tuple[ImportItem, ...], span)`;
  `ImportItem(name, rename | None)`.
- `Qualifier{segments: tuple[str, ...]}` preserves the raw syntactic qualifier, where
  `segments == ()` = current module (`::name`). Alias-rooted qualifiers carry the alias + trailing
  segments, but the AST does not classify a qualifier as an alias or module path; scope resolution
  records the resolved module in its side tables. Reference nodes (`VarRef`, `Constructor`, type
  refs) gain optional `qualifier: Qualifier | None`.
- `is_private: bool` on the four importable declaration node types (`def`, `record`, `enum`,
  `type`).
- `SourceSpan` gains a stable source identity in addition to source-relative coordinates. File
  modules use their canonical path identity; inline command and REPL sources use synthetic
  identities. Diagnostics and runtime errors derive their displayed source from the span, so
  attribution survives every pipeline stage without formatter-side inference.
- Keep module parse → a `Module` node (imports + body). Loader assembles a `ModuleGraph`
  (`{ModuleId: Module}` + entry id + SCC/topo info) consumed downstream.

## 6. Module loader (new `modules/` package)

- `ModuleId` = tuple of segments; reserved entry id = `main`.
- `roots.py`: assemble and canonicalize the **unordered** root set from invocation context +
  origin-aware config paths + `-I` flags; deduplicate canonical roots.
- `resolver.py`:
  - single id → search all roots for `<root>/<segments>.agl`; canonicalize and deduplicate matching
    files; **exactly one** canonical match required (zero → not-found listing roots; ≥2 →
    ambiguity error).
  - wildcard prefix → glob `<root>/<prefix>.agl` and `<root>/<prefix>/**/*.agl` across **all**
    roots; map files to dot-path ids; enforce global uniqueness per id.
- `loader.py`:
  1. Parse entry source (path or `-c`) → entry `Module`; collect `ImportDecl`s.
  2. DFS/BFS over imports, resolving ids to files, parsing with `parse_program_seeded(text,
     start_id=…)` for globally-unique node IDs (disjoint ranges per module). Terminate when an id
     is already loaded — this makes **cycles** finite and safe.
  3. Produce a `ModuleGraph`; retain cycles (D8); compute SCCs only for diagnostics.
  4. Reject any import whose canonical file identity equals the entry file, regardless of the
     logical module id used to reach it.
  5. Errors: not-found (list roots); ambiguous id; parse error (file + span); attempt to import
     the entry (D9); duplicate-alias / alias-root collision (D3).

## 7. Name resolution (scope/)

- Build each module's **export table** (top-level names minus `private`).
- Build each module's **top-level environment**: own decls (incl. `private`) + imports. For each
  `ImportDecl` (single or wildcard-expanded per module), compute S, the unqualified bindings
  (unless `qualified`, applying renames), and the qualifier handle (alias replaces; alias-rooted
  for wildcard re-rooting). **Merge** multiple imports by unioning their normalized unqualified
  and qualified binding views, retaining every exposed-name → original-binding mapping.
- **Reference resolution**:
  - `MODQUAL::name` → look up `name` in that module's imported set S (S bounds qualified access);
    error if not in S or `private`. Alias-rooted qualifiers resolve via the alias table.
  - `::name` → current module's top-level decl `name`.
  - bare `name` follows explicit precedence: the nearest lexical binding wins, then the current
    module's own top-level declaration, then open-imported candidates, then contextual builtins.
    Multiple distinct open-import candidates are retained and produce an *ambiguous reference*
    diagnostic only at an unqualified use-site, listing disambiguating qualifiers. An own
    declaration therefore shadows an imported name; the import remains available qualified.
- **Whole-program pre-passes** collect every module's functions and types, plus the entry program's
  agents, before resolving any body → cross-file mutual recursion (D8).
- **Enforcement**: declaration-only in non-entry modules (D4); `program`/`config`/`param`/`agent`
  entry-only (D6); imports top-level and header-only (D4); `private` boundary (D5);
  duplicate-alias / alias-root collision static errors (D3).
- `BindingRef` gains owning `ModuleId`; `ModuleResolution` becomes a graph-shaped
  `ResolvedProgram` preserving per-node side tables keyed by global node IDs.

## 8. Typecheck & eval

### 8.1 Typecheck
- Type identity is **module-qualified**: `RecordType`/`EnumType` carry their owning `ModuleId`
  (distinct `foo::Color` vs `bar::Color`). Whole-program type pre-pass collects all type decls
  (cycles allowed), then checks bodies. Qualified type refs resolve via S.
- Agent typing remains program-owned and unqualified; imported functions may accept agent-typed
  arguments without declaring or importing agents.

### 8.2 Eval / runtime
- Build **per-module top-level frames** (closures and type bindings), plus entry-program agent
  values — pure, no side effects. References (statically resolved to `(ModuleId, name)`) fetch
  from the right frame; cross-file mutual recursion works once all frames exist.
- Execute **only the entry module's block**.
- `AgentRegistry` keeps its existing program-owned keys; only entry-program agent declarations
  build runners. REPL agent declarations and `ConfirmingAgent` wiring remain program-local.

## 9. Host integration, config, CLI

- New entrypoint, e.g. `PipelineDriver.prepare_program(entry_source, *, entry_path, roots)`:
  loader → resolve → typecheck/eval. Existing module callers (no imports) behave identically.
- `agm exec <file>`: file's dir is a root; entry id = `main`. `agm exec -c <code>`: cwd is a root;
  entry id = `main`.
- Lib root default `~/.agm/lib`, configurable; extra roots via config and repeatable
  `-I/--module-path`. Add completion for the new option.
- REPL: resolve imports against the cwd root; `::name` targets the REPL main module;
  declaration-only restriction does **not** apply to REPL input (interactive, like the entry).

## 10. Diagnostics

Every located diagnostic and runtime error carries the originating `SourceSpan`, including its
source identity. Located messages for: module-not-found or module-prefix-not-found (roots
searched); ambiguous module id (multiple roots);
ambiguous unqualified reference (candidates + suggested qualifiers); reference to a `private`
name; qualified reference outside the imported set S; `using`/`hiding` naming a non-exported name;
declaration-only violation in an imported module; `program`/`config`/`param` in a non-entry
module; `agent` in a non-entry module; import outside the top-level header; attempt to import the
entry; duplicate alias to different modules; alias-equals-module-root collision.

## 11. Testing strategy (TDD, 100% coverage, 100% e2e command coverage)

Write failing tests first; implement to green. Group test files by module/category.
- **Lexer/parser**: every import form; `.*` wildcard; any-case aliases; `MODQUAL::name`,
  `::name`, alias-rooted `A.baz::y`; `foo.bar::Color.Red`; `private`; grammar conflict-free.
- **Loader/resolver (modules/)**: id→path across the root set; not-found; **ambiguous id across
  roots**; wildcard glob across roots; empty wildcard error; graph build; dedup; **cycles**
  terminate & link; node-id disjointness; deterministic ordering across shuffled root and
  filesystem discovery order.
- **Scope**: open import; `using`/`hiding`/`qualified`/`as`/rename; S bounds qualified; clash
  deferred to use-site; **merge of multiple imports**; duplicate-alias / alias-root collision
  errors; `::name`; `private` boundary; declaration-only & entry-only enforcement; wildcard
  subtree expansion + re-rooting, including compatible/conflicting overlaps; cross-file mutual
  recursion.
- **Typecheck**: distinct module-qualified type identity; qualified type refs; whole-program
  pre-pass with cycles.
- **Eval**: qualified value dispatch; cross-module + mutually-recursive calls; passing
  entry-program agent values into imported functions.
- **Integration/e2e/system**: multi-file projects via `agm exec` (incl. a lib-root module and
  `-I` roots); **multiple input/mock-response scenarios** per program (not one combination);
  wildcard-import program; REPL import session. Maintain 100% `src/` line coverage and 100% e2e
  command coverage.

## 12. Milestones

Each compiles, passes `just check`, and is committed (`type: subject`) before the next. Within a
milestone, decompose into <200k-token subagent tasks (≤5 parallel implementers), Opus review per
task, Fable review per milestone; fix every reviewer finding.

- **M1 — Surface syntax**: lexer soft keywords + `.*`; grammar (import decl incl. wildcard &
  any-case alias, qualified refs, `::name`, alias-rooted qualifiers, `private`); AST + transform.
  Parser/AST tests; grammar conflict-free; update reference grammar and lexical/syntax docs.
- **M2 — Module loading**: `modules/` package (ids, roots, resolver with global-uniqueness +
  wildcard glob, loader), roots config + `-I` flag, node-id seeding, cycle-safe graph. Loader unit
  tests with fixture roots; update `docs/arch/agl.md` for graph loading and source-aware spans.
- **M3 — Name resolution**: per-module symbol tables, exports/`private`, full import semantics
  (single + wildcard + merge), open-by-default, clash-on-use, `::name`, declaration-only &
  entry-only enforcement, duplicate/collision errors, whole-program pre-passes. Resolver tests;
  update module, visibility, import, and scope reference docs plus resolver architecture.
- **M4 — Typecheck**: module-qualified type identity, qualified type refs, whole-program type
  pre-pass; update type reference and typechecker architecture docs.
- **M5 — Eval & host**: per-module frames, qualified/mutually-recursive dispatch, program-owned
  agent registry, `agm exec` module loading. Eval tests + first multi-file e2e (multi-scenario);
  update execution, agent, and evaluator/runtime architecture docs.
- **M6 — REPL, CLI, docs, polish**: REPL imports, `-I/--module-path` + completion, full
  diagnostics, final consistency review across the language reference and architecture docs,
  README / commands.md / grammar.md / help texts, coverage to 100%. Final `just check`.

## 13. Done criteria

`just check` green; 100% `src/` coverage; 100% e2e command coverage; docs and help texts updated;
all settled decisions (D1–D10) honored exactly; non-goals (§3) excluded.
