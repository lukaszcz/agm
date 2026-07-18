# AgL Namespace and Module System Redesign

Status: **settled design** (all owner decisions made 2026-07-18); not yet implemented.
Supersedes the import/qualification parts of the current module system described in
`docs/agl/reference/modules.md`. Module loading, roots, cycles, declaration-only library
modules, visibility, and re-export identity semantics carry over unchanged except where
noted.

## Goals

- Stop overloading `.`: field access and module paths become disjoint syntaxes.
- Qualified-by-default imports with minimal-ceremony short qualification.
- One clash philosophy everywhere: *declaring* potential ambiguity is fine; *using* an
  ambiguous name is a loud static error. Nothing may ever re-resolve silently.
- A consistent, principled rule set with graded, one-line repairs for every collision.

## Decision summary

| # | Decision |
|---|----------|
| 1 | Module paths use `/` (`import std/config`); `.` is field access only; `::` is qualification only. |
| 2 | Imports are qualified by default; bare names come only from `using` lists, `open import`, or the prelude. |
| 3 | Qualified references resolve by **unambiguous path suffix** over the imported modules, with hard use-site ambiguity errors and a leading-`/` exact-path anchor. |
| 4 | `using`/`hiding` keep the symmetric one-set rule: an import contributes a set S bounding both bare injection and qualified access; same-module imports union. |
| 5 | `open` is prepositive: `open import std/core`. The `qualified` keyword is removed. |
| 6 | Wildcard imports stay, respecified as per-module expansion sugar; handle re-rooting is dropped. Local opens are deferred. |

## Module identity

A module is a file; its identity is its **slash-path** — the file path relative to a
library root with the `.agl` suffix stripped (`utils/strings.agl` → `utils/strings`).
Global uniqueness across the unordered root set is unchanged: zero matches is
"not found", two distinct files is "ambiguous", no shadowing. The entry module still has
no path of its own. Extern (`.py`) companion derivation is unaffected.

## Import declarations

```ebnf
import_decl ::= ["open"] "import" module_path ["/" "*"]
                ["as" ref_name]
                [using_clause | hiding_clause]

module_path   ::= NAME ("/" NAME)*
using_clause  ::= "using" import_item ("," import_item)*
hiding_clause ::= "hiding" ref_name ("," ref_name)*
import_item   ::= ref_name ["as" ref_name]
```

Import paths are always **absolute** slash-paths resolved against the roots (no leading
`/`, no suffix matching in import position). Header-position-only placement, soft
keywords, and the import-before-items rule are unchanged. `open` combined with `using`
is a static error (redundant — `using` already injects its names).

### The one-set rule

Every import declaration contributes a set **S** of the target module's public names:

- plain `import m` — S = all public names; nothing injected bare.
- `import m using n1, n2` — S = the listed names (each must be public, else error);
  the listed names are **injected bare**. `using N as M` renames canonically: `M` is the
  name everywhere (bare and qualified); `N` becomes inaccessible through this import.
- `import m hiding n1, n2` — S = all public names except the listed; nothing injected
  bare.
- `open import m [hiding …]` — S as above; **all of S is injected bare**.

Both bare injection and qualified access are bounded by S. Multiple imports of the same
module **union** their contributions (bare-injection sets and S alike). The idiom for
"a few names bare plus the full API qualified" is therefore two lines:

```agl
import utils/strings
import utils/strings using trim
```

`hiding` doubles as the surgical ambiguity repair: a hidden name is absent from the
module's contribution to qualified/suffix resolution, so one `hiding` line resolves an
ambiguity without touching any use site.

Bare-name clashes keep the lazy rule: names reachable bare via multiple imports may
coexist; *using* a clashing bare name is a static error at that use.

### Aliases

`as A` registers the single-NAME alias `A` as a handle for the module **instead of** its
path: an aliased import does not participate in suffix matching. Multiple imports may
share one alias — they merge, with member-level resolution (below). If the same module
is imported both plainly and with `as`, both routes are available (union rule).

### Wildcard imports

`import prefix/*` is sugar: it expands to one import declaration per module whose
slash-path starts with `prefix/`, matched across all roots under the global-uniqueness
rules, carrying the same clauses. `open` and `hiding` distribute per matched module;
`using` distributes too (each listed name must be public in every matched module);
`as A` aliases **every** matched module to `A`, forming an ad-hoc facade with
member-level resolution. The former dotted-handle "alias re-rooting" is removed.

## Qualified references

```ebnf
qual_prefix ::= ["/"] NAME ("/" NAME)* "::"     (* module qualifier *)
              | "::"                            (* self-reference *)
```

### Resolution

For a reference `Q::name`:

1. **Candidate modules.**
   - `Q` = `/p` (anchored): the imported module whose full path is exactly `p`,
     unless that module is alias-only (aliased imports are reachable only via alias).
   - `Q` = a plain (possibly multi-segment) path `S`: every imported module that is
     aliased to `S` (exact match, single-segment only), plus every non-aliased imported
     module whose path has `S` as a trailing segment sequence. A full path is a suffix
     of itself.
2. **Member filter.** Keep candidates whose contributed set S (union across their import
   declarations) contains `name`.
3. **Verdict.** Exactly one candidate → resolve. Zero → error (unknown qualifier, or
   name-not-contributed, with distinct diagnostics). Two or more → **ambiguity error
   listing every candidate module** and suggesting repairs (`hiding`, a longer suffix,
   the `/`-anchor, or `as`).

There is **no preference order** of any kind — not longest-suffix, not alias-first, not
import-order. Any addition (a new import, or an upstream module gaining an export) can
only turn a working reference into a loud error, never silently retarget it.

```agl
import std/config
import std/list/config
import extra/config

config::retries        # ok iff exactly one contributes retries
config::opt            # error if two contribute opt — both named in the error
list/config::opt       # narrowed to paths ending in list/config
/std/config::opt       # anchored: exactly std/config
```

Qualified access to a private name remains a static error. Enum variants still travel
with their enum through `using`/`hiding` and qualification. Writes through a qualifier
(e.g. the `std/config` engine-setting builtin vars) resolve by the same algorithm.

### Interplay with type qualification and self-reference

`::` retains its other roles unchanged: `::name` self-reference, type-qualified
constructors (`Color::red`, `Option[int]::some`), and explicit generic instantiation
(`f::[int]`). In `Q::name`, a single-segment `Q` naming both an in-scope type and a
module handle is a use-site ambiguity error like any other. The
`module-qualifier :: type-qualifier :: name` chain composes as before, with the module
qualifier now a suffix path.

## Lexical rules

In expression, type, and pattern positions, a **tight** run — optional leading `/`, then
`NAME ("/" NAME)*` with no interior whitespace, immediately followed by `::` — lexes as a
module qualifier. Any other `/` keeps its existing meaning (division, param markers).
Single-segment qualifiers (`config::x`) involve no new lexing at all. Consequence:
`a/b::c` is a qualified reference; `a / b::c` (spaced) is division. In import/export
headers, `/` is always a path separator; no tightness requirement applies there.

## Re-exports

`export` carries over with slash-paths and the same one-set clause semantics:

```ebnf
export_decl ::= "export" module_path ["/" "*"] [using_clause | hiding_clause]
```

Declaration-only, transparent origin identity, distinct-origin conflicts are static
errors, diamond re-exports collapse — all unchanged. A facade's consumers qualify
re-exported names through the facade's own path/alias as before, now via suffix
resolution.

## Prelude

The implicit entry-module prelude becomes an ordinary implicit `open import std/core`,
disabled by `--no-stdlib`; explicit re-imports merge normally. Nothing about the prelude
is otherwise special — any module can be opened the same way.

## REPL

Imports may appear at any entry and persist. A new import declaration for an
already-imported module **replaces** that module's prior declaration (the interactive
exception to batch union-merging, kept so options can be changed mid-session). Suffix
resolution, anchors, aliases, and all error rules match batch semantics. `::name`
self-reference over the accumulated session is unchanged.

## Diagnostics catalog

- Import: module not found; module path ambiguous across roots (unchanged); `using` name
  not public; `open … using` redundancy; import after first non-import item (unchanged).
- Qualified use: unknown qualifier (no imported module matches — may hint at importable
  modules with that suffix); name not contributed (module matched, name outside S or
  private); ambiguous reference (all candidates listed, repairs suggested).
- Bare use: ambiguous bare name (unchanged lazy clash rule).

## Migration from the current system

- `import a.b.c` → `import a/b/c`; likewise `export`.
- `qualified` keyword: delete (now the default). Former plain open imports that relied on
  unqualified injection: add `open` or a `using` list.
- Dotted qualifier handles (`foo.bar::x`) → suffix or full-path form (`bar::x`,
  `foo/bar::x` tight, or `/foo/bar::x`).
- Wildcard `import utils.* as A` re-rooting has no equivalent; use `import utils/* as A`
  (single merged facade) or individual aliased imports.
- Small entry programs using only prelude names are unaffected (`std/core` stays open).

## Deferred

- Local open expressions (`open Q in expr`) — precedented (OCaml, Lean), nothing in this
  design blocks adding them later.
- Single-line sugar for the bare+full two-line idiom, if it proves annoying in practice.
- "Did you mean to import …" suffix hints against the module universe (tooling-level).

## Design rationale

The decisions below came out of a precedent survey (Gleam, Elixir, Elm, Haskell, Agda,
Rust, OCaml, Lean, Coq, Unison, C++/C#, PHP, plus the ML-modules literature), with the
key claims verified against primary sources. One criterion organizes all of them: the
documented failure mode in namespace design is not ambiguity but **silent
re-resolution** — an addition elsewhere changing what existing code means with no
diagnostic. Every surveyed disaster is of that kind: PHP's `foo::bar()` changing meaning
when a namespace declaration was added to the file; C#'s closest-enclosing-namespace
lookup silently hiding outer names when upstream adds a type; Coq's name table letting a
new `Import` retarget short names; the "materializing names" hazard that drove Rust
2018's eager import errors. Systems that instead make ambiguity a *loud use-site error*
(Haskell 2010 qualifier merging, Agda opens) have decades of unremarkable production
use. This design therefore permits ambiguity freely at declaration sites, forbids any
resolution preference order, and errors loudly at use sites — extending the clash
philosophy AgL's open imports already had.

### 1. `/` as the module path separator

The trigger was `.` being overloaded between module paths and record field access. The
options were keeping `.`, using `::` for path segments too (Rust-style), or `/`.
PHP's history is the verified case against overloaded separators: reusing its access
operator `::` as the namespace separator made identical source lines resolve differently
depending on a distant namespace declaration, and the RFC that replaced it with `\`
argued a dedicated, otherwise-unused token makes qualification unambiguous and "maps to
filesystem layouts intuitively". GHC's OverloadedRecordDot shows the machinery cost of
keeping `.` overloaded instead: precedence rules plus whitespace-sensitive lexing.
`::`-everywhere was rejected because AgL's `::` already carries qualification,
constructor qualification, and generic instantiation — adding path segmentation would
overload it four ways. `/` matches the file-based reality of the module system (Gleam's
`import gleam/io` + `io.println` is the exact precedent; Clojure, Zig, and Deno also
separate path syntax from member access), and it is syntactically free in import headers
because they are a dedicated item position. Its only conflict — division in expression
position — is confined to the rare multi-segment qualifier and handled by the tight-run
lexical rule.

### 2. Qualified by default; `using`/`open` as explicit opt-ins

The evidence here was the most one-sided of the survey: Gleam and Elixir are
qualified-by-default by semantics, and Elm ("It is best practice to always use
qualified names"), Elixir (`import` "generally discouraged"), and OCaml ("avoid
`open`") all recommend against open imports in their official guidance — Haskell, the
main open-by-default precedent, is the design its own community works around with style
rules. The structural reasons: provenance (every bare name is traceable to a `using`
line, an `open` line, or the prelude) and robustness (a module that gains a new public
name cannot break plain importers). The stricter Gleam-exact alternative — no
whole-module open at all — was rejected because AgL has a real prelude (`std/core`)
that would become inexpressible magic, because vocabulary/DSL-shaped modules would need
hand-maintained many-name lists in every consumer, and because `open` is genuinely
useful in the REPL; making it explicit and rare captures nearly all of the benefit.

### 3. Suffix-based qualification with hard errors and the `/`-anchor

The path to this decision went through three schemes. *Last-segment handles* (Gleam,
Elixir) give `config::max_iters` for free but force `as`-renames whenever leaf names
collide — and leaf names like `config`, `utils`, `types` collide constantly, so renames
would be the norm, producing uninformative invented qualifiers. *Eager collision errors*
(Rust E0252) were rejected as false-positive-heavy (they fire even for disjoint
modules; Rust's own maintainer called the analogous 2018 restriction "not technically
necessary" and it was later relaxed) and as philosophically backwards for AgL, whose
open-import clashes were already lazy. *Qualifier merging with use-site errors*
(Haskell 2010 §5.3/5.5.2: multiple modules may share a qualifier "provided that all
names can still be resolved unambiguously"; Agda: "allow the introduction of ambiguous
names, but give an error if an ambiguous name is used") supplied the safety model.

The adopted scheme (proposed by the owner) generalizes merging from
last-segment-equality to *any trailing segment sequence*: leaf collisions resolve by
adding segments (`list/config::opt`) instead of renames, and last-segment merging is
just the one-segment case. The direct precedent is Coq's partially qualified names —
"any partial suffix of the absolute name" is usable, proving the ergonomics over
decades — but Coq resolves collisions by *silent hiding* (its name table lets a new
`Import` retarget existing short names), which is exactly the hazard class above. Two
amendments remove it: ambiguity is always a hard use-site error listing all candidates
(no preference order of any kind), and candidates are drawn only from the import
header, never from the module universe (universe-wide suffix resolution is only viable
in Unison, whose content-addressed code cannot re-resolve; a file-based language
re-resolves from source every run). Consequently an addition — a new import, or an
upstream module gaining an export — can only turn working code into a loud, one-line-
repairable error (`hiding`, a longer suffix, the anchor, or `as`), never silently
change it.

The leading-`/` anchor closes the residual corner where one imported path is a suffix
of another (`std/config` vs `new/std/config`): without it, even the shorter module's
full path stays ambiguous and a rename is forced. `/std/config::x` restores Coq's
guarantee that "absolute names can never be hidden", in filesystem-flavored syntax
(cf. C++ leading `::`, Rust leading `::`/`crate::`, Lean `_root_.`), so every imported
module is always reachable without renaming. `as` *replaces* the path rather than
adding to it so that aliasing shrinks the candidate set — keeping it usable as a
repair. The tight-lexing cost of multi-segment qualifiers was accepted because it is
confined to the rare disambiguation case (single-segment qualifiers, the common case,
need no new lexing), and because the scheme degrades gracefully: dropping multi-segment
suffixes would yield exactly the last-segment-merging subset.

### 4. Symmetric one-set `using`/`hiding`

The initially recommended design was asymmetric (`using` injects bare names without
bounding the qualifier — the Gleam/Agda reading — while `hiding` removes names from
everything), motivated by Haskell's documented double-import annoyance: symmetric
import lists force two lines per module to get a few names bare plus the full API
qualified. The owner's counter-argument overturned this: under the union-merge rule the
second line is purely *additive* (`import utils/strings` + `import utils/strings using
trim`), so the alleged non-monotone-edit hazard and the pressure toward `open` both
collapse — the only real cost of symmetry is that one extra line. Symmetry then wins on
principle: a single sentence ("an import contributes a set S bounding both channels;
same-module imports union") covers restriction, injection, and repair; the `hiding`
ambiguity repair becomes a *consequence* of the rule instead of a bolted-on special
case; restriction stays expressible (a `using` list is a checkable interface bound and
proactively shrinks the module's contribution to suffix resolution); renames stay
canonical; and existing `using` semantics carry over unchanged. Single-line sugar for
the bare+full idiom remains an option if the two-line form proves annoying (a 2023
Haskell community proposal sought exactly that sugar).

### 5. Prepositive `open import`

Agda's `open import M` is the exact precedent for the compound. The counter-lesson —
GHC's ImportQualifiedPost, added because a prepositive `qualified` misaligned module
names across import blocks — was judged inapplicable: that pain arose because the
keyword appeared on *most* lines in idiomatic Haskell, whereas `open` marks the rare,
deliberately-discouraged form, so the misalignment is rare and doubles as a useful
visual flag on exactly the lines deserving scrutiny. Prefix position also matches the
keyword's whole-declaration semantic scope and reads better before trailing
`hiding` clauses.

### 6. Wildcards as expansion sugar; local opens deferred

The old wildcard's `as`-re-rooting produced dotted multi-segment handles, which are
incompatible with the new qualifier model — but the rest of the feature survives by
respecifying `import prefix/*` as pure sugar for one import per matched module: every
rule (suffix reachability, clause distribution, loud collisions) then follows from
decisions already made, and `as` on a wildcard becomes coherent for free under merging
(all matches aliased to one name = an ad-hoc facade with member-level resolution).
Local open expressions (`open Q in expr`) are well-precedented (OCaml's local opens are
blessed by the same guidelines that discourage file-level `open`; Lean has `open … in`)
and would suit an expression-oriented language, but were deferred to keep the initial
surface small — AgL programs are short enough that file-level `open import` covers most
of the need, and nothing in the adopted design blocks adding them later.
