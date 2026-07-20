# AgL Namespace and Module System

Status: **implemented**

## Module identity

A module is a `.agl` file identified by its slash path relative to a library
root: `utils/strings.agl` has identity `utils/strings`. An import resolves
against the unordered root set and succeeds only when exactly one matching file
exists. The entry module has no path identity.

## Imports

```ebnf
import_decl ::= ["open"] "import" module_path ["/" "*"]
                ["as" ref_name]
                [using_clause | hiding_clause]
module_path ::= NAME ("/" NAME)*
using_clause ::= "using" import_item ("," import_item)*
hiding_clause ::= "hiding" ref_name ("," ref_name)*
import_item ::= ref_name ["as" ref_name]
```

Each import contributes a selected set **S** of public names. A plain import
selects all public names without making them bare; `using` selects and makes
its listed names bare; `hiding` selects all except its listed names; and
`open import` makes all selected names bare. `open import` with `using` is
invalid. Repeated declarations for one module union their selected sets and
bare contributions. A bare collision is reported where the bare name is used.

A `using N as M` rename is canonical: `M` is the selected member name for both
bare and qualified access through that import. `as A` gives a module the
single-name alias `A` instead of a plain path route. Distinct modules may share
an alias; each referenced member must still resolve to one contributing module.

`import prefix/*` expands to one declaration for every module at or under the
prefix. Its clauses distribute to every matched module. A wildcard alias is a
shared alias facade.

## Reference routes

```ebnf
qual_prefix ::= ["/"] NAME ("/" NAME)* "::" | "::"
```

A plain module route may use any unambiguous trailing sequence of an imported
path. A leading `/` anchors a route to an exact complete path. An alias is a
single-segment route and is not considered by suffix or anchored matching.
The requested member filters route candidates by each module's selected set
**S**. Zero candidates, an inaccessible member, and more than one contributing
candidate are static errors; there is no route-preference order.

`::name` is a current-module self-reference. The same routing rules apply to
value references, writable `builtin var` targets, type references, and
module/type/constructor chains.

## Lexical structure

In expression, type, and pattern positions, every module qualifier is a tight
run: optional leading `/`, then `NAME ("/" NAME)*`, immediately followed by
`::`. This applies uniformly to single-segment and multi-segment qualifiers:
`config::key` and `tools/config::key` are routes, while `config ::key` and
`tools / config::key` are not the same forms. Import and export headers obey
the same adjacency rule: `/` is a path separator written tight against both
segments.

`/` is division only with whitespace on both sides, matching `+`, `-` and `*`,
which need it because they are identifier characters. A `/` touching an operand
on exactly one side (`a/ b`, `a /b`) is a path that went nowhere and is
rejected. The positional-parameter marker `/` touches no operand and is
unaffected.

## Re-exports, prelude, and REPL

`export` uses the same slash path and selection clauses as `import`. Re-exports
preserve their defining-module identity; distinct origins exposed under one
name are static conflicts, while duplicate paths to one origin collapse.

Every loaded entry and library module, except `std/core` itself, receives an
implicit `open import std/core`; `--no-stdlib` disables that automatic opening
throughout the loaded program. Explicit imports of `std/core` follow the
ordinary rules.

REPL imports persist after successful entries. An entry replaces the retained
declaration for each module it names; declarations for one module within that
entry merge. Failed entries do not change retained imports, and `:reset` clears
them. The REPL uses the same route and selected-set rules as batch programs.

## Diagnostics

Static diagnostics cover missing or ambiguous module paths, selection of a
non-public name, redundant `open ... using`, imports after non-import items,
unknown routes, inaccessible members, ambiguous routes, and ambiguous bare
names. Route ambiguities identify their candidates and can be resolved with a
longer suffix, an anchor, an alias, or a narrower contribution.
