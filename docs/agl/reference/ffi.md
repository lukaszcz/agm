# Python FFI: `extern def`

[← Index](index.md)

`extern def` declares a function whose implementation lives outside AgL, in a
companion Python file. An extern is a normal, fully typed, first-class AgL
function: it can be called, stored, passed, and returned exactly like a `def`
or `fn` value ([Functions](functions.md)). Only its body is different — instead
of an AgL expression, invoking it crosses into Python.

<!-- agl-check: fragment -->
```agl
# mylib.agl
extern def to_slug(title: text) -> text
```

```python
# mylib.py — the companion file
def to_slug(title):
    return title.lower().replace(" ", "-")
```

## Declaration syntax

```ebnf
extern_func_def ::= "extern" NEWLINE? "def" name type_params? "(" param_list? ")" "->" type_expr
```

An `extern def` has the same signature surface as an ordinary `def` —
type parameters, parameter zones (`/`, `*`, `@pos`/`@std`/`@named`), and AgL
default expressions all work identically ([Functions](functions.md),
[Generics](generics.md)) — with two differences: it has no body, and the
`-> type_expr` return-type annotation is **mandatory** (as for `builtin def`).
Defaults are ordinary AgL expressions evaluated on the AgL side, before the
call crosses the boundary — a companion never sees an unfilled default. The
`extern` marker may be on the same line as `def` or on the line directly above
it; the optional newline is insignificant.

`private` composes with `extern def` exactly as it does with `def`: a private
extern is callable only from within its declaring module
([Modules](modules.md#public-and-private-names)).

An extern's declared name must be a **valid Python identifier and not a
Python keyword** — this is a static error otherwise, since the companion is
looked up by that exact name.

## Placement

`extern def` is only allowed in a **file-backed module** — a library module,
or an entry program loaded from a file. Declaring `extern def` in program text
with no backing file (for example, inline program text, or a direct entry at
an interactive prompt) is a static error: there is no file path to derive a
companion from. Importing a file-backed module that declares externs works
normally from any context, including an interactive session.

## The companion file

A module that declares at least one `extern def` requires a companion Python
file at the same path with a `.py` extension in place of `.agl` (`utils/nlp.agl`
requires `utils/nlp.py`). The companion is imported — its top-level code runs
— once per program run, **before any AgL expression evaluates**, so a missing
file, a missing attribute, or a non-callable attribute is reported as a
load-time diagnostic naming the module and the extern, never a mid-run
surprise. In an interactive session the companion imports once per session,
regardless of how many entries import or call it.

The companion must define a **plain function with exactly the extern's
declared name**; there is no separate mapping clause. Arguments are always
passed **positionally, in declaration order** — named arguments, zones, and
defaults are all AgL-side call mechanics that resolve to a plain positional
argument list before the call crosses the boundary, so the companion's own
parameter names are unconstrained:

<!-- agl-check: fragment -->
```agl
extern def greet(name: text, /, greeting: text = "Hello") -> text

greet("Ada")                    # -> greet("Ada", "Hello")
greet("Ada", greeting = "Hi")   # -> greet("Ada", "Hi")
```

```python
def greet(n, g):     # parameter names are the companion's own business
    return f"{g}, {n}!"
```

## Type mapping

Every value crossing the boundary is deep-copied, so neither side can observe
the other's later mutations. `decimal` always crosses as Python's exact
`decimal.Decimal` — **never** `float` — preserving AgL's exact-decimal
guarantee end to end.

| AgL type | Python value |
|---|---|
| `int` | `int` |
| `decimal` | `decimal.Decimal` (never `float`) |
| `bool` | `bool` |
| `text` | `str` |
| `unit` | `None` |
| `json` | a JSON-shaped value: `dict` / `list` / `str` / `int` / `Decimal` / `bool` / `None` |
| `list[T]` | a `list` of mapped `T` elements |
| `dict[text, V]` | a `dict` of `str` keys to mapped `V` values |
| a record | a `dict` of its mapped fields, keyed by field name |
| an enum | `{"$case": <variant name>, ...mapped fields}` |
| an exception | a `dict` of its mapped fields, keyed by field name |
| a bare type variable | an opaque **sealed handle** (see below) |
| a function or agent type | not allowed anywhere in an extern's signature — static error |

`Option[T]` gets no special treatment: it is an ordinary two-variant generic
enum, so `None`/`Some(value = ...)` cross as `{"$case": "None"}` and
`{"$case": "Some", "value": ...}` respectively, just like any other enum.

A recursive record or enum crosses just like any other, nesting its mapped
shape to whatever depth the value reaches. The only requirement is that the
type has a finite schema: a type whose recursive instantiations never close
(growing polymorphic recursion) cannot appear as an extern parameter or return
type — the same restriction that applies to agent-output and cast targets.

### Return values are validated strictly

A companion's return value is checked against the extern's declared return
type with the same strict rules used everywhere else values enter AgL from
outside ([Types](types.md#casts-and-convertibility)), with one added
tolerance: a plain Python `int` is accepted where `decimal` is declared
(widened exactly, mirroring AgL's own `int` → `decimal` assignability).
Everywhere else the match is exact:

- `bool` is **rejected** where `int` or `decimal` is declared (Python's `bool`
  is a subtype of `int`, but AgL's is not).
- `float` is **never** accepted anywhere, including nested inside a `json`
  value.
- A record, enum, or exception must match its declared shape exactly —
  missing fields, extra fields, misnamed fields, and unknown enum variants are
  all rejected.
- A `unit`-returning extern's companion must return exactly `None`.
- A bare type-variable return position must carry a sealed handle for that
  variable, minted during the very call in progress (see below).

Any mismatch raises `ExternError` ([Errors](#errors)).

## Generics and sealed handles

An `extern def` may declare type parameters, just like a generic `def`. AgL
enforces the same **strict parametricity** guarantee across the Python
boundary that it enforces within the language itself
([Generics](generics.md#strict-parametricity)): a companion cannot inspect,
depend on, or fabricate a value at a type-variable position. Every value at a
type-variable position — an argument or a nested element inside a `list[T]`
or `dict[text, T]` — crosses as an **opaque sealed handle** instead of its
underlying representation. A fresh seal is minted for every extern call and
every type parameter of that call, so a handle is only ever valid for the
call and the type variable it came from.

A companion may, with a handle it received:

- pass it along unchanged, including inside a container it rebuilds
  (rearranging, filtering, or duplicating a list of handles is fine),
- compare two handles for equality (`==`) — equal exactly when the AgL values
  they wrap are equal,
- hash a handle and use it in a Python `set` or as a `dict` key,
- print or `repr()` it for debugging.

A companion may **not**:

- inspect what a handle wraps, or otherwise recover the AgL value's
  representation from it,
- construct an unrelated Python object and return it where a type-variable
  result is expected — this is rejected the same way a wrong-shaped concrete
  return value is,
- return a handle received from a **different call** — a handle stashed
  across calls (in a module-level variable, for example) is stale and
  rejected,
- return a handle received at a **different type variable** of the same
  call — a handle for `T` returned where `U` is expected is rejected.

Every one of these is enforced at every call: an implementation that tries to
peek behind a handle fails the same way regardless of which concrete types the
extern happens to be instantiated at.

<!-- agl-check: fragment -->
```agl
extern def reverse[T](xs: list[T]) -> list[T]
```

```python
def reverse(xs):
    return list(reversed(xs))   # rearranges handles; never inspects them
```

## Errors

`ExternError` extends the base `Exception` type ([Exceptions](exceptions.md))
with two extra fields:

```text
function: text       # the extern's declared name
python_type: text    # the raising Python exception's class name; empty
                      # when the failure was a return-value mismatch
```

`ExternError` is raised when the companion callable itself raises, or when its
return value does not conform to the extern's declared return type (including
an invalid, missing, or stale sealed handle at a type-variable position). It
is catchable with `try`/`catch` like any other exception:

<!-- agl-check: fragment -->
```agl
try
  let slug = to_slug(title)
catch ExternError as e =>
  print "to_slug failed (${e.python_type}): ${e.message}"
```

A problem discovered before any extern is ever called — a missing companion
file, a missing attribute, or a non-callable attribute — is a **load-time
diagnostic**, not an `ExternError`: it is reported before the program runs at
all, the same way a static type error is, never as a catchable exception.

## Trust

A companion's top-level code, and every extern call into it, runs
**unsandboxed and in-process**, with the full privileges of whatever is
running the program — the same trust boundary as `exec`
([Shell execution](shell-execution.md)), but for Python code instead of a
shell command. Only load a companion file whose source you trust as much as
the AgL program that imports it.

A host may disable the Python FFI entirely; a program that declares any
`extern def` is then rejected before it runs, with a clear diagnostic —
mirroring how a host may statically disallow `exec`.
