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
extern_func_def ::= "extern" NEWLINE? "def" decl_head type_params? "(" param_list? ")" "->" type_expr
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

An extern's declared name must be a **valid Python identifier and not a
Python keyword** — this is a static error otherwise, since the companion is
looked up by that exact name.

### Methods

An `extern def` whose first parameter is `self` in a record, enum, or exception
scope is a method. Its companion function receives the receiver as its first
positional argument, followed by the method's remaining arguments in declaration
order:

<!-- agl-check: fragment -->
```agl
# counters.agl
record Counter(value: int)

extern def Counter::increment(self, amount: int) -> int
```

```python
# counters.py
def increment(counter, amount):
    return counter["value"] + amount
```

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
— once per extern registry (normally once per program run), **before any AgL
expression evaluates**, so a missing file is reported as a load-time diagnostic
naming the module and expected companion path, while a missing or non-callable
attribute names the module and extern; neither is a mid-run surprise. In an
interactive session the companion imports once until `:reset`; its next use
imports it again.

The companion must provide a **callable attribute with the extern's final
declared member name**; there is no separate mapping clause. Thus `extern def
Tools::slug(...)` resolves `slug` in the companion. Two scoped externs in one
module cannot use the same final member name. Arguments are always
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

`decimal` always crosses as Python's exact `decimal.Decimal` — **never**
`float` — preserving AgL's exact-decimal guarantee end to end.

| AgL type | Python value |
|---|---|
| `int` | `int` |
| `decimal` | `decimal.Decimal` (never `float`) |
| `bool` | `bool` |
| `text` | `str` |
| `unit` | `None` |
| `json` | an independent JSON-shaped value: `dict` / `list` / `str` / `int` / `Decimal` / `bool` / `None` |
| `array[T]` | a live `MutableSequence` view of the caller's array, with mapped `T` elements |
| `dict[text, V]` | a live `MutableMapping` view of the caller's dict, with `str` keys and mapped `V` values |
| a record | an independent `dict` of its mapped fields, keyed by field name |
| an enum | an independent `{"$case": <variant name>, ...mapped fields}` dict |
| an exception | an independent `dict` of its mapped fields, keyed by field name |
| a bare type variable | an opaque **sealed handle** (see below) |
| a function or agent type | not allowed anywhere in an extern's signature — static error |

### Container views

An `array[T]` or `dict[text, V]` argument is a live view of the caller's
container. Mutating the view mutates that caller's array or dict. Repeated
array or dict occurrences of the same AgL container in one call — through two
parameters, or through fields of records, enums, or exceptions — are the same
Python view object when their boundary schemas are compatible. Equal schemas
are compatible. A generic schema and its compatible concrete schema reconcile
to the concrete representation, including through nested containers and fields
of the same nominal shape. Once reconciled, their one shared view uses that
concrete schema: it yields and accepts concrete Python values. Schemas that
disagree on concrete structure, nominal shape, or distinct type-variable seals
are incompatible, and encoding the later occurrence fails. In particular, a
bare generic `T` occurrence crosses as a sealed handle, not a view; `array[T]`
and `array[U]` (and likewise dicts) use
distinct seals and cannot share a view. A record, enum, or exception mapping is
independently built for each occurrence; its nested container fields use shared
views only under these compatibility rules.

Reading a nested `array` or `dict` from a view produces another live view.
Reading a nested record, enum, or exception produces a plain Python `dict`;
mutating that dict has no effect on the AgL nominal value. Reading `json`
produces an independent copy. The same rule applies to fields of a nominal
value: its outer Python dict is independent, while a nested array or dict
field is live.

Every write through a view is checked against its current element or value
schema at the moment of the write, with the same strictness as an extern return
value. At an unreconciled `array[T]` element position, the only accepted value
is a sealed handle for that `T` received during the call in progress, so a
companion can put back only values it received. The corresponding rule applies
to an unreconciled `dict[text, T]` value position. The compatible
generic/concrete alias case above instead uses its concrete schema, so the
shared view accepts concrete values.

Returning a received view returns the same AgL container. Returning a plain
Python `list` or `dict` constructs a new AgL container instead. A view from
another call, or a view whose element or value type does not match the return
position, is rejected.

A view is valid only while its companion call is running. A companion that
retains one and accesses it after the call returns gets an error. Array-view
iteration reads by index, so a mutation at a position not yet reached is
observed. Dict-view iteration uses a snapshot of its keys.

A view is not a `list` or `dict`: `isinstance(xs, list)` is `False`, while
`isinstance(xs, MutableSequence)` is `True`; analogously, a dict view is a
`MutableMapping`, not a `dict`. `list(xs)` and `dict(d)` make independent
plain-Python snapshots. C-level fast paths that require built-in containers,
such as `json.dumps`, need such a snapshot. This is a breaking caveat for
existing companion code that uses `isinstance(x, list)`, `json.dumps(x)`, or
`x.copy()` on a container argument.

Records, enums, exceptions, and `json` cross as independent values: mutating
the Python record/enum/exception dict or JSON value itself has no AgL effect.
A nested array or dict obtained from one remains a live view under the rules
above.

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

An `extern def` may declare type parameters, just like a generic `def`. At
the public FFI API, AgL applies the same **strict parametricity** rules that it
uses within the language itself ([Generics](generics.md#strict-parametricity)):
a companion cannot inspect, depend on, or fabricate a value at a type-variable
position through that API. Every value at a type-variable position — an
argument or a nested element inside an `array[T]` or `dict[text, T]` — crosses
as an **opaque sealed handle** instead of its underlying representation. The
container remains a live view; thus a view of `array[T]` or `dict[text, T]`
yields sealed handles for its elements or values. A fresh seal is minted for
every extern call and every type parameter of that call, so a handle is only
ever valid for the call and the type variable it came from.

This opacity and parametricity are public FFI API properties, not a sandbox or
security guarantee against an arbitrary Python companion. A companion runs
unsandboxed and in-process, and must be trusted; the rules here describe its
supported FFI behavior and the boundary's ordinary validation.

A companion may, with a handle it received:

- pass it along unchanged, including inside a container it rebuilds
  (rearranging, filtering, or duplicating a list of handles is fine),
- compare two handles for equality (`==`) — a handle wrapping an `array` or
  `dict` (or a record, enum, or exception whose fields include one) is equal
  only to a handle wrapping the very **same** container object, never merely
  an equal one; a handle wrapping a scalar, `json`, or a purely-scalar
  nominal value compares by value,
- hash a handle and use it in a Python `set` or as a `dict` key — consistent
  with the equality above,
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

The boundary validates these rules at every call through the public FFI API:
an implementation that tries to peek behind a handle through that API fails the
same way regardless of which concrete types the extern happens to be
instantiated at.

<!-- agl-check: fragment -->
```agl
extern def reverse[T](xs: array[T]) -> array[T]
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

`ExternError` is raised when the companion callable raises an ordinary Python
`Exception` subclass, or when its return value does not conform to the
extern's declared return type (including an invalid, missing, or stale sealed
handle at a type-variable position). A Python `BaseException` subclass raised
by the companion propagates instead. `ExternError` is catchable with
`try`/`catch` like any other exception:

A write through a live view that violates its declared element or value type
raises Python's `BoundaryTypeError`, a `TypeError`, while the call is active.
Accessing a retained view after its call has returned raises Python's
`BoundaryViewRevoked`, a `RuntimeError`. A companion may catch either. If one
escapes the companion, AgL raises `ExternError` and its `python_type` field
names the escaping type (`BoundaryTypeError` or `BoundaryViewRevoked`).

<!-- agl-check: fragment -->
```agl
try
  let slug = to_slug(title)
catch ExternError as e =>
  print "to_slug failed (%{e.python_type}): %{e.message}"
```

A problem discovered before any extern is ever called — a missing companion
file, a missing attribute, or a non-callable attribute — is a **load-time
diagnostic**, not an `ExternError`: it is reported before the program runs at
all, the same way a static type error is, never as a catchable exception.

A cyclic `array` or `dict` argument can cross the boundary as a live view.
`CyclicValueError` ([Exceptions](exceptions.md#cyclicvalueerror)) arises if a
companion `repr()`s a sealed handle or a view over a cyclic container; it is
not folded into `ExternError`.

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
