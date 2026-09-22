# Python FFI: `extern def`

[← Index](index.md)

`extern def` declares a normal, typed, first-class AgL function whose body is
implemented by a sibling Python companion file. The declaration is an
obligation on that companion: AgL converts values by their runtime
representation and does not re-check the declared argument or return type at
each call.

<!-- agl-check: fragment -->
```agl
# mylib.agl
extern def to_slug(title: text) -> text
```

```python
# mylib.py
def to_slug(title):
    return title.lower().replace(" ", "-")
```

## Declarations and companions

An extern declaration has the same parameters, type parameters, defaults, and
first-class behavior as an ordinary `def`, but has no body and requires a
return annotation. Arguments are passed positionally in declaration order, after
any [target contracts](#target-type-parameters), once AgL has applied its own
defaults and named-argument rules.

The companion callable is found by name. By default that is the extern's final
declared member name, used verbatim, which therefore has to be a valid Python
identifier that is not a Python keyword. `@extern-name` supplies the companion
name instead, freeing the AgL name from Python's spelling — the usual case for
a kebab-case or `?`-suffixed name:

<!-- agl-check: fragment -->
```agl
@extern-name("first_option")
extern def first?(xs: array[int]) -> Option[int]
```

```python
# Companion
from agl import option_none, option_some


def first_option(xs):
    return option_some(xs[0]) if len(xs) else option_none()
```

The supplied name is itself subject to the same rule. Every extern of a module
shares that module's one companion, wherever it is declared, so no two of them
may resolve to the same companion name.

An extern is allowed only in a file-backed module. A module with externs needs
a `.py` sibling, imported once before evaluation begins. Missing companions,
missing callables, and import failures are load-time diagnostics. In a REPL
session a companion is imported once until `:reset`.

During the import, AGM temporarily supplies a module named `agl`. A companion
imports the program's nominal classes, value constructors, and exception carrier from it:

```python
from agl import AglException, Box, Shape, array, dict, json
```

`option_some(value)` and `option_none()` build the standard `Option`; they
exist only when `std/option` is loaded.

`agl` is available only for the companion import. The imported classes and
constructors remain valid afterwards. The fixed module name means concurrent
program loads in one Python process are not supported.

### Interpreter-local companion state

A companion that needs mutable state may retain `runtime` and allocate a value
by a stable key. The value belongs to the interpreter making the active extern
call, even when interpreters share an imported companion module:

```python
import random

from agl import runtime


def rng() -> random.Random:
    return runtime.state("mylib/rng", random.Random)
```

`runtime.state(key, factory)` calls `factory` once for each interpreter state
bag and returns that value on later calls. It must be called during an extern
invocation; direct host calls outside evaluation use a separate detached
state, which is not closed automatically.

This is distinct from ordinary Python module globals. A cached companion's
globals are shared by every interpreter using that registry and last until its
companion is re-imported. `runtime.state` instead uses the interpreter active
for the current extern call: its bag lasts for that interpreter's run, not for
the Python module. Thus a batch run gets a fresh bag, and every REPL entry gets
a fresh bag even though its session keeps the companion module cached until
`:reset`.

A value that owns a resource passes a `close` callable receiving that value,
run once when the owning interpreter's run ends — on a clean return as well as
an escaping exception. `close` is recorded only on the creating call:

```python
import requests

from agl import runtime


def session() -> requests.Session:
    return runtime.state("mylib/session", requests.Session, close=requests.Session.close)
```

A `close` that raises never replaces a run already failing: it is attached as
a note on that in-flight error, while on an otherwise clean exit it is itself
raised as the run's error.

### Trace hook

A companion may emit its own structured trace records with
`runtime.trace(kind, payload)`, tagged with the declaring module and an
attributed call site:

```python
from agl import runtime


def fetch(url: str) -> str:
    runtime.trace("http_request", {"url": url})
    ...
```

Like `runtime.state`, this must be called during an extern invocation; a
direct host call outside evaluation is a silent no-op. Nothing is written when
tracing is off. A payload value with no JSON representation, including a
reference cycle, is replaced by a marker rather than failing the call.

The record is `{ts, run_id, kind, origin, site, line, col, ...payload}`: `ts`
and `run_id` identify the record's moment and run like every other trace
record, `kind` is the hook's first argument, `origin` is the extern's own
declaring module and never changes with the caller, `site` is the module
owning the attributed `line`/`col` (a different module than `origin`
whenever the call is attributed across a package boundary), and every
payload field sits at the top level beside them. `site`, `line`, and `col`
are absent when the call has no source location to attribute (a detached or
host-issued call). A payload key of `ts`, `run_id`, `kind`, `origin`, `site`,
`line`, or `col` collides with this envelope and raises `ValueError`.

The attributed call site is the nearest one outside the extern's own mount
(same leading module-path segment: a package name, `std`, or a loose module's
top-level directory): a package's own wrapper functions calling its extern
are an implementation detail, so their internal calls are skipped in favor
of the call the program author actually wrote. When the whole active call
chain stays inside that mount (an entry point that is itself part of it),
the immediate call is reported for lack of an outside one, and `site` is
that call's own module.

## Value mapping

The mapping is injective, so conversion is directed by the actual value rather
than an extern signature.

| AgL value | Python representation |
|---|---|
| `unit` | `None` |
| `bool` | `bool` |
| `int` | `int` |
| `decimal` | `decimal.Decimal` |
| `text` | `str` |
| `json` | `agl.json(value)` / `AglJson` |
| `array[T]` | a mutable sequence view (`MutableSequence`) over the AgL array |
| `dict[text, V]` | a mutable mapping view (`MutableMapping[str, object]`) over the AgL dict |
| record | snapshot instance, or a live view when its declaration has a `var` field |
| enum value | instance of its member record's class, live when that member has a `var` field |
| exception | instance of its synthesized class |
| function | a callback proxy, valid only inside the invocation window |

`bool` is considered before `int` on return because Python makes `bool` an
`int` subclass while AgL does not. A bare Python `list` or `dict` is never an
AgL boundary value and is rejected. Construct a new AgL container with
`agl.array([...])` or `agl.dict({...})` instead. Wrap every JSON value,
including `None` and scalars, with `agl.json(value)`; this keeps JSON `null`
and JSON `3` distinct from `unit` and `int`.

Every AgL `text` value is a sequence of Unicode scalar values: a companion
must return a `str` with no lone surrogate. A companion reading an OS name or
decoding bytes checks it (for example with `agm.util.unicode`) or decodes
strictly; this is not checked at the boundary, so a companion that returns an
invalid `str` produces text that fails wherever the program first encodes it.

A `json` payload crosses without being copied, so the companion carries two
obligations: the payload must be JSON-shaped — dicts keyed by `str`, lists,
`str`, `int`, `decimal.Decimal`, `bool`, `None` — and a payload it passed or
received must not be retained and mutated afterwards.

```python
from agl import array, dict, json

def duplicate(xs):
    return array([*xs, *xs])

def settings():
    return dict({"retries": 3})

def null_json():
    return json(None)
```

## Nominal values

Each program receives one synthesized class per nominal identity. Records and
enum-member records without `var` fields, and exceptions, have immutable fields
and `__match_args__`; records with `var` fields cross as live mutable views. An
enum class is a namespace over its member classes. An inline member record is
available below its enum scope, so `Shape.circle` remains its Python spelling. A
nominal whose final name is unique and does not collide with a built-in `agl` API
can be imported directly. Use the identity-preserving `nominals` namespace,
rooted by module path — or by `entry` for a module with no path identity — and
then by AgL scope, for name collisions and reserved names such as an exception
named `AglException`:

```python
from agl import Box, Shape, nominals

def bump(box):
    return Box(value=box.value + 1)

def area(shape):
    match shape:
        case Shape.circle(radius=r):
            return r * r
        case Shape.square(side=s):
            return s * s

LeftBox = nominals.left.Box
RightBox = nominals.right.Box
```

A class denotes the declaration it was imported for, permanently. Redeclaring
a record, enum, or exception — for example across incremental REPL entries —
never changes the shape of a class a companion already imported: it keeps
constructing and recognizing values of the declaration it was captured from,
whether held in a module global, a closure, or a default argument. A
companion that imports afterward instead sees the redeclaration, under both
its bare name and its `nominals` path.

A record or enum-member record with at least one `var` field crosses as a
live view. Reading an attribute reads the current AgL field. Assigning a `var`
field decodes the Python value and updates the AgL value in place; assigning an
unmarked field raises `AttributeError`. The view is unhashable and compares by
AgL value equality. Records without `var` fields remain immutable snapshots.
Fields are encoded eagerly when a snapshot nominal object is built; a nested
array, dictionary, or mutable record field is therefore already a live view.
Reading a function-typed field yields a callback proxy under the window below,
whichever closure the field currently holds and however the view was obtained.

Constructors always use the original AgL field spelling. Python-compatible
field names work with ordinary keyword arguments and dot access. For another
legal AgL spelling, pass it through `**` and retrieve it with `getattr`, for
example `Prompt(**{"ask-prompt": "continue"})` and
`getattr(prompt, "ask-prompt")`.

Exception classes are plain Python objects, not `Exception` subclasses. They
cross as values. To raise one from a companion, wrap it in `AglException`, as
described in [Raising AgL exceptions](#raising-agl-exceptions).

## Container views

Arrays and dicts are lazy, mutable views over the original AgL container.
Mutating a view mutates the caller's value, including a view stored inside a
nominal field. Mutable-record views have the same write-through behavior for
their `var` fields. Views hold only their container and remain usable after an
extern call returns; retaining one is therefore part of the companion's
contract. The one exception is a function value: reading one out of a view
needs the extern call that owns it, as described in
[Callback invocation window](#callback-invocation-window). Two views over the
same AgL container compare equal and hash alike, but they need not be the same
Python object.

Views are not built-in `list` or `dict`. Use `list(view)` or `dict(view)` for a
detached Python snapshot. A view encodes and decodes elements lazily, so a
companion may write any supported boundary value. An unsupported write raises
`TypeError`.

## Callbacks

An AgL function passed to an extern is a Python callable. Its Python arguments
are decoded as ordinary extern return values, and its result is encoded as an
ordinary extern argument. The companion calls it positionally; its AgL arity
applies, and Python keyword arguments are not accepted.

<!-- agl-check: fragment -->
```agl
extern def apply(f: (int) -> int, value: int) -> int

def increment(value: int) -> int = value + 1

let result = apply(increment, 4)
```

```python
# Companion

def apply(f, value):
    return f(value)
```

A callback may itself call another extern. As with every extern declaration,
the companion is responsible for honoring the declared types, including
parametric type positions. Python callables do not cross in the other
direction: a companion cannot return a bare Python function as an AgL function
value.

## Callback invocation window

A callback is valid only on the interpreter thread while an extern call is
active. A companion may retain a callback and invoke it during a later extern
call on that thread, including a nested extern call. Calling it after the
outermost extern call returns, or from another thread, raises a Python-side
error before AgL execution resumes.

Obtaining a callback follows the same window: an AgL function crosses only
inside an extern call, so reading a function out of a retained view outside
every call raises rather than yielding a callback nothing could invoke. Per-call
context — the callback encoder and `runtime` state alike — belongs to the
calling thread's context; a thread the companion spawns sees it only when it
runs in a copy of that context (`contextvars.copy_context()`).

## Transparent callback exceptions

If a callback raises an AgL exception, that exception passes through companion
frames without becoming `ExternError`. The companion can let it escape, or
catch `AglException` and re-raise the same carrier; AgL then resumes the
original raise with the same exception value, type, and fields.

```python
from agl import AglException

def retrying_callback(f):
    try:
        return f()
    except AglException as error:
        raise error
```

A companion must not replace this carrier with an ordinary Python exception if
it intends the AgL exception to remain catchable by its original type.

## Raising AgL exceptions

A companion may create an AgL exception with its synthesized exception class
and raise it through `AglException`. AgL receives it as an ordinary `raise`,
so its `catch` clauses match it normally.

<!-- agl-check: fragment -->
```agl
exception ParseFailure extends Exception
  input: text

extern def parse(input: text) -> json
```

```python
from agl import AglException, ParseFailure

def parse(input):
    raise AglException(ParseFailure(message="invalid input", input=input))
```

`AglException` requires a synthesized AgL exception value. Every other Python
value, including a record, enum, scalar, container, callable, or arbitrary
object, raises a Python-side `TypeError`. `str()` of an `AglException` is the
wrapped AgL exception's `message` field.

## Generics and trust

Type-variable positions receive their ordinary runtime representation; the
boundary does not enforce parametricity itself. A companion is trusted to
respect the same parametricity rules that its AgL declaration promises;
violations are latent and can produce an incorrect result later. In
particular, `f[T, U](xs, xs)` is allowed: two generic positions never need
schema reconciliation. A type parameter that no parameter mentions is the one
exception: its instantiation crosses too, as a
[target contract](#target-type-parameters).

Likewise, a companion must honor the declared argument and return types, and
the same obligation covers a value written into a live `array`, `dict`, or
mutable-record view. This is trusted, not checked: a representable value of the wrong type
is accepted at the boundary, and the program is then free to fail later, at
an unrelated point, with an error the program cannot catch. An unsupported
Python value (such as a bare `list`) raises `ExternError`. Ordinary Python
exceptions also become `ExternError`, whose `python-type` holds the original
exception class name. A `BaseException` still propagates.

## Target type parameters

A type parameter of an `extern def` that no value parameter mentions, the
receiver `self` included, is a **target type parameter**. Like `ask`, such a
**type-directed** extern returns a type its caller chooses: every call site
resolves each target parameter to a concrete type, and the companion receives
it as an `agl.TypeContract`. The contracts lead the companion's arguments, one
per target parameter in declaration order (the order of an explicit `::[…]`
list), before the receiver and the declared arguments. There is no attribute
to write and no AgL value for a contract.

<!-- agl-check: fragment -->
```agl
enum Team
  | @doc("Invoices and refunds.") Billing
  | Technical

record Triage
  @doc("Which team should handle this?")
  team: Team
  urgent: bool

extern def classify[T](question: text) -> T

program def main() -> unit =
  let team: Team = classify("Which team?")
  let triage = classify::[Triage]("Triage this ticket.")
  if triage.urgent => print(team)
```

```python
# Companion
from agl import TypeContract


def build(target: TypeContract):
    match target.kind:
        case "bool":
            return False
        case "enum":
            return next(iter(target.members.values())).nominal()
        case "record":
            return target.nominal(
                **{field.name: build(field.contract) for field in target.fields.values()}
            )
    raise ValueError(f"unsupported target {target.label}")


def classify(target, question):
    return build(target)
```

A target resolves as an [`ask` target](agent-calls.md#target-types-types-as-contracts)
does: an explicit `::[…]` type argument wins, else the expected type — an
annotation, a `:=` target, a parameter type, or an expected function type.
Unlike `ask`, there is **no default**: a target that neither supplies is a
static error asking for `::[…]`, so a target parameter the result does not
mention is always explicit. The resolved target must also be a valid JSON
output contract, as for a JSON-decoded `ask`: concrete, containing no type
variable of an enclosing generic `def` (a generic wrapper cannot forward its
own type parameter), with a finite JSON Schema, and JSON-serializable — never
`unit`, a function, or an exception.

A target is resolved per **occurrence**. A reference (`let f: (text) -> Team =
classify`, `classify::[Team]`), a method projection, and a partial application
(`classify::[Team](?)`) resolve it where they occur, and every call through
the resulting function value delivers that occurrence's contract. Each
occurrence resolves independently of the others.

A `TypeContract` describes the resolved target:

| Attribute | Meaning |
|---|---|
| `kind` | `"text"`, `"int"`, `"decimal"`, `"bool"`, `"json"`, `"array"`, `"dict"`, `"record"`, `"enum"`, or `"member"` |
| `label` | the type's printed name, such as `array[Team]`; library types are qualified, such as `std/option::Option[Team]` |
| `doc` | a record's, enum's, or member's `@doc`, else `None` |
| `nominal` | a record's or member's synthesized class, an enum's namespace class, else `None` |
| `fields` | record or member fields keyed by JSON name, in declaration order: `(name, doc, contract)`, with the declared name and the field's `@doc` |
| `members` | enum member contracts keyed by JSON tag, in declaration order |
| `items`, `values` | an array's element contract and a dict's value contract, else `None` |
| `schema` | a fresh copy of the target's self-contained [derived JSON Schema](agent-calls.md#derived-json-schema) |

`Option[T]` and every other enum, generic or not, have kind `"enum"`. A
recursive target is a cyclic graph: each recursive type is one shared contract
object wherever it recurs, so a walk over one keeps a visited set. A contract
is immutable, only the host creates one, and contracts compare by identity.
`TypeContract` is importable from `agl` for annotations and `isinstance`
checks.

The return is trusted like every extern's: the companion constructs a value of
the target, and nothing checks it. It builds a record or member by calling
`nominal` with declared field names, as in the example above, and everything
else as the [value mapping](#value-mapping) prescribes.

## Trust boundary

A companion runs unsandboxed and in-process with the privileges of the AGM
process. Load only companions you trust as much as the AgL program. A host can
disable the Python FFI entirely; then any program declaring an extern is
rejected before execution.
