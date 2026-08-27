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
return annotation. Its final declared member name must be a non-keyword Python
identifier. Arguments are passed positionally in declaration order after AgL
has applied its own defaults and named-argument rules.

An extern is allowed only in a file-backed module. A module with externs needs
a `.py` sibling, imported once before evaluation begins. Missing companions,
missing callables, and import failures are load-time diagnostics. In a REPL
session a companion is imported once until `:reset`.

During the import, AGM temporarily supplies a module named `agl`. A companion
imports the program's nominal classes, value constructors, and exception carrier from it:

```python
from agl import AglException, Box, Shape, array, dict, json
```

`agl` is available only for the companion import. The imported classes and
constructors remain valid afterwards. The fixed module name means concurrent
program loads in one Python process are not supported.

### Interpreter-local companion state

A companion that needs mutable state may retain `runtime` and allocate a value
by a stable key. The value belongs to the interpreter making the active extern
call, even when interpreters share an imported companion module:

```python
import random
from typing import cast

from agl import runtime


def rng() -> random.Random:
    return cast(random.Random, runtime.state("mylib/rng", random.Random))
```

`runtime.state(key, factory)` calls `factory` once for each interpreter state
bag and returns that value on later calls. It must be called during an extern
invocation; direct host calls outside evaluation use a separate detached state.

This is distinct from ordinary Python module globals. A cached companion's
globals are shared by every interpreter using that registry and last until its
companion is re-imported. `runtime.state` instead uses the interpreter active
for the current extern call: its bag lasts for that interpreter's run, not for
the Python module. Thus a batch run gets a fresh bag, and every REPL entry gets
a fresh bag even though its session keeps the companion module cached until
`:reset`.

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
rooted by module path (or `entry`) and then by AgL scope, for name collisions and
reserved names such as an exception named `AglException`:

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
contract. Two views over the same AgL container compare equal and hash alike,
but they need not be the same Python object.

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
object, raises a Python-side `TypeError`.

## Generics and trust

Type-variable positions receive their ordinary runtime representation; the
boundary does not enforce parametricity itself. A companion is trusted to
respect the same parametricity rules that its AgL declaration promises;
violations are latent and can produce an incorrect result later. In
particular, `f[T, U](xs, xs)` is allowed: two generic positions never need
schema reconciliation.

Likewise, a companion must honor the declared argument and return types, and
the same obligation covers a value written into a live `array`, `dict`, or
mutable-record view. This is trusted, not checked: a representable value of the wrong type
is accepted at the boundary, and the program is then free to fail later, at
an unrelated point, with an error the program cannot catch. An unsupported
Python value (such as a bare `list`) raises `ExternError`. Ordinary Python
exceptions also become `ExternError`, whose `python_type` holds the original
exception class name. A `BaseException` still propagates.

## Trust boundary

A companion runs unsandboxed and in-process with the privileges of the AGM
process. Load only companions you trust as much as the AgL program. A host can
disable the Python FFI entirely; then any program declaring an extern is
rejected before execution.
