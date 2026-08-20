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
imports the program's nominal classes and value constructors from it:

```python
from agl import Box, Shape, array, dict, json
```

`agl` is available only for the companion import. The imported classes and
constructors remain valid afterwards. The fixed module name means concurrent
program loads in one Python process are not supported.

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
| record | instance of its synthesized class |
| enum value | instance of its member record's synthesized class |
| exception | instance of its synthesized class |

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

Each program receives one synthesized class per nominal identity. Records,
enum-member records, and exceptions have immutable fields and `__match_args__`.
An inline member record is available below its enum scope, so `Shape.circle`
remains its Python spelling. A nominal whose final name is unique can be
imported directly. When names collide, use the identity-preserving `nominals`
namespace, rooted by module path (or `entry`) and then by AgL scope:

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

AgL records are immutable, so assigning a synthesized nominal field raises
`AttributeError`. Fields are encoded eagerly when a nominal object is built:
a nested array or dict field is therefore already a live view, while replacing
the outer record is impossible.

Constructors always use the original AgL field spelling. Python-compatible
field names work with ordinary keyword arguments and dot access. For another
legal AgL spelling, pass it through `**` and retrieve it with `getattr`, for
example `Prompt(**{"ask-prompt": "continue"})` and
`getattr(prompt, "ask-prompt")`.

Exception classes are plain Python objects, not `Exception` subclasses. AgL
exceptions cross as values; a companion cannot raise one directly as an AgL
raise.

## Container views

Arrays and dicts are lazy, mutable views over the original AgL container.
Mutating a view mutates the caller's value, including a view stored inside a
nominal field. Views hold only their container and remain usable after an
extern call returns; retaining one is therefore part of the companion's
contract. Two views over the same AgL container compare equal and hash alike,
but they need not be the same Python object.

Views are not built-in `list` or `dict`. Use `list(view)` or `dict(view)` for a
detached Python snapshot. A view encodes and decodes elements lazily, so a
companion may write any supported boundary value. An unsupported write raises
`TypeError`.

## Generics and trust

Type-variable positions receive their ordinary runtime representation; the
boundary does not enforce parametricity itself. A companion is trusted to
respect the same parametricity rules that its AgL declaration promises;
violations are latent and can produce an incorrect result later. In
particular, `f[T, U](xs, xs)` is allowed: two generic positions never need
schema reconciliation.

Likewise, a companion must honor the declared argument and return types, and
the same obligation covers a value written into a live `array` or `dict`
view. This is trusted, not checked: a representable value of the wrong type
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
