# Standard Library

[← Language reference](reference/index.md)

Standard-library modules are imported explicitly. `std/core` is the automatic
prelude unless the host disables it. When that prelude is enabled, the loader
also injects `std/builtin-methods` if the optional registry exists; its
builtin-type methods are then ambient, while their owning modules' free
functions still require an import. With `--no-stdlib`, or a custom standard
library without that registry, import the owning module before calling its
methods.

## Conventions

An operation that can fail normally raises an exception; its `?` counterpart,
when provided, returns `Option` instead. A `!` suffix identifies an in-place
operation, usually the counterpart of a copy-producing operation; it can also
name an inherently mutating operation with no copy-producing counterpart, such
as `shuffle!`. Operations without either suffix may still mutate when their
purpose is inherently mutating, such as `append` or `set`.

A plain `import std/M` keeps a module's free functions qualified
(`math::sum([1, 2, 3])`); `import std/M::*` also makes them bare. Methods are
reached differently: they are ambient when the prelude injects
`std/builtin-methods`, and otherwise require their owning module to be
imported.

## `std/builtin-methods`

`std/builtin-methods` is an optional registry for method-owning modules. With
the standard-library prelude enabled, the loader injects it when present; its
methods from `std/array`, `std/dict`, `std/text`, `std/json`, and `std/math` are
then available on their receiver types. It exports no application API. It is
not injected with `--no-stdlib`, and a custom standard library may omit it; in
either case, importing an owning module loads that module's methods.

## `std/core`

`std/core` is the prelude. It re-exports `std/option`, `std/pair`,
`std/either`, and `std/result`; it also declares these types and operations:

```text
record ExecResult(stdout: text, exit_code: int, stderr: text, timed_out: bool)
enum ParsePolicy = Abort | Retry(n: int)
enum Agent =
  | AgentCommand(command: text)
  | AgentClaude(model: text, thinking: text)
  | AgentCodex(model: text, thinking: text)
  | AgentPi(provider: text, model: text, thinking: text)
record AgentRequest(
  agent: Agent, prompt: text, target_type: Option[text],
  format_instructions: Option[text], json_schema: Option[json], attempt: int,
  previous_error: Option[text], metadata: json,
)

print[T](value: T) -> unit
render[T](value: T) -> text
copy[T](value: T) -> T
shallow_copy[T](value: T) -> T
resource(path: text) -> text
resource-dir() -> text
ask[T](prompt: text, agent: Agent = std/config::default-agent,
       format: text = "", strict_json: bool = false,
       on_parse_error: ParsePolicy = ParsePolicy::Abort) -> T
Agent::ask[T](self, prompt: text, format: text = "", strict_json: bool = false,
              on_parse_error: ParsePolicy = ParsePolicy::Abort) -> T
ask-request(prompt: text, agent: Agent = std/config::default-agent) -> AgentRequest
Agent::ask-request(self, prompt: text) -> AgentRequest
exec(command: text, env: Environ = std/env::environ,
     cwd: Option[text] = Option[text]::None,
     timeout: Option[text] = std/config::timeout) -> ExecResult
```

`copy` makes a deep copy that preserves sharing and cycles; `shallow_copy`
copies only the outer value. `resource` reads a declared module resource and
`resource-dir` returns that resource root. `ask`, `ask-request`, and `exec`
are described in [Agent calls](reference/agent-calls.md) and
[Shell execution](reference/shell-execution.md).

`Exception` is the abstract, nonconstructible base type. Its named-only
`message: text` field is inherited by every concrete exception. `std/core`
owns these concrete shapes:

```text
AgentCallError(message: text, agent: Agent, cause: text, metadata: json)
AgentParseError(message: text, agent: Agent, target_type: text,
                expected_schema: json, raw: text, normalized_raw: text,
                validation_errors: json, attempts: int, metadata: json)
ExecError(message: text, command: text, exit_code: int, stdout: text,
          stderr: text, timed_out: bool)
ExternError(message: text, function: text, python_type: text)
MaxIterationsExceeded(message: text, limit: int, condition: text,
                      last_condition_value: bool, metadata: json)
MatchError(message: text, scrutinee_type: text, scrutinee: json)
IndexError(message: text, index: int, length: int)
KeyError(message: text, key: text)
TypeError(message: text)
ArithmeticError(message: text, operation: text)
UndefinedVariableError(message: text, name: text)
ImmutableBindingError(message: text, name: text, operation: text)
Abort(message: text)
RecursionError(message: text, limit: int)
CastError(message: text, source_type: text, target_type: text, raw: text)
JsonParseError(message: text, raw: text)
RangeError(message: text)
CyclicValueError(message: text)
```

See [Exceptions](reference/exceptions.md) for their use.

The infix functions are `|>[A, B](A, (A) -> B) -> B`,
`<|[A, B]((A) -> B, A) -> B`, `>>[A, B, C]((A) -> B, (B) -> C) -> (A) -> C`,
and `<<[A, B, C]((B) -> C, (A) -> B) -> (A) -> C`. `|>` is left-associative at
priority 5 and `<|` right-associative at priority 4, so `increment <| 2 |>
double` is `increment(double(2))`; `>>` and `<<` compose left-to-right and
right-to-left at priority 60, and a chain cannot mix them without parentheses.
See [Operator precedence](reference/lexical-structure.md#operator-precedence).

## `std/option`

```text
UnwrapError(message: text)
enum Option[T] = None | Some(value: T)
Option::map[U](f: (T) -> U) -> Option[U]
Option::and-then[U](f: (T) -> Option[U]) -> Option[U]
Option::filter(p: (T) -> bool) -> Option[T]
Option::or-else(other: Option[T]) -> Option[T]
Option::with-default(x: T) -> T
Option::unwrap() -> T
Option::is-some() -> bool
Option::is-none() -> bool
Option::each(f: (T) -> unit) -> unit
Option::to-result[E](error: E) -> Result[T, E]
```

`null` is a value of type `json` only; ordinary AgL types are not nullable, so
`Option[T]` is how an absent value is expressed. `map` transforms a present
value, `and-then` chains an operation returning an `Option`, `filter` keeps a
present value only when its predicate succeeds, and `or-else` supplies an
alternative. `with-default` returns the contained value or its argument, while
`unwrap` returns it or raises `UnwrapError`. `each` runs its callback only for
`Some`, and `to-result(error)` maps `Some(value)` to `Ok(value)` and `None` to
`Err(error)`.

## `std/pair`

```text
record Pair[A, B](first: A, second: B)
Pair::map-first[C](f: (A) -> C) -> Pair[C, B]
Pair::map-second[C](f: (B) -> C) -> Pair[A, C]
Pair::swap() -> Pair[B, A]
```

`map-first` and `map-second` transform one component and preserve the other;
`swap` reverses them.

## `std/either`

```text
enum Either[A, B] = Left(value: A) | Right(value: B)
Either::map-left[C](f: (A) -> C) -> Either[C, B]
Either::map-right[C](f: (B) -> C) -> Either[A, C]
Either::is-left() -> bool
Either::is-right() -> bool
Either::left?() -> Option[A]
Either::right?() -> Option[B]
Either::swap() -> Either[B, A]
```

`Either` carries no error convention. `map-left` and `map-right` transform
their branches, `is-left`/`is-right` test one, `left?`/`right?` project one as
an `Option`, and `swap` exchanges them.

## `std/result`

```text
enum Result[T, E] = Ok(value: T) | Err(error: E)
Result::map[U](f: (T) -> U) -> Result[U, E]
Result::map-err[F](f: (E) -> F) -> Result[T, F]
Result::and-then[U](f: (T) -> Result[U, E]) -> Result[U, E]
Result::or-else(f: (E) -> Result[T, E]) -> Result[T, E]
Result::with-default(x: T) -> T
Result::unwrap() -> T
Result::is-ok() -> bool
Result::is-err() -> bool
Result::ok?() -> Option[T]
Result::err?() -> Option[E]
attempt[T](f: () -> T) -> Result[T, Exception]
```

`Result` represents an explicit successful or fallible outcome. `map`,
`map-err`, `and-then`, and `or-else` transform or chain outcomes;
`with-default` returns the success value or a fallback; `unwrap` returns it or
raises `UnwrapError`; `is-ok`/`is-err` test the branch and `ok?`/`err?` project
it as an `Option`. `attempt` calls a nullary function and returns `Ok` on
success or `Err` holding the `Exception` it raised.

## `std/array`

`std/array` owns methods on `array[E]`.

```text
size() -> int                         is-empty() -> bool
first() -> E                          first?() -> Option[E]
last() -> E                           last?() -> Option[E]
append(value: E) -> unit              insert(index: int, value: E) -> unit
pop() -> E                            pop?() -> Option[E]
remove-at(index: int) -> E            clear() -> unit
contains(value: E) -> bool            index-of(value: E) -> int
index-of?(value: E) -> Option[int]    count(predicate: (E) -> bool) -> int
map[U](function: (E) -> U) -> array[U]
map!(function: (E) -> E) -> unit
filter(predicate: (E) -> bool) -> array[E]
filter!(predicate: (E) -> bool) -> unit
each(function: (E) -> unit) -> unit
fold[B](initial: B, function: (B, E) -> B) -> B
fold-right[B](initial: B, function: (B, E) -> B) -> B
any(predicate: (E) -> bool) -> bool   all(predicate: (E) -> bool) -> bool
find?(predicate: (E) -> bool) -> Option[E]
find-index?(predicate: (E) -> bool) -> Option[int]
reverse() -> array[E]                 reverse!() -> unit
sort(comparator: (E, E) -> int) -> array[E]
sort!(comparator: (E, E) -> int) -> unit
slice(start: int, end: int) -> array[E]
take(count: int) -> array[E]          drop(count: int) -> array[E]
concat(other: array[E]) -> array[E]   extend(other: array[E]) -> unit
zip[F](other: array[F]) -> array[Pair[E, F]]
enumerate() -> array[Pair[int, E]]

join(values: array[text], separator: text) -> text
flatten[T](values: array[array[T]]) -> array[T]
unzip[A, B](values: array[Pair[A, B]]) -> Pair[array[A], array[B]]
repeat[T](value: T, count: int) -> array[T]
range(start: int, end: int) -> array[int]
```

`first`, `last`, and `pop` raise `IndexError` when empty, as does `index-of`
for an absent value; `contains` and `index-of` use AgL equality. Array
positions support negative indexes, `remove-at` included; `insert` accepts
boundaries from `-size()` through `size()` and raises `IndexError` outside
them. `slice` is half-open, `take` and `drop` clamp negative counts to zero,
`repeat` is empty for a negative count, and `range` is inclusive in either
direction. `zip` truncates to the shorter input, and `enumerate` pairs each
element with its zero-based index.

`fold` and `fold-right` call `function(accumulator, element)`, reducing
left-to-right and right-to-left respectively. Every callback-taking method
invokes its closure in encounter order; a callback may capture local bindings
and may raise normally.

## `std/dict`

`std/dict` owns methods on `dict[text, V]`; traversal preserves dictionary
order.

```text
size() -> int                         is-empty() -> bool
get(key: text) -> V                   get?(key: text) -> Option[V]
remove(key: text) -> V                remove?(key: text) -> Option[V]
set(key: text, value: V) -> unit      contains(key: text) -> bool
clear() -> unit
keys() -> array[text]                 values() -> array[V]
entries() -> array[Pair[text, V]]
merge(other: dict[text, V]) -> dict[text, V]
merge!(other: dict[text, V]) -> unit
map-values[U](function: (V) -> U) -> dict[text, U]
filter(predicate: (text, V) -> bool) -> dict[text, V]
filter!(predicate: (text, V) -> bool) -> unit
each(function: (text, V) -> unit) -> unit
from-entries[V](values: array[Pair[text, V]]) -> dict[text, V]
```

`get` and `remove` raise `KeyError` for absent keys. `merge` overlays its
receiver with `other`, whose values win; `from-entries` similarly retains the
last value for a duplicate key. `filter` and `each` receive `(key, value)`.

## `std/text`

`std/text` owns methods on `text` and exports `interp(template, vars)` for
runtime name-only interpolation.

```text
size() -> int                         is-empty() -> bool
chars() -> array[text]                lines() -> array[text]
split(separator: text) -> array[text]
trim() -> text                        trim-start() -> text
trim-end() -> text                    upper() -> text
lower() -> text
contains(substring: text) -> bool
starts-with(prefix: text) -> bool     ends-with(suffix: text) -> bool
index-of(substring: text) -> int      index-of?(substring: text) -> Option[int]
replace(old: text, new: text) -> text
slice(start: int, end: int) -> text   repeat(count: int) -> text
pad-start(length: int, fill: text) -> text
pad-end(length: int, fill: text) -> text
interp(template: text, vars: dict[text, text]) -> text
```

Lengths, indexing, slices, padding, and `chars` operate on Unicode code
points, and `lines` recognizes Unicode line boundaries and omits the
terminators. `split` divides on an exact separator and `replace` rewrites every
non-overlapping occurrence. `index-of` raises `IndexError` when absent, while
`index-of?` returns `None`. Slice bounds clamp, non-positive repeats yield
empty text, and padding truncates the repeated fill to reach the requested
code-point length — an empty fill leaves the receiver unchanged.

`interp` performs runtime name-only interpolation from an explicit
`dict[text, text]`; see
[Strings and interpolation](reference/strings-and-interpolation.md).

## `std/json`

`std/json` parses and inspects untyped JSON values.

```text
parse(value: text) -> json            parse?(value: text) -> Option[json]
parse-lenient(value: text) -> json    parse-lenient?(value: text) -> Option[json]
json::kind() -> text                  json::size() -> int
json::keys() -> array[text]           json::has(key: text) -> bool
json::get(key: text) -> json          json::get?(key: text) -> Option[json]
```

```agl
import std/json

program def main() -> unit =
  let strict = json::parse('{"count": 2}')
  let recovered = json::parse-lenient("```json\n[1, 2]\n```")
  print(strict)
  print(recovered)
```

`parse` accepts exactly one JSON value, apart from surrounding whitespace, and
raises `std/core`'s `JsonParseError`; its `?` form returns `None`. The lenient
forms recover a JSON value from fenced or prose-wrapped text, using the same
recovery rules as structured agent and shell output; the strict forms never
recover or repair input. `kind` is `null`, `bool`, `int`, `decimal`, `text`,
`array`, or `object`; `size` is zero for scalars, `keys` is empty for a
non-object, and `has` is `false` for one. `get` raises `KeyError` for a missing
key or a non-object.

Rendering is not a `std/json` operation: use `render(value)` or `value as
text`. Data access into a JSON tree is index-only — `value["key"]` and
`value[index]`.

## `std/toml`

```text
TomlParseError(message: text, raw: text)
TomlRenderError(message: text)
parse(value: text) -> json            parse?(value: text) -> Option[json]
render(value: json) -> text
```

```agl
import std/toml

program def main() -> unit =
  let settings = toml::parse('''
[server]
port = 8080
''')
  print(toml::render(settings))
```

`parse` accepts one TOML document and returns its root table as `json`. Nested
tables and arrays retain their JSON object and array shapes, floats become
`decimal`, and date, time, and datetime values become ISO-8601 text. Malformed
input raises `TomlParseError`; its `?` form returns `None`.

`render` serializes a JSON object as a TOML document, and its output parses
back to the same JSON representation. TOML has a table root and no null value,
so a non-object root or any contained `null` raises `TomlRenderError`, as does
an integer outside the signed 64-bit range. Decimals render as TOML floats,
ordinary signed `nan` and infinity included; a signaling or payload NaN cannot
be preserved and raises `TomlRenderError` too.

## `std/env`

```text
record Environ(vars: dict[text, text])
Environ::get(name: text) -> text
Environ::get?(name: text) -> Option[text]
Environ::set(name: text, value: text) -> unit
Environ::unset(name: text) -> unit
Environ::contains(name: text) -> bool
Environ::extended(overrides: dict[text, text]) -> Environ
builtin var environ: Environ
getenv(name: text) -> text             getenv?(name: text) -> Option[text]
setenv(name: text, value: text) -> unit
unsetenv(name: text) -> unit
```

<!-- agl-check: fragment -->
```agl
import std/env::*

let _ = setenv("MODE", "test")
let mode = getenv("MODE")
let child_env = environ.extended({"DEBUG": "1"})
let _ = unsetenv("MODE")
```

`environ` is seeded from one full snapshot of the host's startup process
environment and does not change afterwards. It is independent of `os.environ`:
`set`, `unset`, and their free-function counterparts change only that AgL
value, never the host process. `extended` returns an overlaid copy. `get` and
`unset` raise `KeyError` when absent, while `get?` returns `None`. With
`--no-stdlib`, no ambient environment binding is installed.

## `std/process`

```text
exit(code: int = 0) -> unit
cwd() -> text                          pid() -> int
hostname() -> text
```

```agl
import std/process

program def main() -> unit =
  let directory = process::cwd()
  let process_id = process::pid()
  let host = process::hostname()
  ()
```

`cwd`, `pid`, and `hostname` report process metadata. `exit` terminates the
program and its host process and does not return to later expressions; zero
indicates success and a nonzero code failure. The code must be in the portable
`0..255` range so every host reports the same number — an out-of-range code is
an ordinary runtime error and terminates nothing.

## `std/config`

```text
builtin var default-agent: Agent = AgentClaude("sonnet", "medium")
builtin var log: bool = false          builtin var log-file: Option[text] = Option[text]::None
builtin var strict-json: bool = false  builtin var max-iters: int = 0
builtin var timeout: Option[text] = Option[text]::None
```

`std/config` provides the mutable engine settings described in
[Host environment](reference/host-environment.md).

## `std/math`

`std/math` owns scalar numeric methods and exports `pi`, `e`, `sum`, and
`sum-decimal`.

```text
int::abs() -> int                      int::min(other: int) -> int
int::max(other: int) -> int            int::clamp(lower: int, upper: int) -> int
int::compare(other: int) -> int        int::pow(exponent: int) -> int
int::sign() -> int                     int::to-decimal() -> decimal

decimal::abs() -> decimal              decimal::min(other: decimal) -> decimal
decimal::max(other: decimal) -> decimal
decimal::clamp(lower: decimal, upper: decimal) -> decimal
decimal::compare(other: decimal) -> int
decimal::sign() -> int                 decimal::floor() -> int
decimal::ceil() -> int                 decimal::round(digits: int = 0) -> decimal
decimal::sqrt() -> decimal             decimal::pow(exponent: int) -> decimal

sum(values: array[int]) -> int
sum-decimal(values: array[decimal]) -> decimal
pi: decimal                            e: decimal
```

`compare` and `sign` return `-1`, `0`, or `1`; `compare` serves directly as an
`array::sort` comparator through `fn(left, right) => left.compare(right)`.
`round`, `sqrt`, and `pow` work under the language decimal context. Integer
powers require a non-negative exponent, `decimal::sqrt` requires a non-negative
receiver, and `decimal::pow` requires a non-negative exponent on a zero base;
each otherwise raises `std/core`'s `RangeError`. Any zero exponent yields `1`.

## `std/time`

```text
TimeParseError(message: text, raw: text)
now() -> decimal                       now-iso() -> text
monotonic() -> decimal                 sleep(seconds: decimal) -> unit
parse-iso(value: text) -> decimal      format-iso(epoch: decimal) -> text
parse(value: text, fmt: text) -> decimal
format(epoch: decimal, fmt: text) -> text
```

```agl
import std/time

program def main() -> unit =
  let epoch = time::parse-iso("2024-01-02T03:04:05+00:00")
  print(time::format(epoch, "%Y-%m-%d"))
```

Time values are UTC Unix epoch seconds. `monotonic` is a non-decreasing clock
for measuring elapsed time, not a calendar timestamp. `parse-iso` accepts an
ISO-8601 timestamp with or without an offset, and `parse`/`format` take
Python-compatible `strptime` and `strftime` directives. A parsed value without
an offset is UTC, as is every formatted epoch value. Invalid input or an
out-of-range temporal conversion raises `TimeParseError`.

## `std/random`

```text
seed(value: int) -> unit               below(upper: int) -> int
between(lower: int, upper: int) -> int uniform() -> decimal
choice[T](values: array[T]) -> T       choice?[T](values: array[T]) -> Option[T]
shuffle![T](values: array[T]) -> unit  uuid() -> text
```

```agl
import std/random

program def main() -> unit =
  random::seed(42)
  print(random::between(1, 6))
```

Each running interpreter owns an independent seedable sequence, which `seed`
resets: `below` is in `0..upper-1`, `between` includes both bounds, and
`uniform` is in `[0, 1)`. `choice` raises `IndexError` for an empty array and
`choice?` returns `None`. `shuffle!` shuffles its receiver in place through the
normal live array view. `uuid` returns a fresh UUIDv4 and is deliberately
independent of the seedable sequence.

## `std/regex`

```text
record Match(
  matched: text, start: int, end: int, groups: array[Option[text]],
  named-groups: dict[text, text],
)
RegexError(message: text, pattern: text)
test(pattern: text, s: text) -> bool
find?(pattern: text, s: text) -> Option[Match]
find-all(pattern: text, s: text) -> array[Match]
replace(pattern: text, s: text, replacement: text) -> text
split(pattern: text, s: text) -> array[text]
escape(s: text) -> text
```

```agl
import std/regex

program def main() -> unit =
  case regex::find?("(?P<word>[A-Za-z]+)-([0-9]+)", "item-42") of
    | Option::Some(value = _ as found) => print(found.named-groups["word"])
    | Option::None => ()
```

`std/regex` uses Python's
[`re`](https://docs.python.org/3/library/re.html) pattern and replacement
syntax. `test` reports whether the pattern occurs anywhere, `find?` returns the
first occurrence, and `find-all` returns non-overlapping occurrences left to
right. A `Match` carries the matched text, zero-based half-open `start` and
`end`, and `groups` in numbered-group order — `None` for a group that did not
participate, `Some(text)` for one that did; `named-groups` holds the
participating named groups.

`replace` rewrites every match and supports backreferences such as `\1` and
`\g<name>`. `split` follows Python `re.split`: boundary empty strings and
captured separators are retained, and an unmatched captured separator becomes
empty text because split results are text. `escape` returns a pattern matching
its argument literally. Every pattern-taking operation raises `RegexError` when
Python cannot compile it; compiled patterns are reused by a bounded internal
cache with no AgL-visible state.

## `std/path`

```text
join(parts: array[text]) -> text        dirname(path: text) -> text
basename(path: text) -> text            extension?(path: text) -> Option[text]
with-extension(path: text, extension: text) -> text
absolute(path: text) -> text            normalize(path: text) -> text
relative(path: text, base: text) -> text
is-absolute(path: text) -> bool         parts(path: text) -> array[text]
home() -> text
```

`std/path` performs lexical host-platform path operations without accessing
the filesystem. `extension?` includes its leading dot, and joining an empty
array yields empty text.

## `std/fs`

```text
FsError(message: text, path: text, operation: text)
read(path: text) -> text                read?(path: text) -> Option[text]
write(path: text, content: text) -> unit
append(path: text, content: text) -> unit
exists(path: text) -> bool              is-file(path: text) -> bool
is-dir(path: text) -> bool              list(path: text) -> array[text]
glob(pattern: text) -> array[text]      mkdir(path: text) -> unit
remove(path: text) -> unit               copy(source: text, destination: text) -> unit
move(source: text, destination: text) -> unit
```

<!-- agl-check: fragment -->
```agl
import std/fs

program def main() -> unit =
  let prompt = fs::read(resource("prompts/review.md"))
  fs::write("draft.md", prompt)
  fs::append("draft.md", "\n")
  let names = fs::list(".")
  print(fs::exists("draft.md"))
```

`std/fs` performs explicit UTF-8 filesystem effects. Every relative path
resolves against the invocation working directory, not the importing module or
a resource anchor, and a relative `list` or `glob` result stays relative while
an absolute input yields absolute paths. Failed non-optional operations raise
`FsError`, carrying the requested `path` and `operation`; `read?` returns
`None` for a missing, unreadable, invalidly encoded, or NUL-containing path,
and the predicates return `false` for a missing path. `mkdir` creates missing
parents, `remove` removes a file, symbolic link, or directory tree without
following a symbolic link, and filesystem-changing operations honor AGM dry-run
mode.
