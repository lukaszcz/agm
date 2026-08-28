# Standard Library

[← Index](index.md)

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
are described in [Agent calls](agent-calls.md) and
[Shell execution](shell-execution.md).

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

See [Exceptions](exceptions.md) for their use.

The infix functions are `|>[A, B](A, (A) -> B) -> B`,
`<|[A, B]((A) -> B, A) -> B`, `>>[A, B, C]((A) -> B, (B) -> C) -> (A) -> C`,
and `<<[A, B, C]((B) -> C, (A) -> B) -> (A) -> C`.

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

`unwrap` raises `UnwrapError` for `None`.

## `std/pair`

```text
record Pair[A, B](first: A, second: B)
Pair::map-first[C](f: (A) -> C) -> Pair[C, B]
Pair::map-second[C](f: (B) -> C) -> Pair[A, C]
Pair::swap() -> Pair[B, A]
```

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

`Result::unwrap` raises `UnwrapError` for `Err`; `attempt` catches an
`Exception` raised by its callback.

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
for an absent value. Array positions support negative indexes; `slice` is
half-open, `take` and `drop` clamp negative counts to zero, and `range` is
inclusive in either direction. `zip` truncates to the shorter input.

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
last value for a duplicate key.

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
points. `index-of` raises `IndexError` when absent; text slice bounds clamp,
non-positive repeats are empty, and empty padding text leaves the receiver
unchanged.

## `std/json`

`std/json` parses and inspects untyped JSON values.

```text
parse(value: text) -> json            parse?(value: text) -> Option[json]
parse-lenient(value: text) -> json    parse-lenient?(value: text) -> Option[json]
json::kind() -> text                  json::size() -> int
json::keys() -> array[text]           json::has(key: text) -> bool
json::get(key: text) -> json          json::get?(key: text) -> Option[json]
```

`parse` accepts exactly one JSON value and raises `std/core`'s
`JsonParseError`; its `?` form returns `None`. The lenient forms recover a JSON
value from fenced or prose-wrapped text. `kind` is `null`, `bool`, `int`,
`decimal`, `text`,
`array`, or `object`; `size` is zero for scalars. `get` raises `KeyError` for
a missing key or a non-object.

## `std/toml`

```text
TomlParseError(message: text, raw: text)
TomlRenderError(message: text)
parse(value: text) -> json            parse?(value: text) -> Option[json]
render(value: json) -> text
```

`std/toml::parse` raises `TomlParseError` for malformed input; its `?` form
returns `None`. TOML tables and arrays become JSON objects and arrays, floats
become `decimal`, and temporal values become ISO-8601 text. `render` requires
a JSON object root with no null values and raises `TomlRenderError` for values
TOML cannot represent.

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

`environ` is a startup snapshot. `set`, `unset`, and their free-function
counterparts change only that AgL value, never the host process environment;
`extended` returns an overlaid copy. `get` and `unset` raise `KeyError` when
absent.

## `std/process`

```text
exit(code: int = 0) -> unit
cwd() -> text                          pid() -> int
hostname() -> text
```

`exit` terminates the hosting process with a status from `0` through `255`;
`cwd`, `pid`, and `hostname` report process metadata.

## `std/config`

```text
builtin var default-agent: Agent = AgentClaude("sonnet", "medium")
builtin var log: bool = false          builtin var log-file: Option[text] = Option[text]::None
builtin var strict-json: bool = false  builtin var max-iters: int = 0
builtin var timeout: Option[text] = Option[text]::None
```

`std/config` provides the mutable engine settings described in
[Host environment](host-environment.md).

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

`compare` and `sign` return `-1`, `0`, or `1`. Integer powers require a
non-negative exponent, `decimal::sqrt` requires a non-negative receiver, and
`decimal::pow` requires a non-negative exponent on a zero base; each otherwise
raises `std/core`'s `RangeError`. Any zero exponent yields `1`.

## `std/time`

```text
TimeParseError(message: text, raw: text)
now() -> decimal                       now-iso() -> text
monotonic() -> decimal                 sleep(seconds: decimal) -> unit
parse-iso(value: text) -> decimal      format-iso(epoch: decimal) -> text
parse(value: text, fmt: text) -> decimal
format(epoch: decimal, fmt: text) -> text
```

Time values are UTC Unix epoch seconds. `monotonic` is for elapsed-time
measurement. Time parsers raise `TimeParseError`; offset-free parsed values
are UTC.

## `std/random`

```text
seed(value: int) -> unit               below(upper: int) -> int
between(lower: int, upper: int) -> int uniform() -> decimal
choice[T](values: array[T]) -> T       choice?[T](values: array[T]) -> Option[T]
shuffle![T](values: array[T]) -> unit  uuid() -> text
```

Random draws use an interpreter-local seedable sequence: `below` is
in `0..upper-1`, `between` includes both bounds, and `uniform` is in `[0, 1)`.
`choice` raises `IndexError` for an empty array; `uuid` is independent of that
sequence.

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

`std/regex` uses Python-compatible regular expressions and replacement
syntax. Matches are zero-based and half-open. `find-all` is ordered and
non-overlapping. Invalid patterns raise `RegexError`.

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

`std/fs` reads and writes UTF-8 text relative to the invocation working
directory. Failed non-optional operations raise `FsError`; `read?` returns
`None` for a missing, unreadable, invalidly encoded, or NUL-containing path.
`mkdir` creates missing parents, `remove` removes a file, symbolic link, or
directory tree without following a symbolic link, and filesystem-changing
operations honor AGM dry-run mode.
