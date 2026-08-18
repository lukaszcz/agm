# Standard Library

[← Index](index.md)

Standard-library modules are imported explicitly. `std/core` is the automatic
prelude unless the host disables it.

## `std/path`

`std/path` provides lexical, host-platform path operations. It does not access
the filesystem.

- `join(parts) -> text` — joining an empty array returns empty text, the identity value
  for path concatenation.
- `dirname(path) -> text`, `basename(path) -> text`
- `extension?(path) -> Option[text]` — the extension includes its leading dot.
- `with-extension(path, extension) -> text`
- `absolute(path) -> text`, `normalize(path) -> text`
- `relative(path, base) -> text`, `is-absolute(path) -> bool`
- `parts(path) -> array[text]`, `home() -> text`

## `std/fs`

`std/fs` provides UTF-8 text-file and directory operations relative to the
invocation working directory. Failed filesystem operations raise
`FsError(path, operation)`, which extends `Exception`.

- Text files: `read(path)`, `read?(path) -> Option[text]`, `write(path, content)`,
  and `append(path, content)`. Reads require valid UTF-8; `read?` returns `None`
  for a missing, unreadable, or invalidly encoded file.
- Inspection: `exists(path)`, `is-file(path)`, `is-dir(path)`, `list(path)`, and
  `glob(pattern) -> array[text]`.
- Changes: `mkdir(path)` creates missing parent directories; `remove(path)` removes
  a file, symbolic link, or directory tree without following a symbolic link; `copy(source, destination)` copies a file, and
  `move(source, destination)` moves a file or directory.

A NUL character in a non-optional path or glob pattern is invalid and raises `FsError`;
`read?` instead returns `None` for the same input. Filesystem-changing operations honor
AGM dry-run mode.
