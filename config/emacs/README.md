# AgL Emacs mode

An Emacs major mode for AgL: context-sensitive syntax, structural
highlighting, layout-aware indentation, declaration navigation, on-save
diagnostics from the real AGM pipeline, and an inferior REPL.

## Files

| File | Contents |
| ---- | -------- |
| `agl-mode.el` | Entry point: syntax table, `syntax-propertize`, font-lock, imenu, defun motion, keymap |
| `agl-indent.el` | Indentation engine |
| `agl-flymake.el` | Flymake backend over `agm check` |
| `agl-run.el` | `agl-run` / `agl-check` through `compile` |
| `agl-repl.el` | Inferior AgL REPL over `agm repl --plain` |
| `tests/` | ERT suites; not part of the installed package |

## Install

```bash
just setup-emacs
```

This stages the Elisp as a package and installs it with Emacs's own
package machinery, which generates the autoloads and byte-compiles the
sources. The package version tracks AGM's. `just install` runs the same
step last, when an `emacs` binary is available, and skips it with a notice
otherwise. Nothing is ever written to your init file.

Pass a prefix to install under a different `HOME`:

```bash
just setup-emacs /tmp/somewhere
```

For Doom, straight.el, or any setup that does not use package.el, load
the directory directly:

```elisp
(add-to-list 'load-path "/path/to/agm/config/emacs")
(require 'agl-mode)
```

## Keys

| Key | Command |
| --- | ------- |
| `C-c C-c` | `agl-run` — run the file with `agm exec` |
| `C-c C-k` | `agl-check` — statically check it with `agm check` |
| `C-c C-z` | `agl-repl` — start or switch to the inferior REPL |
| `C-c C-r` | `agl-send-region` |
| `C-c C-b` | `agl-send-buffer` |
| `C-c C-l` | `flymake-show-buffer-diagnostics` — list this file's diagnostics |

A diagnostic's tooltip lasts only while the mouse hovers it, so the listing
buffer is where a message can be read at length or copied; `C-c C-k` puts the
same diagnostics in a compilation buffer. Setting `help-at-pt-display-when-idle`
echoes the message under point in the minibuffer instead of a tooltip.

A sent region goes to the REPL unsplit. When its last line is indented the
region leaves a block open — a layout block always accepts one more line — so
the blank line that closes it is sent too.

## Customization

- `agl-indent-offset` — block indentation, default 2.
- `agl-exec-command`, `agl-check-command`, `agl-repl-command` — the
  command vectors the integrations run; put any flags here.
- `agl-flymake-enable` — whether entering the mode turns on
  `flymake-mode`.
- `agl-repl-buffer-name` — the inferior REPL buffer's name.
- `agl-interpolation-face` — the face for `%{`/`}` delimiters.

## Known limits

- Diagnostics describe the **saved** file. The checker resolves module
  roots and imports from the entry file's real path, so checking a
  temporary copy would resolve a different program; a modified buffer
  therefore reports the last state written to disk.
- AgL expressions inside `%{…}` interpolation holes are not fontified —
  only the delimiters are.
- Constructor use-sites in expressions are unfaced. Capitalization
  carries no meaning in AgL, so faces come from declaration and
  annotation positions only, and a bare name in an expression cannot be
  classified lexically.
- Indentation offers the levels a line could legally take rather than
  guessing one: `TAB` cycles through them, and `indent-region` leaves a
  line that already sits at a valid level. The levels are the columns
  enclosing lines actually sit at, not multiples of
  `agl-indent-offset` — a body or a `|` branch takes its column from the
  line that starts it, so a wider body and guard markers aligned under
  an inline `if | …` both survive a region re-indent. For the same
  reason `RET` indents the line it opens but never re-indents the line
  it ends.
- Where the grammar settles a line's level, it is used instead of the
  column carried over from the line above: a `def`, `record`, `import`,
  or other module item returns to the root or scope region that encloses
  it, `until`, `done`, `catch`, and `else` align with the header they
  name, and a bracket's closer returns to the line that opened it. A
  line is re-indented as soon as typing settles which construct it is:
  on the `@` or closing bracket that can start nothing else, on the
  space that tells a declaration keyword from a name beginning with the
  same letters, on the last letter of a branch marker, and — for a line
  holding nothing but `builtin`, `extern`, or `program`, which no
  separator ever follows — on the newline that ends it.

## Tests

```bash
just test-emacs
```

The suites never run `agm` and need no AGM installation. They are not
part of `just check`, which stays deterministic on machines without
Emacs.
