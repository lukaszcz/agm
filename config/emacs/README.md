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
step when an `emacs` binary is available and skips it with a notice
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
  line that already sits at a valid level.

## Tests

```bash
just test-emacs
```

The suites never run `agm` and need no AGM installation. They are not
part of `just check`, which stays deterministic on machines without
Emacs.
