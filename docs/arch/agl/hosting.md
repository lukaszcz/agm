# AgL Pipeline and Hosting

`PipelineDriver` (`agl/pipeline.py`) is the public entry point to AgL. It drives parse → scope → typecheck → match compile → lower/link → evaluate, assembles the host environment, and hands every artifact forward so a program compiles and lowers exactly once however often a host resumes it. `agm exec` and registered package commands (through `commands/exec_program.py`), `agm check`, package discipline validation, and the REPL's entry pipeline all sit on it. There is never a second pipeline: anything a host wants compiled is compiled with the program.

## Host Capabilities

`HostCapabilities` (`agl/capabilities.py`) is a frozen descriptor of the host features that affect checking — shell `exec`, externs, codec kinds. The checker sees descriptors, never runtime implementations, and cached artifacts are keyed by the capabilities they were checked under.

## Programs, Entries, and Parameters

Preparation records every `program def` with its module and scope path, and the `param` inventory of the selected program module's transitive imports, so a host can select an entry and wire parameters before execution. `agm exec` exposes params as CLI flags ([cli.md](../cli.md)) and resolves each as external value > source default > required error. The selected entry runs after module initialization inside the interpreter's normal boundary. `agm check` uses `check_prepared`, which stops after lowering and binds no parameters, so a required param without a default is accepted.

`agl/runtime/arguments.py` holds the host-side counterpart for a `program def`'s own value parameters (as opposed to module `param` declarations): `ProgramArguments` carries a host's raw positional/named values, and `bind_program_arguments` runs them through the shared zone binder (`agl/semantics/arguments.py`) and decodes each supplied value through its `ParamDecoder`. A structural violation (unknown name, duplicate, positional overflow, a positional-only parameter supplied by name) is reported as a single pre-execution diagnostic; a missing required parameter or a decode failure is reported for every offending parameter, not just the first. `ProgramSignature.fuse` pairs an executable's per-parameter decoders with a declaration's spans and types by name, so the two independent descriptions can never be mismatched by position; `bind_program_arguments_for` is the one place that fuses a `ProgramDeclInfo` with its `ExecutableProgram` and binds, so `PipelineDriver.preflight_arguments` and `commands.exec_program.run` — both of which hold the same two descriptions for a selected program — can never pair them differently. `agm exec` builds a selected program's `ProgramArguments` from its own CLI option map (`cli_support/program_options.py`) and qualified config table, resolved beneath any CLI value with the same precedence as `param`; a colliding or duplicate flag projection is reported before any value is parsed.

`PipelineDriver.discover_programs` mirrors `discover_params`, sharing the same static-pipeline steps, but reports each `program def`'s own typed parameter signature instead of the `param` inventory. `PipelineDriver.preflight_arguments` mirrors `preflight_params`: it binds a host's `ProgramArguments` against one selected program's signature and hands the bound arguments together with the lowered executable to `run_prepared`, so a program is lowered once and its own value parameters are bound once, however many times a host resumes the pipeline.

## Engine Settings and Host Seeds

Engine settings (`default-agent`, `log`, `log-file`, `strict-json`, `max-iters`, `timeout`) are root `builtin var` bindings of `std/config`, catalogued in `config/engine_keys.py` ([config.md](../config.md)). Hosts seed them, and any other host-backed binding, through typed `builtin_var_seeds` keyed by module, scope path, and name; a source write overrides a seed from its program point onward, and the evaluator routes each write by the catalog's consuming side — a live interpreter field or a host-consumed register.

A host-supplied AgL *literal* — `--agent` or `[exec] default-agent` — is not parsed separately: `SettingOverride` (`agl/setting_overrides.py`) pairs the source text with its origin, and the pipeline splices it in as the binding's default before scope resolution, so the program's own compilation checks it and a rejection names the origin. A per-run request such as `--agent` is `required` and is a diagnostic when no `std/config` is loaded; a config-file default is silently inert. The winning `default-agent` is validated before any statement runs, so a malformed custom command is a pre-execution diagnostic rather than a runtime error.

## Diagnostics and Recursion

All passes report through `agl/diagnostics.py`; scope and typecheck diagnostics carry a phase tag. `agl/recursion.py` is the boundary that turns stack exhaustion on over-deep source into an ordinary pre-execution diagnostic; the pipeline and the REPL apply it around every pass they drive.

## Self-Validation

The compiler carries self-checks that re-verify artifacts it just produced (checker closure, match artifacts, deep IR validation, FFI class shapes). One toggle in `agl/self_validation.py` gates them; the test suite turns it on and production leaves it off ([testing.md](../testing.md)).

## Code Entry Points

- `src/agm/agl/pipeline.py` — `PipelineDriver`: preparation, checking, execution, host-environment assembly.
- `src/agm/agl/capabilities.py`, `diagnostics.py`, `recursion.py`, `setting_overrides.py`, `self_validation.py` — the host-facing leaves.
- `src/agm/agl/runtime/arguments.py` — `program def` argument binding and decoding; `src/agm/agl/runtime/params.py` — module `param` and engine-setting decoding/config helpers.
- `src/agm/commands/exec_program.py`, `check.py`, `repl.py` — the CLI hosts; `src/agm/cli_support/` — parameter discovery and engine-setting seeds.
- Tests: `tests/test_agl_pipeline_*.py`, `test_exec_*.py`, `test_check_command.py`, `test_default_agent_setting.py`, `test_agl_self_validation.py`, `test_agl_runtime_arguments.py`.
