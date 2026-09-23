# Agent Workflows

AGM runs real coding agents (claude, codex, pi, and custom runner commands) as subprocesses to drive task loops and code-review workflows. The agent layer owns how a runner is invoked, how prompts are prepared, how output is captured under timeouts, how conversations continue across calls, and how completion is detected. AgL orchestrates agents from inside typed programs through this same layer ([agl/index.md](agl/index.md)).

## Agent Specs and the Runner

An agent is described by an immutable host *spec* (`agent/spec.py`): one per supported kind, each building its own argv, plus a verbatim custom command. The catalog of specs is what AgL's `Agent` enum decodes into. The runner attaches the prompt the way the spec declares — an interpolated placeholder, an appended `@<path>` argument, or stdin — runs the command with output capture, and enforces an *idle timeout*: a process that produces no output for the configured duration is terminated and that invocation fails, leaving control with the caller. Results carry return code, elapsed time, timeout or spawn-error status, and the captured streams still undecoded, so each is decoded strictly only where its text is needed: output that is not valid UTF-8 is a transport protocol failure, never replacement characters in a reply. Prompt text and runner arguments interpolate `%{name}` holes from the environment and workflow context under one shared rule set (`util/interp.py`).

Each spec's `argv`/`session_argv`/`rpc_argv` takes a `PermissionMode` (`NATIVE`, `UNRESTRICTED`, `NONE`; default `NONE`) and appends its own flag for that mode after its other options — `claude`/`codex` each own a two-entry table, `pi` and a verbatim `AgentCommand` add nothing; each spec owns its table, so no other module does flag string surgery. `codex` keeps a second table for its resume form, which re-asserts no approval policy: a resumed thread inherits the one recorded when its session was created. The runner owns sandbox preparation and teardown for an agent run, given a `SandboxRun` (`sandbox.md`); a preparation failure is an ordinary spawn failure, not a new failure kind. `AgentCallInfo` records whether a call was sandboxed and under which permission mode. An agent run under `--dry-run` never reaches this path: AgL's pipeline stops before execution.

## Host Agent Values

Every CLI token or TOML string whose checked type is the standard `Agent` accepts one shared host syntax (`agent/values.py`), including `--default-agent` and the `default-agent` config key, which decode through the identical dispatch as a program parameter. `claude/MODEL-EFFORT` and `codex/MODEL-EFFORT` select their native CLIs; `pi/PROVIDER/MODEL-EFFORT` selects Pi explicitly, while an otherwise matching `PROVIDER/MODEL-EFFORT` defaults to Pi. The final hyphen separates an opaque effort suffix. Failing shorthand, text is read as canonical tagged JSON, then an AgL value-syntax constructor call; text matching none of these forms is a verbatim `AgentCommand`.

## Sessions

`SessionService` (`agent/session/service.py`) gives every caller one backend-neutral lifecycle for continuing conversations: it owns opaque handles and their backend instances, snapshots agent and transport selection when a session opens, and releases every owned process at the command or interpreter boundary. Two backend families implement the protocol: CLI adapters that translate the lifecycle into each agent CLI's create/continue/compact/fork flags and defer native transcript creation to the first prompt, and a persistent Pi RPC backend that owns one streaming JSONL child process. Which transport an agent uses by default is a property of its spec. AgL reaches the service through `agl/runtime/sessions.py`.

## Runner Resolution

Loop, review, and revise resolve their runner from explicit CLI arguments, then a per-command sub-table layered over their own base section (`[loop]`, `[review]`, `[revise]`), then a shared built-in floor. Each command reads only its own section.

## Loop

The `loop` group drives iterative agent work over a set of tasks: a *selector* chooses the next task and a *runner* works it. `loop run` drives the full cycle, `loop step` one iteration, `loop select` selection only. A timed-out call is retried or leaves its iteration incomplete rather than terminating the loop; when enabled, the loop log receives step output and every diagnostic.

## Review, Revise, Refine

**review** runs a review prompt and writes the result to a file; **revise** applies an existing review file; **refine** alternates the two until the work is complete or a step limit is reached. They share prompt preprocessing that merges scope, aspects, and context into the prompt, and the runner resolution above.

## Code Entry Points

- `src/agm/agent/spec.py` — host agent specs and the spec catalog; `values.py` — host Agent-value syntax; `defaults.py` — the built-in runner floor.
- `src/agm/agent/runner.py` — runner parsing, prompt attachment, subprocess execution with idle timeout, run results.
- `src/agm/agent/session/` — the session protocol, service, CLI adapters, and the Pi RPC backend.
- `src/agm/agent/prompt.py`, `prompt_source.py`, `response.py`, `output.py` — prompt preparation, source resolution, completion detection, output formatting.
- `src/agm/agent/loop.py`, `src/agm/agent/review/` — loop settings and the review/revise workflow implementations.
- `src/agm/commands/loop/`, `review.py`, `revise.py`, `refine.py` — the driving commands.
