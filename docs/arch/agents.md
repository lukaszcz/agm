# Agent Workflows

AGM runs real coding agents (claude, codex, and configurable runners) as subprocesses to drive task loops and code-review workflows. The agent layer owns how a runner is invoked, how prompts are prepared, how output is captured under timeouts, and how completion is detected. This is distinct from AgL, which orchestrates agents from inside a typed program (see [agl/index.md](agl/index.md)).

## Agent Runner

An agent invocation is a subprocess that receives a prompt and produces output. The runner module parses a configured runner command, validates the executable exists, attaches the prompt (via an interpolated placeholder or by appending a prompt-file reference), and runs it with output capture. Both prompt content and runner command arguments interpolate `%{name}` holes from environment and workflow context under one consistent set of rules; see [loop.md](../commands/loop.md#prompt-file-path) and [agents.md](../commands/agents.md#runner-command-interpolation) for the interpolation and escaping rules. It tracks an *idle timeout* — if the process produces no output for a configured duration the current agent process is terminated and that invocation fails; workflow control remains with the caller. Structured results carry return code, captured streams, elapsed time, and timeout/spawn-error status.

Prompts are resolved from inline text or a file and preprocessed to expand environment variables, writing a temporary prompt file when substitution changes the content. Normal runs clean these files up; dry runs retain them so the printed prompt path can be inspected. Completion is detected by inspecting the agent's final output for a completion marker. An AgL value typed `Agent` is its selected member `RecordValue`, not an enum wrapper; it decodes into an immutable host spec. Pure per-kind builders produce argv for Claude, Codex, Pi, or a verbatim custom command. The prepared-prompt seam accepts those argv directly, retaining the shared prompt-file and process-result behavior. Most specs attach the prompt as a placeholder or an appended `@<path>` argument; promptless native operations receive explicit EOF rather than inheriting AGM's stdin. A spec can instead declare stdin delivery (as `AgentCodex` does, since `codex exec` reads `-` as "prompt on stdin" rather than expanding an `@<path>` argument), in which case the runner pipes the prompt file's contents in and never appends a target.

## Session Service

The session layer presents one backend-neutral lifecycle for continuing agent conversations. `SessionService` owns opaque handles and their backend instances, snapshots agent and transport selection at open/default time, retains closed handles for idempotent close, and releases all owned processes at the command or interpreter boundary. A host-wide reset closes and forgets successfully drained handles and clears the lazy default generation; failed closes remain retryable. Persistent/default handles live for their host run or REPL session; explicit `Agent::ask` calls use ephemeral handles spanning the complete parse-retry loop.

CLI adapters defer native transcript creation until the first prompt. A failure known to occur before process launch leaves that creation state retryable; failures after launch conservatively retain it because the native transcript may already exist. Claude, Codex, and Pi adapters translate the common lifecycle into their CLI-specific creation, continuation, compact, and fork protocols; custom `AgentCommand` sessions interpolate one stable session ID. Pi's RPC backend instead owns one streaming JSONL child process, bounds both pipe writes and response waits with the idle timeout, and isolates the process group so teardown includes tool descendants. Prompt completion correlates Pi's acknowledgement with a state probe, allowing extension commands and input handlers that complete without an agent run while still waiting for ordinary streaming runs to settle. Streaming deltas provide progress, while the final assistant message is authoritative; automatic retries discard output from failed attempts. AgL bridges this service through `agl/runtime/sessions.py`; its dispatcher-backed compatibility host preserves the same handle ownership while dispatching each prompt as a legacy one-shot call.

## Runner Resolution

Loop, review, and revise each resolve their runner from explicit CLI arguments, then their own config section (a `[<section>.<command-name>]` sub-table layered over the base `[loop]`/`[review]`/`[revise]` table), then a shared built-in runner floor. Each command reads only its own section, so review and revise never inherit `[loop]`'s runner. Loop's selector and timeout resolve through the same precedence, without a floor.

## Loop

The `loop` command group drives iterative agent work over a set of tasks. A *selector* chooses the next task and a *runner* works it; `loop run` drives the full cycle, `loop step` performs a single iteration, and `loop select` performs selection only. A timed-out call is retried or leaves its iteration incomplete rather than terminating the loop. When enabled, the loop log receives step output and all loop diagnostics, including setup failures, timeouts, interruption, and completion.

## Review, Revise, Refine

The review workflows compose the runner into a code-review cycle:

- **review** runs a review prompt and writes the review to a (timestamped or specified) file.
- **revise** runs a revision prompt against an existing review file to apply its findings.
- **refine** alternates reviewer and reviser until the work is complete or a step limit is reached.

These share prompt-preprocessing that merges scope, aspects, and other context into the prompt, and they resolve their runner/reviewer/reviser through the same precedence as loops, including per-command config overrides (see [config.md](config.md)).

## Code Entry Points

- `src/agm/agent/spec.py` — host agent specs, each building its own backend argv, plus `AGENT_SPECS`, the member-to-spec catalog used by AgL decoding. A pure data leaf; the generic decoder lives on the AgL side, in `agl/runtime/agents.py`.
- `src/agm/agent/defaults.py` — the built-in runner floor (`DEFAULT_AGENT_RUNNER`) shared by loop, review, and revise.
- `src/agm/agent/runner.py` — runner command parsing, prompt attachment, prepared argv handling, subprocess execution with idle timeout, the run-result structure.
- `src/agm/agent/session/` — session protocol and service ownership, deferred CLI adapters, and the persistent Pi RPC backend.
- `src/agm/agl/runtime/sessions.py` — AgL session-host bridge and dispatcher compatibility host.
- `src/agm/agent/prompt.py`, `prompt_source.py`, `response.py`, `output.py` — prompt preparation, source resolution, completion detection, and output formatting.
- `src/agm/agent/loop.py` — loop runner/selector/timeout resolution.
- `src/agm/agent/review/` — the review, revise, and refine workflow implementations and prompt preprocessing.
- `src/agm/commands/loop/`, `review.py`, `revise.py`, `refine.py` — the commands that drive these workflows.
