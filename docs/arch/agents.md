# Agent Workflows

AGM runs real coding agents (claude, codex, and configurable runners) as subprocesses to drive task loops and code-review workflows. The agent layer owns how a runner is invoked, how prompts are prepared, how output is captured under timeouts, and how completion is detected. This is distinct from AgL, which orchestrates agents from inside a typed program (see [agl/index.md](agl/index.md)).

## Agent Runner

The runner parses configured commands, interpolates prompt and context values, chooses prompt delivery, and executes with captured output and an idle timeout. Stateful configurable runners receive a generated id through `%{SESSION_ID}`; Claude, Codex, and Pi use their backend-specific CLI protocols. Prompt preparation and completion detection live beside the runner, while typed AgL `Agent` values decode to immutable argv-building host specs. See [loop.md](../commands/loop.md#prompt-file-path) and [agents.md](../commands/agents.md#runner-command-interpolation) for user-facing interpolation and prompt rules.

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

- `src/agm/agent/spec.py` — host agent specs, each building its own backend argv, plus `AGENT_SPECS`, the variant-to-spec catalog used by AgL decoding. A pure data leaf; the generic decoder lives on the AgL side, in `agl/runtime/agents.py`.
- `src/agm/agent/transport.py` — host-neutral agent-call diagnostics used by runner-backed session adapters.
- `src/agm/agent/session/` — the backend-neutral session protocol and host lifecycle service, CLI continuation adapters, and Pi's persistent RPC adapter. Claude and Pi share generated-session-id lifecycle state while retaining backend-specific argv and fork behavior; Pi cancels extension dialog requests because AGM has no interactive RPC UI; Codex captures its created thread before resuming it. See `protocol.py`, `service.py`, `cli_adapters.py`, and `rpc.py` for the respective boundaries.
- `src/agm/agent/defaults.py` — the built-in runner floor (`DEFAULT_AGENT_RUNNER`) shared by loop, review, and revise.
- `src/agm/agent/runner.py` — runner command parsing, prompt attachment, prepared argv handling, subprocess execution with idle timeout, the run-result structure.
- `src/agm/agent/prompt.py`, `prompt_source.py`, `response.py`, `output.py` — prompt preparation, source resolution, completion detection, and output formatting.
- `src/agm/agent/loop.py` — loop runner/selector/timeout resolution.
- `src/agm/agent/review/` — the review, revise, and refine workflow implementations and prompt preprocessing.
- `src/agm/commands/loop/`, `review.py`, `revise.py`, `refine.py` — the commands that drive these workflows.
