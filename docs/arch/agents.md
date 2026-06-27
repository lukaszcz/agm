# Agent Workflows

AGM runs real coding agents (claude, codex, and configurable runners) as subprocesses to drive task loops and code-review workflows. The agent layer owns how a runner is invoked, how prompts are prepared, how output is captured under timeouts, and how completion is detected. This is distinct from AgL, which orchestrates agents from inside a typed program (see [agl/index.md](agl/index.md)).

## Agent Runner

An agent invocation is a subprocess that receives a prompt and produces output. The runner module parses a configured runner command, validates the executable exists, attaches the prompt (either by appending a prompt-file reference or substituting a placeholder), and runs it with output capture. It tracks an *idle timeout* — if the process produces no output for a configured duration it is terminated — and returns a structured result carrying return code, captured streams, elapsed time, and timeout/spawn-error status.

Prompts are resolved from inline text or a file and preprocessed to expand environment variables, writing a temporary prompt file when substitution changes the content. Completion is detected by inspecting the agent's final output for a completion marker.

## Runner Resolution

Which runner (and which selector, for loops) is used is resolved by precedence: explicit CLI arguments override per-command config, which overrides the base config section, which falls back to a built-in default runner. The default runner is always the floor, so a runner is always available. The same precedence resolves timeouts.

## Loop

The `loop` command group drives iterative agent work over a set of tasks. A *selector* chooses the next task and a *runner* works it; `loop run` drives the full cycle, `loop step` performs a single iteration, and `loop select` performs selection only. Prompts are preprocessed per step and step output is logged with headers and timestamps.

## Review, Revise, Refine

The review workflows compose the runner into a code-review cycle:

- **review** runs a review prompt and writes the review to a (timestamped or specified) file.
- **revise** runs a revision prompt against an existing review file to apply its findings.
- **refine** alternates reviewer and reviser until the work is complete or a step limit is reached.

These share prompt-preprocessing that merges scope, aspects, and other context into the prompt, and they resolve their runner/reviewer/reviser through the same precedence as loops, including per-command config overrides (see [config.md](config.md)).

## Code Entry Points

- `src/agm/agent/runner.py` — runner command parsing, prompt attachment, subprocess execution with idle timeout, the run-result structure.
- `src/agm/agent/prompt.py`, `prompt_source.py`, `response.py`, `output.py` — prompt preparation, source resolution, completion detection, and output formatting.
- `src/agm/agent/loop.py`, `config.py` — runner/selector/timeout resolution and the default runner.
- `src/agm/agent/review/` — the review, revise, and refine workflow implementations and their prompt preprocessing.
- `src/agm/commands/loop/`, `review.py`, `revise.py`, `refine.py` — the commands that drive these workflows.
