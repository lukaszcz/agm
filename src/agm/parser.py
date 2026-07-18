"""Help text and usage utilities for AGM's Typer CLI."""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Sequence
from typing import NoReturn, Protocol

from agm.command_catalog import COMMAND_OVERVIEW


class _Writeable(Protocol):
    def write(self, data: str) -> object: ...


_HELP_TEXTS: dict[str, str] = {
    "open": textwrap.dedent("""\
        agm open [-d|--detach] [-n|--num-panes PANES] [-p|--parent PARENT] TARGET

        Open a tmux session for an AGM workspace, creating or checking out a branch as needed.

        Options:
          -d, --detach            Create the tmux session without attaching to it.
          -n, --num-panes PANES   Create the session with PANES panes.
          -p, --parent PARENT     Base a newly created branch workspace on PARENT instead of
                                  the main workspace's current branch.

        Behavior:
          repo           Open the main workspace.
          default branch Open the main workspace when TARGET matches the
                         branch currently checked out there.
          existing branch workspace
                         Open the tmux session for an existing branch workspace.
                         With --parent, this is an error.
          existing branch Check out BRANCH into a Git worktree, then open it as a workspace.
                         With --parent, warn and ignore --parent.
          missing branch  Create BRANCH from PARENT/current branch, then open it.

        Examples:
          agm open repo
          agm open -d repo
          agm open main
          agm open feat/login
          agm open --num-panes 4 feat/login
          agm open --parent main feat/search
    """),
    "close": textwrap.dedent("""\
        agm close [-f|--force] [-D] [--keep-branch] [--keep-workspace] BRANCH

        Close a branch workspace.

        Remove the workspace's Git worktree unless --keep-workspace is used, then kill
        the corresponding tmux session.

        Options:
          -f, --force   Force remove the branch workspace's Git worktree
                        (even with untracked or uncommitted changes) and force delete the branch
                        (git branch -D). Implies -D.
          -D            Force delete the branch (git branch -D) instead of
                        safe delete (git branch -d). The Git worktree is only
                        removed if the branch deletion would succeed.
          --keep-branch
                        Remove the Git worktree but keep the local branch.
          --keep-workspace
                        Keep the Git worktree and local branch; only close the
                        workspace session. Implies --keep-branch.
    """),
    "init": textwrap.dedent("""\
        agm init [--embedded | --split]
                 [--no-git-init | --no-repo-git | --no-config-git | --no-notes-git]
        agm init [--embedded | --split]
                 [--no-git-init | --no-repo-git | --no-config-git | --no-notes-git]
                 PROJECT_NAME
        agm init [--embedded | --split] [-b|--branch BRANCH]
                 [--no-git-init | --no-repo-git | --no-config-git | --no-notes-git]
                 [PROJECT_NAME] REPO_URL
        agm init --clone [--embedded | --split] [-b|--branch BRANCH]
                 [--no-git-init | --no-repo-git | --no-config-git | --no-notes-git]
                 REPO_URL

        Initialize a project. Without PROJECT_NAME, agm initializes the current
        directory. With PROJECT_NAME, agm initializes a child directory with that
        name. When REPO_URL is provided, agm also clones it into repo/ by default,
        or into the project root with --embedded. Use --clone with a URL-only init
        to initialize a child directory derived from the repo URL. Without an
        explicit layout flag, agm chooses the embedded layout when the target
        project directory is already a git repo; otherwise it chooses the
        split layout.

        Options:
          --embedded   Force the embedded layout with AGM data under .agm/.
          --split      Force the split layout with repo/, deps/, notes/,
                       worktrees/, and config/ under the project root.
          --clone      Initialize a new project directory derived from REPO_URL.
          -b, --branch BRANCH
                       Clone this branch when REPO_URL is provided.
          --no-git-init
                       Do not create git repositories in repo/, config/, and notes/.
          --no-repo-git
                       Do not create a git repository in repo/.
          --no-config-git
                       Do not create a git repository in config/.
          --no-notes-git
                       Do not create a git repository in notes/.
    """),
    "workspace": textwrap.dedent("""\
        agm workspace open  [-d|--detach] [-n|--num-panes PANES] [-p|--parent PARENT] TARGET
        agm workspace close [-f|--force] [-D] [--keep-branch] [--keep-workspace] BRANCH
        agm workspace setup
        agm workspace list  [-v|--verbose]
        agm workspace shell-regen SHELL_DIR
        agm wsp open        [-d|--detach] [-n|--num-panes PANES] [-p|--parent PARENT] TARGET
        agm wsp close       [-f|--force] [-D] [--keep-branch] [--keep-workspace] BRANCH
        agm wsp setup
        agm wsp list        [-v|--verbose]

        Manage AGM workspaces. A workspace may be the main repo or a linked
        Git worktree, interpreted with AGM project config, workspace config,
        dependency environment, setup scripts, and tmux session lifecycle.
    """),
    "sync": textwrap.dedent("""\
        agm sync fetch
        agm sync pull

        Synchronize the main repository, dependency repositories, and their
        checked-out Git worktrees.

        Commands:
          fetch   Fetch the main repository and all checked-out dependencies,
                  then create missing local tracking branches.
          pull    Run sync fetch, then run git merge in every Git worktree.
    """),
    "loop": textwrap.dedent("""\
        agm loop [--runner COMMAND] [--selector COMMAND|--no-selector]
                 [--tasks-dir DIR] [--no-log|--log-file PATH]
                 [--prompt TEXT|--prompt-file PATH]
                 [--selector-prompt TEXT|--selector-prompt-file PATH]
                 [--extra-prompt TEXT|--extra-prompt-file PATH]
                 [--extra-selector-prompt TEXT|--extra-selector-prompt-file PATH]
                 [--timeout DURATION]
                 CMD [RUNNER_ARGS...]
        agm loop run [--runner COMMAND] [--selector COMMAND|--no-selector]
                     [--tasks-dir DIR] [--no-log|--log-file PATH]
                     [--prompt TEXT|--prompt-file PATH]
                     [--selector-prompt TEXT|--selector-prompt-file PATH]
                     [--extra-prompt TEXT|--extra-prompt-file PATH]
                     [--extra-selector-prompt TEXT|--extra-selector-prompt-file PATH]
                     [--timeout DURATION]
                     [CMD [RUNNER_ARGS...]]
        agm loop step [--runner COMMAND] [--selector COMMAND|--no-selector]
                      [--tasks-dir DIR] [--no-log|--log-file PATH]
                      [--prompt TEXT|--prompt-file PATH]
                      [--selector-prompt TEXT|--selector-prompt-file PATH]
                      [--extra-prompt TEXT|--extra-prompt-file PATH]
                      [--extra-selector-prompt TEXT|--extra-selector-prompt-file PATH]
                      [--timeout DURATION]
                      CMD [RUNNER_ARGS...]
        agm loop select [--runner COMMAND] [--selector COMMAND|--no-selector]
                        [--tasks-dir DIR] [--prompt TEXT|--prompt-file PATH]
                        [--selector-prompt TEXT|--selector-prompt-file PATH]
                        [--extra-prompt TEXT|--extra-prompt-file PATH]
                        [--extra-selector-prompt TEXT|--extra-selector-prompt-file PATH]
                        [--timeout DURATION]
                        [CMD [RUNNER_ARGS...]]

        Repeatedly run a prompt command until the selected loop mode reports
        completion, perform one loop iteration, or run the progress-update
        prompt once.

        Command config:
          [loop] runner = "claude -p" in config.toml sets the default runner
          command prefix. [loop] selector = "codex exec" sets the selector
          command prefix. [loop] no_selector = true disables the selector and
          switches to no-selector mode. [loop] tasks_dir = ".agent-files/tasks"
          sets the tasks directory checked for ``PROGRESS.md`` and task files.
          [loop] timeout = "30m" sets an idle timeout that kills the runner
          process tree when no output is received for the given duration.
          Accepts seconds (plain number or ``Ns``), minutes (``Nm``), or
          hours (``Nh``). Disabled by default. ``--timeout DURATION``
          overrides the config value.
          [loop] prompt = "text" or [loop] prompt_file = "path" set the prompt
          text or file, overriding the default task file (selector mode) or
          loop.md (no-selector mode). ``--prompt`` and ``--prompt-file`` are
          mutually exclusive and override the config values.
          [loop] selector_prompt = "text" or [loop] selector_prompt_file = "path"
          set the selector prompt text or file, overriding the default
          select.md prompt. ``--selector-prompt`` and
          ``--selector-prompt-file`` are mutually exclusive and override the
          config values.
          [loop] extra_prompt = "text" or [loop] extra_prompt_file = "path"
          append extra content to the runner prompt, after the primary
          prompt (whether default or explicitly set).
          ``--extra-prompt`` and ``--extra-prompt-file`` are mutually
          exclusive and override the config values.
          [loop] extra_selector_prompt = "text" or
          [loop] extra_selector_prompt_file = "path" append extra content
          to the selector prompt, after the primary selector prompt
          (whether default or explicitly set).
          ``--extra-selector-prompt`` and ``--extra-selector-prompt-file``
          are mutually exclusive and override the config values.
          ``agm loop CMD`` is shorthand for ``agm loop run CMD`` when ``CMD``
          is not a built-in subcommand, and still selects ``[loop.CMD]``
          overrides; those values override ``[loop]``. ``agm loop --runner "..."``,
          ``agm loop --selector "..."``, ``agm loop --no-selector``, and
          ``agm loop --tasks-dir ...`` override those values. ``RUNNER_ARGS``
          are appended to the final runner command after AGM resolves
          ``--runner``, config, or the built-in default. Bare ``agm loop``
          prints this help text.

        Behavior:
          With a selector (the default), AGM runs the selector with
          ``@select.md``. If the selector returns ``COMPLETE`` after
          whitespace is removed, AGM stops. Otherwise the selector output is
          treated as the next task path and AGM runs the runner with that task
          file. When no explicit selector command is configured, the runner
          command is used for the progress update.
          With ``--no-selector`` / ``no_selector = true``, AGM appends
          ``@<resolved-loop-prompt>`` as the final argument to the runner and
          stops when the response is ``COMPLETE`` after whitespace is removed.
          Creates a ``loop-YYYYMMDD-HHMMSS.log`` file in the current directory
          by default, or writes to ``--log-file PATH``. ``--no-log`` disables
          file logging entirely. The command prints each step header and stops
          when the active mode reports completion.
          By default, AGM appends the prompt file path as a trailing
          ``@<path>`` argument to the runner/selector command. To control
          where the path appears, use the ``%%`` or ``%{PROMPT_FILE}``
          placeholder in the command — it is replaced with the resolved
          prompt file path. If neither placeholder is present, the ``@<path>``
          suffix is appended as usual.
          ``--prompt TEXT`` or ``--prompt-file PATH`` (or the corresponding
          config.toml ``prompt`` / ``prompt_file`` keys) specify the prompt
          to feed the runner, replacing the default task file in selector
          mode or the loop.md prompt file in no-selector mode. The prompt
          text is saved to a temporary file and processed with env var
          substitution, just like loop.md. ``%%`` and ``%{PROMPT_FILE}`` in
          the command string resolve to this processed prompt file. In
          selector mode, the selected task file path is available in the
          ``TASK_FILE`` environment variable passed to the runner.
          ``--selector-prompt TEXT`` or ``--selector-prompt-file PATH`` (or
          the corresponding config.toml ``selector_prompt`` /
          ``selector_prompt_file`` keys) specify the prompt to feed the
          selector, replacing the default ``select.md`` prompt.
          The prompt text is saved to a temporary file and processed with
          env var substitution, just like the default prompt.
          ``--extra-prompt TEXT`` or ``--extra-prompt-file PATH`` (or the
          corresponding config.toml ``extra_prompt`` / ``extra_prompt_file``
          keys) append extra content to the runner prompt, after the
          primary prompt (whether default or explicitly set). ``--extra-prompt``
          and ``--extra-prompt-file`` are mutually exclusive.
          ``--extra-selector-prompt TEXT`` or ``--extra-selector-prompt-file PATH``
          (or the corresponding config.toml ``extra_selector_prompt`` /
          ``extra_selector_prompt_file`` keys) append extra content to the
          selector prompt, after the primary selector prompt (whether default
          or explicitly set). ``--extra-selector-prompt`` and
          ``--extra-selector-prompt-file`` are mutually exclusive.

        Prompt preprocessing:
          Before a prompt file is passed to the runner or selector, AGM
          expands environment variable references in the prompt content
          using ``$VAR`` or ``${VAR}`` syntax. Unrecognized variables are
          left unchanged. When expansions modify the content, AGM writes
          the expanded text to a temporary file; otherwise the original
          file path is used. Beyond the process environment, AGM provides:
            TASKS_DIR  the resolved tasks directory path
            TASK_FILE  the selected task file path (selector mode; set
                       in the runner process environment at runtime)

          ``agm loop step`` performs a single loop iteration using the same
          runner, selector, and logging behavior as ``agm loop run``.
          ``agm loop select`` runs ``select.md`` once using the
          resolved selector, or the resolved runner when no selector is
          configured. It requires selector mode; ``--no-selector`` is an
          error for ``loop select``.
    """),
    "review": textwrap.dedent("""\
        agm review [COMMAND] [--scope REVIEW_SCOPE] [--aspects REVIEW_ASPECTS]
                   [--extra-aspects REVIEW_ASPECTS] [--runner COMMAND]
                   [--prompt TEXT|--prompt-file PATH]
                   [--extra-prompt TEXT|--extra-prompt-file PATH]
                   [--review-file FILE|auto|none|--no-review-file]

        Run the review prompt with REVIEW_SCOPE and REVIEW_ASPECTS available
        during prompt preprocessing. The default prompt is review.md.
        Review output is also saved to .agent-files/review-YYYYMMDD-HHMMSS-microseconds.md
        by default. Use --review-file FILE to choose a path, --review-file none
        or --no-review-file to disable saving, and --review-file auto to use
        the default timestamped path.
        When COMMAND is provided, config from [review.COMMAND] is merged over
        [review].

        Command config:
          [review] runner = "claude -p" sets the review runner. When unset,
          AGM uses the same default runner as agm loop.
          [review] scope, aspects, extra_aspects, prompt, prompt_file,
          extra_prompt, extra_prompt_file, and review_file correspond to the
          CLI options.
    """),
    "revise": textwrap.dedent("""\
        agm revise [COMMAND] [--runner COMMAND] [--prompt TEXT|--prompt-file PATH]
                   [--extra-prompt TEXT|--extra-prompt-file PATH]
                   REVIEW_FILE

        Run the revision prompt with REVIEW_FILE available during prompt
        preprocessing. The default prompt is revise.md.
        When COMMAND is provided before REVIEW_FILE, config from
        [revise.COMMAND] is merged over [revise].

        Command config:
          [revise] runner = "claude -p" sets the revision runner. When unset,
          AGM uses the same default runner as agm loop.
          [revise] prompt, prompt_file, extra_prompt, and extra_prompt_file
          correspond to the CLI options.
    """),
    "refine": textwrap.dedent("""\
        agm refine [COMMAND] [--max-steps N|unlimited] [--no-max-steps] [--runner COMMAND]
                   [--reviewer COMMAND] [--reviser COMMAND]
                   [--scope REVIEW_SCOPE] [--aspects REVIEW_ASPECTS]
                   [--review-prompt TEXT|--review-prompt-file PATH]
                   [--extra-review-prompt TEXT|--extra-review-prompt-file PATH]
                   [--revise-prompt TEXT|--revise-prompt-file PATH]
                   [--extra-revise-prompt TEXT|--extra-revise-prompt-file PATH]
                   [--save-review|--no-save-review] [--review-file FILE|auto|none]
                   [--log-file PATH|--no-log]

        Run review/revise cycles until revise returns COMPLETE, or until the
        maximum number of revision attempts is reached. A CONTINUE response
        starts a fresh review; any other response retries revise with the same
        review file. The default maximum is 12.
        Review output is saved to the default timestamped review path by
        default. Use --no-save-review to keep review handoff files temporary
        only, or --review-file FILE to choose a custom path.
        When COMMAND is provided, config from [refine.COMMAND] is merged over
        [refine] and the same command name is forwarded to review/revise
        config lookup.

        Logging:
          By default writes refine-YYYYMMDD-HHMMSS.log in the current
          directory. --log-file PATH writes to a specific file. --no-log
          disables file logging.

        Command config:
          [refine] max_steps, no_max_steps, runner, reviewer, reviser, scope, aspects,
          review_prompt, review_prompt_file, extra_review_prompt,
          extra_review_prompt_file, revise_prompt, revise_prompt_file,
          extra_revise_prompt, extra_revise_prompt_file, save_review,
          log_file, and no_log correspond to the CLI options.
    """),
    "config": textwrap.dedent("""\
        agm config copy DIRNAME
        agm config cp   DIRNAME
        agm config env
        agm config update

        Copy project dot configuration files into an existing target directory.
        Print shell statements that refresh the current workspace environment
        from project and workspace config.toml [deps] tables, .env,
        .env.local, and env.sh files.
        Create missing project and workspace config.toml files under the project
        config directory.

        To apply the environment to the current shell:
          eval "$(agm config env)"
    """),
    "worktree": textwrap.dedent("""\
        agm worktree new      [-d|--dir DIR] BRANCH
        agm worktree remove   [-f|--force] BRANCH
        agm wt new            [-d|--dir DIR] BRANCH
        agm wt rm             [-f|--force] BRANCH

        Low-level git worktree management.

        Options:
          agm worktree new --dir DIR BRANCH
              Create the worktree under DIR instead of the default project
              worktrees directory.
          agm worktree remove --force BRANCH
              Force removal even when git reports uncommitted or locked state.
    """),
    "dep": textwrap.dedent("""\
        agm dep list   [-v|--verbose] [--all]
        agm dep new    [-b|--branch BRANCH] REPO_URL
        agm dep rm     --all DEP
        agm dep rm     DEP/NAME_OR_BRANCH | DEP/repo | DEP/MAIN_CHECKOUT
        agm dep switch [-b|--branch] DEP BRANCH

        Manage dependency repos and dependency Git worktrees under deps/.
        AGM tracks dependency checkout names in config.toml [deps] tables.

        Options:
          agm dep list --verbose
              Show the checkout path after each dep/branch.
          agm dep list --all
              List all dependency checkouts on disk, instead of
              only the current workspace.
          agm dep new --branch BRANCH REPO_URL
              Clone the dependency's initial checkout from BRANCH instead of
              its default branch.
          agm dep rm --all DEP
              Remove the entire dependency directory, including the main repo
              checkout and any linked worktrees.
          agm dep switch --branch DEP BRANCH
              Create DEP's BRANCH from the dependency's default branch, then
              add a worktree for it. Without this flag, BRANCH must already
              exist in the dependency repo.

        Targets:
          DEP/NAME_OR_BRANCH Remove a dependency checkout by directory name
                             under deps/DEP/ or by checked-out branch name.
          DEP/repo           Remove the main dependency checkout.
          DEP/MAIN_CHECKOUT  Remove the main dependency checkout by directory name.
    """),
    "run": textwrap.dedent("""\
        agm run [--no-sandbox] [--no-patch] [--memory LIMIT] [--swap LIMIT]
        [--no-memory-limit] [--no-swap-limit] [-f|--file SETTINGS] COMMAND [ARGS...]

        Run a command inside an Anthropic Sandbox Runtime container.

        Command config:
          <install-prefix>/.agm/config.toml is loaded first, then
          the AGM home config ($AGM_HOME/config.toml, or
          $HOME/.agm/config.toml when AGM_HOME is unset), followed by the
          project config.toml and ./.agm/config.toml.
          [run.<command>] alias = "<other-command>" makes
          "agm run <command>" execute <other-command> instead.

        Options:
          --no-sandbox
                       Run COMMAND directly without wrapping it in srt.
                       This skips sandbox settings discovery and patching.
          -f, --file SETTINGS
                       Use this settings file directly instead of discovering
                       and combining the default sandbox settings files.
          --memory LIMIT
                       Wrap COMMAND in a delegated systemd-run --user --scope
                       with MemoryMax=LIMIT, optional MemorySwapMax, and
                       Delegate=yes.
                       The wrapper exports SANDBOX_CGROUP and enables the
                       memory controller for descendant cgroups. The default
                       memory limit is 32G. Use 0 for a zero limit or
                       unlimited for no memory cap.
          --swap LIMIT
                       Set MemorySwapMax=LIMIT in the delegated systemd-run
                       scope. In sandbox mode the default is 0. Use unlimited
                       for no swap cap.
          --no-memory-limit
                       Do not set MemoryMax.
          --no-swap-limit
                       Do not set MemorySwapMax.
          --no-patch   Do not append the project notes, deps, and repo .git
                       paths to filesystem.allowWrite after loading the
                       selected settings.

        Settings resolution:
          default      For each directory below, load <command>.json when it
                       exists there; otherwise try the aliased command's
                       settings file, then fall back to default.json.
                       Then merge the existing files in this order:
                         1. $AGM_HOME/sandbox/<command>.json
                            (or $HOME/.agm/sandbox/<command>.json when unset)
                            fallback: $AGM_HOME/sandbox/default.json
                            (or $HOME/.agm/sandbox/default.json when unset)
                         2. the project sandbox config directory
                         3. ./.sandbox/<command>.json
                            fallback: ./.sandbox/default.json
                       Later files are merged over earlier ones. network and
                       filesystem are merged by key; list-valued keys are
                       appended and deduplicated in precedence order. Later
                       network.deniedDomains removes earlier allowedDomains;
                       later filesystem denyRead/denyWrite removes earlier
                       allowRead/allowWrite. ignoreViolations replaces the
                       earlier value; enabled and enableWeakerNestedSandbox are
                       overridden when set.
          -f, --file SETTINGS
                       Skip default discovery and use SETTINGS as-is.

        Automatic patching:
          Unless --no-patch is set, agm adds the project notes, deps, and
          repo .git paths to filesystem.allowWrite when PROJ_DIR is set.
    """),
    "tmux": textwrap.dedent("""\
        agm tmux open   [-d|--detach] [-n|--num-panes PANES] [SESSION]
        agm tmux close  SESSION
        agm tmux layout PANES [-w|--window WINDOW_ID]

        Tmux session and layout management.

        Options:
          agm tmux open --detach
              Create the session without attaching to it.
          agm tmux open --num-panes PANES
              Create the session with PANES panes.
    """),
    "exec": textwrap.dedent("""\
        agm exec [--strict-json|--no-strict-json] [--max-iters N]
                 [--max-call-depth N] [--runner COMMAND]
                 [--timeout DURATION|--no-timeout] [--dry-run]
                 [--log|--log-file PATH|--no-log] [--no-log-file]
                 [--no-stdlib] [-I DIR]...
                 (FILE | -c COMMAND) [--PARAM VALUE]...

        Execute an AgL (Agent Language) workflow program from FILE, or from
        the inline program text given with -c/--command.

        Each `param` declaration in the program becomes a `--<name>` option.
        Boolean params use the `--name/--no-name` flag form. Structured types
        take a JSON string. Run `agm exec FILE --help` to show discovered params.

        Trace logging is OFF by default.  Enable it with --log, --log-file, or
        [exec] log = true in config.toml.  A source ``std/config::KEY := VALUE``
        write takes effect from its program point and overrides the CLI flag,
        which overrides the config-file layer.

        Options:
          -c, --command COMMAND  Execute the program given as COMMAND instead of FILE.
          --strict-json         Require bare JSON output from agents (no recovery).
          --no-strict-json      Use lenient JSON recovery (default).
          --max-iters N         Cap unbounded loops; off by default (CLI > config).
          --max-call-depth N    Override the maximum recursion call depth
                                (CLI > config).
          --runner COMMAND      Override the default agent runner command.
          --timeout DURATION    Override initial shell-exec and agent idle timeouts;
                                seed std/config::timeout to some(DURATION). Mutually
                                exclusive with --no-timeout.
          --no-timeout          Remove any configured shell-exec timeout and seed
                                std/config::timeout to none. Mutually exclusive
                                with --timeout.
          --dry-run             Run the full static pipeline and validate parameters,
                                but do not execute the workflow.
          --log                 Enable trace logging (auto timestamped path).
          --log-file PATH       Write trace log to PATH.
          --no-log-file         Clear the CLI log-file seed only; use --no-log to
                                disable tracing entirely.
          --no-log              Disable trace logging (overrides config).
          --log, --log-file, and --no-log are mutually exclusive.
          --log-file and --no-log-file are mutually exclusive.
          --no-stdlib           Do not automatically open std/core in the entry module.
          -I DIR, --module-path DIR
                                Add DIR as an additional module search root
                                (repeatable). Resolved relative to the invocation
                                working directory. Joins the unordered root set;
                                a module id found in two roots is an ambiguity error.

        FILE and -c/--command are mutually exclusive; exactly one is required.

        Exit codes:
          0  The workflow completed successfully.
          1  Pre-execution failure: unreadable file, static diagnostics
             (lex/parse/scope/type/match), host configuration error, or param
             validation failure.
          2  The workflow executed but ended with an uncaught AgL exception.
    """),
    "repl": textwrap.dedent("""\
        agm repl [--strict-json|--no-strict-json] [--max-iters N] [--max-call-depth N]
                 [--runner COMMAND] [--confirm-agents] [--dry-run]
                 [--quiet] [--log|--log-file PATH|--no-log]

        Start an interactive read-eval-print loop for AgL.  Each entry is
        parsed, type-checked, and evaluated once against a persistent session
        that accumulates bindings, types, and declarations across entries, so
        earlier results stay available and agent calls fire exactly once.  The
        session reuses the [exec] configuration (runner, per-agent commands,
        call-depth limit, JSON strictness, timeout).  Like agm exec, it
        automatically opens std/core so standard-library names are available
        unqualified.

        Trace logging is OFF by default.  A ``std/config::KEY := VALUE`` write
        entered at the REPL prompt takes effect from that point and persists for
        the session; :reset clears it.  Set session-wide defaults via CLI flags
        or [exec] config.

        Params (`param NAME: T`) resolve eagerly when entered: first from
        [<program>] config when a `program NAME` declaration is active,
        then from the param default expression. There is no CLI param seeding.
        Use :params to list declared params and their resolved values.

        Options:
          --strict-json         Require bare JSON output from agents (no recovery).
          --no-strict-json      Use lenient JSON recovery (default).
          --max-iters N         Positive cap for unbounded loops; off by default
                                (source writes > CLI > config).
          --max-call-depth N    Override the maximum recursion call depth
                                (CLI > config; source pragmas are not applied in the REPL).
          --runner COMMAND      Override the default agent runner command.
          --confirm-agents     Confirm each agent call before dispatching it
                                (default: fire agent calls without confirming).
          --quiet               Suppress automatic echoing of entry results.
          --log                 Enable trace logging (auto timestamped path).
          --log-file PATH       Write a JSONL trace log to PATH.
          --no-log              Disable trace logging.
          --log, --log-file, and --no-log are mutually exclusive.
          --dry-run             Statically check only: run the full static pipeline
                                for each entry but never evaluate it (no agent/exec
                                calls, no persisted bindings); echo the inferred type.

        Type :help inside the REPL for the meta-command list; :quit or Ctrl-D
        exits.  Ctrl-C cancels the current entry without exiting.

        Exit codes:
          0  The session ended normally (:quit / :exit / Ctrl-D).
          1  Pre-loop setup failure: invalid [exec] config, --runner, or an
             unwritable --log-file (reported before the prompt).
    """),
    "help": textwrap.dedent("""\
        agm help [COMMAND...]

        Show help information for commands and subcommands.

        Global options:
          --install-completion  Install shell completion for the current shell.
          --show-completion     Print the shell completion script.
    """),
}

_HELP_ALIASES: dict[str, str] = {
    "wt": "worktree",
    "wsp": "workspace",
    "cp": "config",
    "copy": "config",
}

_PATH_HELP_TEXTS: dict[tuple[str, ...], str] = {
    ("workspace", "open"): textwrap.dedent("""\
        agm workspace open [-d|--detach] [-n|--num-panes PANES] [-p|--parent PARENT] TARGET

        Open a tmux session for an AGM workspace, creating or checking out a
        branch workspace as needed.
    """),
    ("wsp", "open"): textwrap.dedent("""\
        agm wsp open [-d|--detach] [-n|--num-panes PANES] [-p|--parent PARENT] TARGET

        Alias form of agm workspace open.
    """),
    ("workspace", "close"): textwrap.dedent("""\
        agm workspace close [-f|--force] [-D] [--keep-branch] [--keep-workspace] BRANCH

        Close a branch workspace, remove its Git worktree unless --keep-workspace is used,
        remove workspace config when removing the worktree, and kill its tmux session.
    """),
    ("wsp", "close"): textwrap.dedent("""\
        agm wsp close [-f|--force] [-D] [--keep-branch] [--keep-workspace] BRANCH

        Alias form of agm workspace close.
    """),
    ("workspace", "setup"): textwrap.dedent("""\
        agm workspace setup

        Run configured setup scripts for the current AGM workspace.
    """),
    ("wsp", "setup"): textwrap.dedent("""\
        agm wsp setup

        Alias form of agm workspace setup.
    """),
    ("workspace", "list"): textwrap.dedent("""\
        agm workspace list [-v|--verbose]

        List all open AGM workspaces.
    """),
    ("wsp", "list"): textwrap.dedent("""\
        agm wsp list [-v|--verbose]

        Alias form of agm workspace list.
    """),
    ("workspace", "shell-regen"): textwrap.dedent("""\
        agm workspace shell-regen SHELL_DIR

        Regenerate the per-session shell wrapper and rc files in SHELL_DIR.

        Invoked by the workspace shell wrapper to self-heal after its cache
        directory is deleted. Not normally called directly.
    """),
    ("wsp", "shell-regen"): textwrap.dedent("""\
        agm wsp shell-regen SHELL_DIR

        Alias form of agm workspace shell-regen.
    """),
    ("sync", "fetch"): textwrap.dedent("""\
        agm sync fetch

        Fetch the main repository and all checked-out dependencies, then create
        missing local tracking branches.
    """),
    ("sync", "pull"): textwrap.dedent("""\
        agm sync pull

        Run agm sync fetch, then run git merge in every Git worktree: all
        dependency worktrees, the main repository workspace, and every branch
        workspace.
    """),
    ("config", "cp"): textwrap.dedent("""\
        agm config cp DIRNAME

        Copy project dot config files into an existing target directory.
    """),
    ("config", "copy"): textwrap.dedent("""\
        agm config copy DIRNAME

        Copy project dot config files into an existing target directory.
    """),
    ("config", "env"): textwrap.dedent("""\
        agm config env

        Print shell statements that refresh the current workspace environment
        from project and workspace config.toml [deps] tables, .env,
        .env.local, and env.sh files.

        To apply the environment to the current shell:
          eval "$(agm config env)"
    """),
    ("config", "update"): textwrap.dedent("""\
        agm config update

        Create missing project and workspace config.toml files under the project
        config directory.
    """),
    ("wt", "new"): textwrap.dedent("""\
        agm wt new [-d|--dir DIR] BRANCH

        Create a bare Git worktree or check out an existing branch.
    """),
    ("wt", "rm"): textwrap.dedent("""\
        agm wt rm [-f|--force] BRANCH

        Remove a worktree and delete its local branch.
    """),
    ("wt", "remove"): textwrap.dedent("""\
        agm wt remove [-f|--force] BRANCH

        Remove a worktree and delete its local branch.
    """),
    ("loop", "select"): textwrap.dedent("""\
        agm loop select [--runner COMMAND] [--selector COMMAND|--no-selector]
                        [--tasks-dir DIR] [--prompt TEXT|--prompt-file PATH]
                        [--selector-prompt TEXT|--selector-prompt-file PATH]
                        [--extra-prompt TEXT|--extra-prompt-file PATH]
                        [--extra-selector-prompt TEXT|--extra-selector-prompt-file PATH]
                        [--timeout DURATION]
                        [CMD [RUNNER_ARGS...]]

        Run the update-progress prompt once using the resolved selector, or
        the resolved runner when no selector is configured. Requires selector
        mode; ``--no-selector`` is an error for this subcommand.
        ``--selector-prompt TEXT`` or ``--selector-prompt-file PATH`` overrides
        the default select.md prompt. ``--timeout DURATION`` sets an idle
        timeout; see ``agm help loop`` for details. Prompt files are
        preprocessed for environment variable expansion.
    """),
    ("loop", "run"): textwrap.dedent("""\
        agm loop run [--runner COMMAND] [--selector COMMAND|--no-selector]
                     [--tasks-dir DIR] [--no-log|--log-file PATH]
                     [--prompt TEXT|--prompt-file PATH]
                     [--selector-prompt TEXT|--selector-prompt-file PATH]
                     [--extra-prompt TEXT|--extra-prompt-file PATH]
                     [--extra-selector-prompt TEXT|--extra-selector-prompt-file PATH]
                     [--timeout DURATION]
                     [CMD [RUNNER_ARGS...]]

        Run the loop prompt until completion. Selector mode is the default;
        ``--no-selector`` switches to the no-selector loop-prompt mode.
        ``--prompt TEXT`` or ``--prompt-file PATH`` overrides the default
        prompt file. ``--selector-prompt TEXT`` or ``--selector-prompt-file
        PATH`` overrides the default select.md selector prompt. ``--timeout
        DURATION`` sets an idle timeout; see ``agm help loop`` for details.
        Prompt files are preprocessed for environment variable expansion; see
        ``agm help loop`` for details.
        Bare ``agm loop CMD`` is a shorthand for this command when ``CMD``
        is not a built-in subcommand.
    """),
    ("loop", "step"): textwrap.dedent("""\
        agm loop step [--runner COMMAND] [--selector COMMAND|--no-selector]
                      [--tasks-dir DIR] [--no-log|--log-file PATH]
                      [--prompt TEXT|--prompt-file PATH]
                      [--selector-prompt TEXT|--selector-prompt-file PATH]
                      [--extra-prompt TEXT|--extra-prompt-file PATH]
                      [--extra-selector-prompt TEXT|--extra-selector-prompt-file PATH]
                      [--timeout DURATION]
                      CMD [RUNNER_ARGS...]

        Perform one loop iteration using the same runner and selector
        resolution as ``agm loop run``. ``--prompt TEXT`` or
        ``--prompt-file PATH`` overrides the default prompt file.
        ``--selector-prompt TEXT`` or ``--selector-prompt-file PATH`` overrides
        the default select.md selector prompt. ``--timeout DURATION`` sets an
        idle timeout; see ``agm help loop`` for details. Prompt files are
        preprocessed for environment variable expansion; see
        ``agm help loop`` for details.
    """),
    ("worktree", "new"): textwrap.dedent("""\
        agm worktree new [-d|--dir DIR] BRANCH

        Create a bare Git worktree or check out an existing branch.
    """),
    ("worktree", "remove"): textwrap.dedent("""\
        agm worktree remove [-f|--force] BRANCH

        Remove a worktree and delete its local branch.
    """),
    ("worktree", "rm"): textwrap.dedent("""\
        agm worktree rm [-f|--force] BRANCH

        Remove a worktree and delete its local branch.
    """),
    ("dep", "list"): textwrap.dedent("""\
        agm dep list [-v|--verbose] [--all]

        List dependency checkouts for the current workspace. With --all, list
        all dependency checkouts for every workspace. By default only dep/branch
        names are printed; with -v/--verbose the checkout path is also shown.
    """),
    ("dep", "new"): textwrap.dedent("""\
        agm dep new [-b|--branch BRANCH] REPO_URL

        Clone a dependency into deps/ using its default branch or BRANCH.
    """),
    ("dep", "switch"): textwrap.dedent("""\
        agm dep switch [-b|--branch] DEP BRANCH

        Select an existing dependency checkout by directory name or checked-out
        branch name. If neither exists, add a worktree at deps/DEP/BRANCH for
        an existing dependency branch. With -b/--branch, create DEP's BRANCH
        from the dependency's default branch first. Updates the relevant
        config.toml [deps] entry with the dependency checkout directory name.
    """),
    ("dep", "rm"): textwrap.dedent("""\
        agm dep rm --all DEP
        agm dep rm TARGET

        Remove a dependency worktree by DEP/NAME_OR_BRANCH, or remove the main
        checkout with DEP/repo, DEP/MAIN_CHECKOUT, or --all DEP.
    """),
    ("dep", "remove"): textwrap.dedent("""\
        agm dep remove --all DEP
        agm dep remove TARGET

        Remove a dependency worktree by DEP/NAME_OR_BRANCH, or remove the main
        checkout with DEP/repo, DEP/MAIN_CHECKOUT, or --all DEP.
    """),
    ("tmux", "open"): textwrap.dedent("""\
        agm tmux open [-d|--detach] [-n|--num-panes PANES] [SESSION]

        Create a tmux session, optionally detached and with a chosen pane count.
    """),
    ("tmux", "close"): textwrap.dedent("""\
        agm tmux close SESSION

        Kill an existing tmux session by name.
    """),
    ("tmux", "layout"): textwrap.dedent("""\
        agm tmux layout PANES [-w|--window WINDOW_ID]

        Apply AGM's tiled pane layout to the current tmux window.
    """),
}


def _overview_text() -> str:
    lines = [
        "agm - Agent Management Framework",
        "",
        "Usage: agm <command> [options] [args]",
        "",
        "Commands:",
    ]
    width = max(len(name) for name, _ in COMMAND_OVERVIEW)
    for name, desc in COMMAND_OVERVIEW:
        lines.append(f"  {name:<{width + 2}} {desc}")
    lines.extend(
        [
            "",
            "Global options:",
            "  --dry-run             Print planned commands and AGM operations only.",
            "  --install-completion  Install shell completion for the current shell.",
            "  --show-completion     Print the shell completion script.",
            "",
            "Run 'agm help <command>' for detailed help on a specific command.",
        ]
    )
    return "\n".join(lines) + "\n"


def help_text_for(command: str) -> str | None:
    canonical = _HELP_ALIASES.get(command, command)
    return _HELP_TEXTS.get(canonical)


def print_overview(file: _Writeable | None = None) -> None:
    output = sys.stdout if file is None else file
    print(_overview_text(), end="", file=output)


def print_command_help(command: str, file: _Writeable | None = None) -> None:
    text = help_text_for(command)
    if text is None:
        print(f"agm: unknown command '{command}'", file=sys.stderr)
        print("\nRun 'agm help' to see available commands.", file=sys.stderr)
        raise SystemExit(1)
    output = sys.stdout if file is None else file
    print(text, end="", file=output)


def _canonical_command_path(command_path: Sequence[str]) -> tuple[str, ...]:
    if len(command_path) == 1:
        return (_HELP_ALIASES.get(command_path[0], command_path[0]),)
    return tuple(command_path)


def _help_text_for_path(command_path: Sequence[str]) -> str:
    normalized = _canonical_command_path(command_path)
    if len(normalized) == 1:
        text = help_text_for(normalized[0])
        if text is None:
            raise ValueError(f"unknown command path: {' '.join(command_path)}")
        return text
    text = _PATH_HELP_TEXTS.get(tuple(command_path))
    if text is None:
        raise ValueError(f"unknown command path: {' '.join(command_path)}")
    return text


def print_help_for_command_path(
    command_path: Sequence[str],
    file: _Writeable | None = None,
) -> None:
    output = sys.stdout if file is None else file
    print(_help_text_for_path(command_path), end="", file=output)


def exit_with_usage_error(
    command_path: Sequence[str], message: str, *, exit_code: int = 1
) -> NoReturn:
    help_text = _help_text_for_path(command_path)
    usage_line, _, _ = help_text.partition("\n")
    print(message, file=sys.stderr)
    print(file=sys.stderr)
    print(f"usage: {usage_line}", file=sys.stderr)
    print(file=sys.stderr)
    print(help_text, end="", file=sys.stderr)
    raise SystemExit(exit_code)
