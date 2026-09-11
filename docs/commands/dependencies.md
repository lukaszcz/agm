# Dependencies

| Command | Description |
|---|---|
| `agm dep list [-v\|--verbose] [--all]` | List dependency checkouts |
| `agm dep new [-b\|--branch BRANCH] REPO_URL` | Clone a new dependency checkout |
| `agm dep switch [-b\|--branch] DEP BRANCH` | Select or add a dependency checkout |
| `agm dep rm --all DEP` | Remove an entire dependency directory |
| `agm dep rm DEP/NAME_OR_BRANCH \| DEP/repo \| DEP/MAIN_CHECKOUT` | Remove a dependency checkout or worktree |
| `agm dep remove --all DEP` | Alias form of `agm dep rm --all` |
| `agm dep remove DEP/NAME_OR_BRANCH \| DEP/repo \| DEP/MAIN_CHECKOUT` | Alias form of `agm dep rm` |

`agm dep new` options:

- `-b`, `--branch BRANCH`: clone the dependency's initial checkout from `BRANCH` instead of its default branch

`agm dep switch` options:

- `-b`, `--branch`: create `DEP`'s `BRANCH` from the dependency's default branch before adding the worktree; without this flag, `BRANCH` must already exist

Dependency commands track selected checkout names in `config.toml`'s `[deps]` table. Environment loading turns each entry into a `_DIR` path variable before `.env` and `env.sh` load, e.g. `[deps].vyper-automation = "feat/app"` provides `VYPER_AUTOMATION_DIR=/path/to/proj/deps/vyper-automation/feat/app`. Opening a branch materializes only dependencies inherited from that branch's parent or the main config; checkouts present on disk aren't added to unrelated branch configs unless declared there.

`agm dep list` options:

- `-v`, `--verbose`: show the checkout path after each dep/branch
- `--all`: list all dependency checkouts on disk instead of only the current workspace's dependencies

`agm dep rm` targets:

- `DEP/NAME_OR_BRANCH`: remove a dependency checkout by directory name under `deps/DEP/` or by checked-out branch name
- `DEP/repo`: remove the main dependency checkout
- `DEP/MAIN_CHECKOUT`: remove the main dependency checkout by directory name

`agm dep rm` options:

- `--all DEP`: remove the entire dependency directory, including the main checkout and any linked worktrees

Removal targets must be relative paths without `.` or `..` components. A dependency must resolve below `deps/`, and its worktrees below that dependency's directory. With `--all`, AGM checks every linked worktree before removing any; a detached worktree or one outside the dependency stops removal.
