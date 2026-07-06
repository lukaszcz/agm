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

- `-b`, `--branch BRANCH`: use `agm dep new --branch BRANCH REPO_URL` to clone the dependency's initial checkout from `BRANCH` instead of the dependency's default branch

`agm dep switch` options:

- `-b`, `--branch`: use `agm dep switch --branch DEP BRANCH` to create `DEP`'s `BRANCH` from the dependency's default branch before adding the new worktree; without this flag, `BRANCH` must already exist

Dependency commands track selected dependency checkout names in config `config.toml` `[deps]` tables. Environment loading turns those entries into dependency path variables, so `[deps].vyper-automation = "feat/app"` provides `VYPER_AUTOMATION=/path/to/proj/deps/vyper-automation/feat/app` before `.env` and `env.sh` are loaded. Opening a branch materializes only dependencies inherited from that branch's parent or the main config; dependency checkouts present on disk are not added to unrelated branch configs unless they are declared there.

`agm dep list` options:

- `-v`, `--verbose`: show the checkout path after each dep/branch
- `--all`: list all dependency checkouts on disk instead of only the current workspace's dependencies

`agm dep rm` targets:

- `DEP/NAME_OR_BRANCH`: remove a dependency checkout by directory name under `deps/DEP/` or by checked-out branch name
- `DEP/repo`: remove the main dependency checkout
- `DEP/MAIN_CHECKOUT`: remove the main dependency checkout by directory name

`agm dep rm` options:

- `--all DEP`: use `agm dep rm --all DEP` to remove the entire dependency directory, including the main dependency checkout and any linked worktrees
