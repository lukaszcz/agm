# Worktrees

| Command | Description |
|---|---|
| `agm worktree new [-d\|--dir DIR] BRANCH` | Create a new branch worktree or check out an existing branch |
| `agm worktree remove [-f\|--force] BRANCH` | Remove a worktree and delete its local branch |
| `agm worktree rm [-f\|--force] BRANCH` | Alias form of `agm worktree remove` |
| `agm wt new [-d\|--dir DIR] BRANCH` | Alias form of `agm worktree new` |
| `agm wt rm [-f\|--force] BRANCH` | Alias form of `agm worktree remove` |
| `agm wt remove [-f\|--force] BRANCH` | Alias form of `agm worktree remove` |

`agm worktree new` options:

- `-d`, `--dir DIR`: use `agm worktree new --dir DIR BRANCH` to create the worktree under `DIR` instead of the project's default worktrees directory

`agm worktree remove` options:

- `-f`, `--force`: use `agm worktree remove --force BRANCH` to force removal even when git reports uncommitted or locked state
