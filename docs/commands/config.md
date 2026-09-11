# Configuration

| Command | Description |
|---|---|
| `agm config copy DIRNAME` | Copy known project config files into an existing target directory |
| `agm config cp DIRNAME` | Alias form of `agm config copy` |
| `agm config env` | Print shell statements for refreshing the current workspace environment |
| `agm config update` | Create missing config.toml files and commit generated changes |

`agm config copy` copies dot-prefixed files and directories from the shared project
config directory. From a branch worktree, it copies shared dot entries first, then
matching workspace config entries, so workspace overrides shared. For `.env` and
`.env.local` it merges dotenv values with `agm config env`'s precedence: shared
`.env`, shared `.env.local`, workspace `.env`, workspace `.env.local`.

`agm config env` resolves the environment like `agm workspace open`: project/workspace
`config.toml` `[deps]` tables first, then project `.env`, `.env.local`, `env.sh`, then
matching workspace config files (branch workspaces only). Dotenv `${VAR}` references
resolve against the current workspace environment plus earlier assignments, including
prior layers; `${VAR:-default}` defaults an undefined variable. Apply with:

```bash
eval "$(agm config env)"
```

`agm config update` creates missing project and workspace `config.toml` files under
the project config directory, updates dependency config entries, and commits generated
changes to the config repo with message `chore: update config`.

## Path-valued settings

Path-valued `config.toml` fields (e.g. `[modules] lib_root`, `roots`) expand `%{VAR}`
from the process environment and support a leading `~`. Interpolation is deliberately
lenient: unresolved or malformed holes stay verbatim — one config file serves every
command, so an irrelevant broken path must not block loading; any resulting failure
follows the consuming command's normal path semantics.

When the config directory is a git repository, AGM auto-commits changes it makes
there — covering `agm config update`, `agm init`, `agm open`, `agm close`,
`agm dep new`, `agm dep switch`, and `agm worktree new`, each committing the config it
adds, updates, or removes for the affected workspace. `agm init --no-git-init` (or
`--no-config-git`) opts out of creating the config git repo, disabling these
auto-commits.
