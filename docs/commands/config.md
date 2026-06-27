# Configuration

| Command | Description |
|---|---|
| `agm config copy DIRNAME` | Copy known project config files into an existing target directory |
| `agm config cp DIRNAME` | Alias form of `agm config copy` |
| `agm config env` | Print shell statements for refreshing the current workspace environment |
| `agm config update` | Create missing config.toml files and commit generated changes |

`agm config copy` copies dot-prefixed files and directories from the shared project
config directory into an existing target directory. When run from a branch
worktree, AGM first copies shared dot entries, then copies matching entries from
the workspace config subdirectory so workspace entries override shared entries.
For `.env` and `.env.local`, AGM writes merged dotenv values using the same
precedence as `agm config env`: shared `.env`, shared `.env.local`, workspace
`.env`, then workspace `.env.local`.

`agm config env` uses the same environment resolution as `agm workspace open`: project and workspace
`config.toml` `[deps]` tables first, then project `.env`, project `.env.local`, project
`env.sh`, and matching workspace config files when the current workspace is a branch workspace.
Apply the printed shell statements with:

```bash
eval "$(agm config env)"
```

`agm config update` creates missing project and workspace `config.toml` files under the project
config directory, updates dependency configuration entries, and commits any generated
changes to the config repository's git history with a `chore: update config` commit message.

When the config directory is a git repository, AGM automatically commits changes it makes to
the config directory. In addition to `agm config update`, this covers `agm init`,
`agm open`, `agm close`, `agm dep new`, `agm dep switch`, and `agm worktree new`, each of
which commits the config it adds, updates, or removes for the affected workspace. Pass
`agm init --no-git-init` (or `--no-config-git`) to opt out of creating the config git
repository, which disables these automatic commits.
