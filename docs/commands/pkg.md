# Packages

| Command | Description |
|---|---|
| `agm pkg check [DIR]` | Validate a package directory |

`agm pkg check` validates the `package.toml` manifest, module-tree naming discipline, and
program references used by manifest command registrations. `DIR` defaults to the current
directory. It does not install, create, or otherwise modify packages.
