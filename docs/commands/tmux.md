# tmux sessions

| Command | Description |
|---|---|
| `agm tmux open [-d\|--detach] [-n\|--num-panes PANES] [SESSION]` | Open a tmux session |
| `agm tmux close SESSION` | Close a tmux session |
| `agm tmux layout PANES [-w\|--window WINDOW_ID]` | Apply AGM's tmux pane layout to a window |

`agm tmux open` options:

- `-d`, `--detach`: create the session without attaching
- `-n`, `--num-panes PANES`: create the session with `PANES` panes

`agm tmux layout` options:

- `-w`, `--window WINDOW_ID`: apply the layout to a specific tmux window ID
