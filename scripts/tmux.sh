#!/usr/bin/env zsh

emulate -L zsh
setopt extendedglob

script_name=${ZSH_ARGZERO:t}

usage() {
  local -i fd=${1:-1}
  print -u$fd "Usage: $script_name [-h|--help] [-d|--detach] [-n pane_count] [session_name]"
}

pane_count=4
detach_requested=0

if (( ${argv[(I)--help]} )); then
  usage
  exit 0
fi

if (( ${argv[(I)--detach]} )); then
  detach_requested=1
  argv=("${(@)argv:#--detach}")
fi

while getopts ":hdn:" opt; do
  case "$opt" in
    h)
      usage
      exit 0
      ;;
    d)
      detach_requested=1
      ;;
    n)
      if [[ $OPTARG != <-> ]] || (( OPTARG < 1 )); then
        print -u2 "Invalid pane count: $OPTARG"
        usage 2
        exit 1
      fi
      pane_count=$OPTARG
      ;;
    :)
      print -u2 "Option -$OPTARG requires an argument"
      usage 2
      exit 1
      ;;
    \?)
      print -u2 "Unknown option: -$OPTARG"
      usage 2
      exit 1
      ;;
  esac
done

shift $((OPTIND - 1))

if (( $# > 1 )); then
  usage 2
  exit 1
fi

session_name="${1:-}"

# Exact variable names to skip.
local -a skip_names=(
  TMUX
  TERM
  DISPLAY
  PWD
  OLDPWD
  SHELL
  SHLVL
  LOGNAME
  USER
  _
)

# Variable-name prefixes to skip.
local -a skip_prefixes=(
  TMUX_
  TERM_
  SSH_
  DBUS_
  XDG_
)

local -a tmux_env_args
local -a tmux_command
local -a session_name_args
local line name value prefix
local skip
local apply_layout_command
local script_dir
local target_session
local window_id
local window_width
local window_height
local -i pane_index
local -i create_detached_session
local -i switch_to_session

create_detached_session=detach_requested
switch_to_session=0

if [[ -n "${TMUX:-}" ]] && (( ! detach_requested )); then
  create_detached_session=1
  switch_to_session=1
fi

while IFS= read -r line; do
  name=${line%%=*}
  value=${line#*=}

  skip=0

  # Skip exact names.
  if (( ${skip_names[(Ie)$name]} )); then
    skip=1
  else
    # Skip matching prefixes.
    for prefix in "${skip_prefixes[@]}"; do
      if [[ $name == ${prefix}* ]]; then
        skip=1
        break
      fi
    done
  fi

  (( skip )) && continue

  tmux_env_args+=(-e "$name=$value")
done < <(env)

script_dir=${0:A:h}
apply_layout_command="$script_dir/tmux-apply-layout.sh $pane_count '#{window_id}' '#{window_width}' '#{window_height}'"

tmux_args=(
  new-session
  -c "$PWD"
  "${tmux_env_args[@]}"
)

if [[ -n "$session_name" ]]; then
  session_name_args=(-s "$session_name")
  tmux_args+=("${session_name_args[@]}")
fi

for (( pane_index = 1; pane_index < pane_count; pane_index++ )); do
  tmux_args+=(
    ';'
    split-window -d -h -c "$PWD"
  )
done

tmux_args+=(
  ';'
  run-shell "$apply_layout_command"
  ';'
  select-pane -t 0
)

if (( create_detached_session )); then
  target_session=$(tmux new-session -dP -F '#{session_name}' -c "$PWD" "${tmux_env_args[@]}" "${session_name_args[@]}")

  for (( pane_index = 1; pane_index < pane_count; pane_index++ )); do
    tmux split-window -d -h -t "${target_session}:0" -c "$PWD"
  done

  window_id=$(tmux display-message -p -t "${target_session}:0" '#{window_id}')
  window_width=$(tmux display-message -p -t "${target_session}:0" '#{window_width}')
  window_height=$(tmux display-message -p -t "${target_session}:0" '#{window_height}')
  "$script_dir/tmux-apply-layout.sh" "$pane_count" "$window_id" "$window_width" "$window_height"
  tmux select-pane -t "${target_session}:0.0"
  if (( switch_to_session )); then
    tmux switch-client -t "$target_session"
    exit $?
  fi
  print "Detached tmux session $target_session created"
  exit 0
fi

tmux_command=(tmux "${tmux_args[@]}")
"${tmux_command[@]}"
