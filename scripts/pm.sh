#!/usr/bin/env zsh

set -euo pipefail

export PROJ_DIR="$PWD"
PROJ=$(basename $PROJ_DIR)

usage() {
    echo "usage: $(basename "$0") open [-n pane_count] [branch]"
    echo "       $(basename "$0") new [-n pane_count] [-p parent] branch"
    echo "       $(basename "$0") {co|checkout} [-n pane_count] [-p parent] branch"
    exit 1
}

source_env_file() {
    local env_file="$1"

    [[ -f "$env_file" ]] || return 0
    source "$env_file"
}

CMD="${1-}"
[[ -z "$CMD" ]] && usage
shift

PANE_COUNT=""

case "$CMD" in
    open)
        while getopts "n:" opt; do
            case $opt in
                n) PANE_COUNT="$OPTARG" ;;
                *) usage ;;
            esac
        done
        shift $((OPTIND - 1))

        BRANCH="${1-}"
        ;;
    new|co|checkout)
        PARENT=""
        while getopts "n:p:" opt; do
            case $opt in
                n) PANE_COUNT="$OPTARG" ;;
                p) PARENT="$OPTARG" ;;
                *) usage ;;
            esac
        done
        shift $((OPTIND - 1))

        BRANCH="${1-}"
        if [[ -z "$BRANCH" ]]; then
            echo "error: $CMD requires a branch name" >&2
            exit 1
        fi
        ;;
    *)
        usage
        ;;
esac

if [[ -n "$PANE_COUNT" ]] && [[ ! "$PANE_COUNT" =~ '^[1-9][0-9]*$' ]]; then
    echo "error: pane count must be a positive integer" >&2
    exit 1
fi

if [[ -n "$BRANCH" ]]; then
    SESSION_NAME="$PROJ/$BRANCH"
    REPO="$PROJ_DIR/worktrees/$BRANCH"
else
    SESSION_NAME="$PROJ"
    REPO="$PROJ_DIR/repo"
fi

mkdir -p "$REPO"
cd "$REPO"

PROJ_ENV="$PROJ_DIR/config/env.sh"
source_env_file "$PROJ_ENV"

SESSION_ENV="$PROJ_DIR/config/$BRANCH/env.sh"
source_env_file "$SESSION_ENV"

case "$CMD" in
    new|co|checkout)
        REPO_BRANCH=$(git -C "$PROJ_DIR/repo" rev-parse --abbrev-ref HEAD)
        PARENT="${PARENT:-$REPO_BRANCH}"

        if [[ "$PARENT" == "$REPO_BRANCH" ]]; then
            cd "$PROJ_DIR/repo"
        else
            cd "$PROJ_DIR/worktrees/$PARENT"
        fi

        if [[ "$CMD" == "new" ]]; then
            mkwt.sh -b "$BRANCH"
        else
            mkwt.sh "$BRANCH"
        fi
        ;;
esac

cd "$REPO"

tmux_args=()

if [[ -n "$PANE_COUNT" ]]; then
    tmux_args+=(-n "$PANE_COUNT")
fi

if [[ "$CMD" == "new" ]]; then
    tmux_args+=(-d)
fi

tmux.sh "${tmux_args[@]}" "$SESSION_NAME"
