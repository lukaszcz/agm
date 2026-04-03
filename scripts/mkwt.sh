#!/usr/bin/env bash

set -euo pipefail

git_setup() {
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        GIT=(git)
    elif [ -d repo ] && git -C repo rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        GIT=(git -C repo)
    else
        echo "Error: current directory is not a git repository and repo/ is not a git repository." >&2
        exit 1
    fi
}

fetch_refs() {
    "${GIT[@]}" fetch
}

project_dir() {
    local current_dir parent_dir grandparent_dir

    current_dir=$(basename "$PWD")
    parent_dir=$(basename "$(dirname "$PWD")")
    grandparent_dir=$(dirname "$(dirname "$PWD")")

    if [ "$current_dir" = "repo" ] && [ -d "$(dirname "$PWD")/worktrees" ]; then
        printf '%s\n' "$(dirname "$PWD")"
    elif [ "$parent_dir" = "worktrees" ] && [ -d "$grandparent_dir/repo" ]; then
        printf '%s\n' "$grandparent_dir"
    elif [ -d "$PWD/repo" ]; then
        printf '%s\n' "$PWD"
    else
        printf '%s\n' "$PWD"
    fi
}

default_worktrees_dir() {
    local proj_dir
    proj_dir=$(project_dir)

    if [ -d "$proj_dir/worktrees" ]; then
        printf '%s\n' "$proj_dir/worktrees"
    else
        printf '%s\n' "$proj_dir/.worktrees"
    fi
}

CREATE_BRANCH=false
WORKTREES_DIR=$(default_worktrees_dir)
while getopts "b:d:" opt; do
    case $opt in
        b) CREATE_BRANCH=true; BRANCH="$OPTARG" ;;
        d) WORKTREES_DIR="$OPTARG" ;;
        *) echo "usage: $(basename "$0") [-b branch-name] [-d dir] [branch-name]"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

if $CREATE_BRANCH; then
    if [ $# -ne 0 ]; then
        echo "usage: $(basename "$0") [-b branch-name] [-d dir] [branch-name]"
        exit 1
    fi
else
    if [ $# -ne 1 ]; then
        echo "usage: $(basename "$0") [-b branch-name] [-d dir] [branch-name]"
        exit 1
    fi
    BRANCH="$1"
fi

git_setup
fetch_refs

WORKTREES_PATH="${WORKTREES_DIR%/}"
if [[ "$WORKTREES_PATH" != /* ]]; then
    WORKTREES_PATH="$PWD/$WORKTREES_PATH"
fi

BASEDIR=$(basename "$("${GIT[@]}" worktree list --porcelain | sed -n 's/^worktree //p' | head -1)")
DIRNAME="${WORKTREES_PATH%/}/$BRANCH"

if $CREATE_BRANCH; then
    "${GIT[@]}" worktree add -b "$BRANCH" "$DIRNAME"
else
    "${GIT[@]}" worktree add "$DIRNAME" "$BRANCH"
fi
cpconfig.sh "$DIRNAME"

cd $DIRNAME

PROJ_DIR=$(project_dir)

if [[ -x "$PROJ_DIR/config/setup.sh" ]]; then
    "$PROJ_DIR/config/setup.sh"
fi

if [ -f .config/setup.sh ] && [ -x .config/setup.sh ]; then
    ./.config/setup.sh
fi

if [ -f .setup.sh ] && [ -x .setup.sh ]; then
    ./.setup.sh
fi
