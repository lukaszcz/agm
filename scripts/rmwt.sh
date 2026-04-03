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

FORCE=false

while getopts "f" opt; do
    case $opt in
        f) FORCE=true ;;
        *) echo "usage: $(basename "$0") [-f] <branch-name>"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -ne 1 ]; then
    echo "usage: $(basename "$0") [-f] <branch-name>"
    exit 1
fi

BRANCH="$1"

git_setup

# Find the worktree path for the given branch using git worktree list
WORKTREE_PATH=$("${GIT[@]}" worktree list --porcelain | awk -v branch="$BRANCH" '
    /^worktree / { path = substr($0, 10) }
    /^branch / && $2 == "refs/heads/" branch { print path; exit }
')

if [ -z "$WORKTREE_PATH" ]; then
    echo "Error: No worktree found for branch '$BRANCH'"
    echo "Available worktrees:"
    "${GIT[@]}" worktree list
    exit 1
fi

if $FORCE; then
    "${GIT[@]}" worktree remove --force "$WORKTREE_PATH"
else
    "${GIT[@]}" worktree remove "$WORKTREE_PATH"
fi

echo "Removed worktree for branch '$BRANCH': $WORKTREE_PATH"

"${GIT[@]}" branch -d "$BRANCH"
