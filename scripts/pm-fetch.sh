#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR=$PWD

first_git_worktree() {
    local parent_dir=$1
    local path

    while IFS= read -r path; do
        if [[ -d "$path" ]] && git -C "$path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            printf '%s\n' "$path"
            return 0
        fi
    done < <(find "$parent_dir" -mindepth 1 -type d -print | sort)

    echo "error: $parent_dir must contain at least one checked out branch" >&2
    exit 1
}

fetch_repo() {
    local repo_path=$1
    local display_path=${repo_path#"$PROJECT_DIR"/}

    echo "Fetching $display_path"
    git -C "$repo_path" fetch
}

main() {
    local dep_dir repo_path

    if [[ ! -d "$PROJECT_DIR/repo" ]]; then
        echo "error: repo does not exist in $PROJECT_DIR" >&2
        exit 1
    fi

    fetch_repo "$PROJECT_DIR/repo"

    if [[ ! -d "$PROJECT_DIR/deps" ]]; then
        exit 0
    fi

    while IFS= read -r dep_dir; do
        repo_path=$(first_git_worktree "$dep_dir")
        fetch_repo "$repo_path"
    done < <(find "$PROJECT_DIR/deps" -mindepth 1 -maxdepth 1 -type d -print | sort)
}

main "$@"
