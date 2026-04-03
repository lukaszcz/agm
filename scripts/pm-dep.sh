#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR=$PWD

usage() {
    echo "usage: $(basename "$0") new [-b branch] repo-url" >&2
    echo "       $(basename "$0") switch dep [-b] branch" >&2
    exit 1
}

derive_dep_name() {
    local repo_url=$1
    local trimmed=${repo_url%/}
    local dep

    dep=$(basename "$trimmed")
    dep=${dep%.git}

    if [[ -z "$dep" || "$dep" == "." || "$dep" == "/" ]]; then
        echo "error: could not derive dependency name from repo url: $repo_url" >&2
        exit 1
    fi

    printf '%s\n' "$dep"
}

default_branch_from_remote() {
    local repo_url=$1
    local branch

    branch=$(git ls-remote --symref "$repo_url" HEAD 2>/dev/null | awk '/^ref:/ { sub("refs/heads/", "", $2); print $2; exit }')

    if [[ -z "$branch" ]]; then
        echo "error: could not determine default branch for $repo_url" >&2
        exit 1
    fi

    printf '%s\n' "$branch"
}

default_branch_from_repo() {
    local repo_path=$1
    local branch

    branch=$(git -C "$repo_path" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null || true)
    branch=${branch#origin/}

    if [[ -z "$branch" ]]; then
        echo "error: could not determine default branch for dependency repo at $repo_path" >&2
        exit 1
    fi

    printf '%s\n' "$branch"
}

first_dep_repo() {
    local dep_dir=$1
    local path
    local repo_path=""

    while IFS= read -r path; do
        if [[ -d "$path" ]] && git -C "$path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            repo_path=$path
            break
        fi
    done < <(find "$dep_dir" -mindepth 1 -type d -print | sort)

    if [[ -n "$repo_path" ]]; then
        printf '%s\n' "$repo_path"
        return 0
    fi

    echo "error: $dep_dir must contain at least one checked out dependency branch" >&2
    exit 1
}

fetch_dep_refs() {
    local repo_path=$1

    git -C "$repo_path" fetch
}

cmd_new() {
    local branch=""
    local repo_url dep dep_dir target_dir

    OPTIND=1
    while getopts ":b:" opt; do
        case "$opt" in
            b) branch=$OPTARG ;;
            *) usage ;;
        esac
    done
    shift $((OPTIND - 1))

    if [[ $# -ne 1 ]]; then
        usage
    fi

    repo_url=$1
    dep=$(derive_dep_name "$repo_url")
    dep_dir="$PROJECT_DIR/deps/$dep"

    if [[ -e "$dep_dir" ]]; then
        echo "error: deps/$dep already exists" >&2
        exit 1
    fi

    if [[ -z "$branch" ]]; then
        branch=$(default_branch_from_remote "$repo_url")
    fi

    target_dir="$dep_dir/$branch"
    mkdir -p "$(dirname "$target_dir")"
    if ! git clone --branch "$branch" "$repo_url" "$target_dir"; then
        rmdir "$dep_dir" 2>/dev/null || true
        exit 1
    fi
}

cmd_switch() {
    local create_branch=false
    local branch dep dep_dir repo_path target_dir default_branch

    if [[ $# -eq 2 ]]; then
        dep=$1
        branch=$2
    elif [[ $# -eq 3 && "$2" == "-b" ]]; then
        dep=$1
        create_branch=true
        branch=$3
    else
        usage
    fi
    dep_dir="$PROJECT_DIR/deps/$dep"

    if [[ ! -d "$dep_dir" ]]; then
        echo "error: deps/$dep does not exist" >&2
        exit 1
    fi

    repo_path=$(first_dep_repo "$dep_dir")
    target_dir="$dep_dir/$branch"

    if [[ -e "$target_dir" ]]; then
        echo "error: deps/$dep/$branch already exists" >&2
        exit 1
    fi

    mkdir -p "$(dirname "$target_dir")"
    fetch_dep_refs "$repo_path"

    if $create_branch; then
        default_branch=$(default_branch_from_repo "$repo_path")
        git -C "$repo_path" worktree add -b "$branch" "$target_dir" "$default_branch"
    else
        git -C "$repo_path" worktree add "$target_dir" "$branch"
    fi
}

cmd=${1-}
if [[ -z "$cmd" ]]; then
    usage
fi
shift

case "$cmd" in
    new) cmd_new "$@" ;;
    switch) cmd_switch "$@" ;;
    *) usage ;;
esac
