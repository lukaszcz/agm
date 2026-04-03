#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "usage: $(basename "$0") [-b branch] [project-name] [repo-url]" >&2
    echo "       $(basename "$0") [-b branch] repo-url" >&2
    exit 1
}

looks_like_repo_url() {
    local value=${1-}

    [[ "$value" == *"://"* ]] || [[ "$value" == git@*:* ]] || [[ "$value" == *github.com[:/]* ]] || [[ "$value" == *.git ]]
}

derive_project_name() {
    local repo_url=$1
    local trimmed=${repo_url%/}
    local name

    name=$(basename "$trimmed")
    name=${name%.git}

    if [[ -z "$name" || "$name" == "." || "$name" == "/" ]]; then
        echo "error: could not derive project name from repo url: $repo_url" >&2
        exit 1
    fi

    printf '%s\n' "$name"
}

write_file_if_missing() {
    local path=$1
    local content=$2

    if [[ -e "$path" ]]; then
        return
    fi

    printf '%s\n' "$content" >"$path"
}

BRANCH=""
while getopts "b:" opt; do
    case "$opt" in
        b) BRANCH="$OPTARG" ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

if [[ $# -eq 0 || $# -gt 2 ]]; then
    usage
fi

PROJ=""
REPO_URL=""

case $# in
    1)
        if looks_like_repo_url "$1"; then
            REPO_URL=$1
        else
            PROJ=$1
        fi
        ;;
    2)
        PROJ=$1
        REPO_URL=$2
        ;;
esac

if [[ -z "$PROJ" && -z "$REPO_URL" ]]; then
    usage
fi

if [[ -z "$PROJ" ]]; then
    PROJ=$(derive_project_name "$REPO_URL")
fi

mkdir -p "$PROJ"/repo "$PROJ"/deps "$PROJ"/worktrees "$PROJ"/notes "$PROJ"/config

write_file_if_missing "$PROJ/config/env.sh" "# Set project-level environment variables here."
write_file_if_missing "$PROJ/config/setup.sh" "# Initialize a newly created worktree here."

chmod +x "$PROJ/config/setup.sh"

if [[ -n "$REPO_URL" ]]; then
    if [[ -n "$(find "$PROJ/repo" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "error: $PROJ/repo already exists and is not empty" >&2
        exit 1
    fi

    clone_args=()
    if [[ -n "$BRANCH" ]]; then
        clone_args+=(--branch "$BRANCH")
    fi

    git clone "${clone_args[@]}" "$REPO_URL" "$PROJ/repo"
fi
