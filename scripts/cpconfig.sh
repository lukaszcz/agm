#!/usr/bin/env bash

set -euo pipefail

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

PROJ_DIR=$(project_dir)
while getopts "d:" opt; do
    case $opt in
        d) PROJ_DIR="$OPTARG" ;;
        *) echo "usage: $(basename "$0") [-d project-dir] dirname"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -ne 1 ]; then
    echo "usage: $(basename "$0") [-d project-dir] dirname"
    exit 1
fi

DIRNAME="$1"
if [[ "$DIRNAME" != /* ]]; then
    DIRNAME="$PWD/$DIRNAME"
fi

FILES=(.setup.sh .env .env.local .config .agents .opencode .codex .claude .pi .mcp.json)

cp -r "${FILES[@]}" "$DIRNAME" 2>/dev/null || true

if [ -d "$PROJ_DIR/config" ]; then
    (cd "$PROJ_DIR/config" && cp -r "${FILES[@]}" "$DIRNAME" 2>/dev/null || true)
fi
