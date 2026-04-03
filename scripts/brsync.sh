#!/usr/bin/env bash

set -euo pipefail

# Creates local tracking branches for remote branches not merged
# into `origin/main`. Fetches and prunes remotes, then for each remote
# branch (except `origin/HEAD`) creates a local branch tracking that
# remote if the local branch does not already exist.

git fetch --prune origin
for branch in $(git for-each-ref --format='%(refname:short)' --no-merged=origin/main refs/remotes/origin); do
  if [[ "$branch" == "origin/HEAD" ]]; then
    continue
  fi
  local_branch="${branch#origin/}"
  if ! git show-ref --verify --quiet "refs/heads/$local_branch"; then
    git branch --track "$local_branch" "$branch"
  fi
done
