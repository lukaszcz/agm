#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sandbox.sh [--no-patch] [-f settings.json] <command> [args...]

Runs a command inside Anthropic Sandbox Runtime (`srt`).

If `-f` is not provided, settings are loaded from:
  1. ~/.sandbox/default.json
  2. ./.sandbox/default.json

If both files exist, the project-local file overrides the home file with
section-aware merging.

If `PROJ_DIR` is set, the selected settings file is patched temporarily to add
`$PROJ_DIR` to `filesystem.allowWrite`.
Use `--no-patch` to disable that behavior.
EOF
}

SETTINGS_FILE=""
PATCH_SETTINGS=1
TEMP_SETTINGS_FILES=()
TRACKED_BWRAP_ARTIFACTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f)
      if [[ $# -lt 2 ]]; then
        echo "Error: -f requires an argument." >&2
        usage >&2
        exit 1
      fi
      SETTINGS_FILE="$2"
      shift 2
      ;;
    -h)
      usage
      exit 0
      ;;
    --no-patch)
      PATCH_SETTINGS=0
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Error: invalid option $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Error: command is required." >&2
  usage >&2
  exit 1
fi

if ! command -v srt >/dev/null 2>&1; then
  echo "Error: srt is not installed or not in PATH." >&2
  echo "Install it with: npm install -g @anthropic-ai/sandbox-runtime" >&2
  exit 1
fi

cleanup() {
  for temp_file in "${TEMP_SETTINGS_FILES[@]}"; do
    [[ -f "$temp_file" ]] && rm -f "$temp_file"
  done

  for artifact in "${TRACKED_BWRAP_ARTIFACTS[@]}"; do
    if [[ -f "$artifact" && ! -s "$artifact" ]]; then
      rm -f "$artifact"
    elif [[ -d "$artifact" ]]; then
      rmdir "$artifact" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT

require_python3() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 >= 3.12 is required to process sandbox settings files." >&2
    exit 1
  fi

  if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
    echo "Error: python3 >= 3.12 is required to process sandbox settings files." >&2
    exit 1
  fi
}

track_bwrap_artifacts() {
  require_python3

  mapfile -t TRACKED_BWRAP_ARTIFACTS < <(
    python3 - "$SETTINGS_FILE" "$PWD" <<'PY'
import json
import os
import sys

settings_path, cwd = sys.argv[1:]

with open(settings_path, "r", encoding="utf-8") as f:
    data = json.load(f)

mandatory_deny_paths = [
    ".gitconfig",
    ".gitmodules",
    ".bashrc",
    ".bash_profile",
    ".zshrc",
    ".zprofile",
    ".profile",
    ".ripgreprc",
    ".mcp.json",
    ".vscode",
    ".idea",
    ".claude/commands",
    ".claude/agents",
]

git_dir = os.path.join(cwd, ".git")
if os.path.isdir(git_dir):
    mandatory_deny_paths.extend([".git/hooks", ".git/config"])

filesystem = data.get("filesystem")
deny_write = []
if isinstance(filesystem, dict):
    raw_deny_write = filesystem.get("denyWrite")
    if isinstance(raw_deny_write, list):
        deny_write = [
            entry
            for entry in raw_deny_write
            if isinstance(entry, str) and not any(ch in entry for ch in "*?[]")
        ]

def normalize(path_pattern: str) -> str:
    if path_pattern == "~":
        normalized = os.path.expanduser("~")
    elif path_pattern.startswith("~/"):
        normalized = os.path.expanduser(path_pattern)
    elif os.path.isabs(path_pattern):
        normalized = path_pattern
    else:
        normalized = os.path.join(cwd, path_pattern)

    normalized = os.path.abspath(normalized)
    if os.path.exists(normalized):
        try:
            normalized = os.path.realpath(normalized)
        except OSError:
            pass
    return normalized

def first_missing_component(target: str) -> str | None:
    relpath = os.path.relpath(target, cwd)
    if relpath == "." or relpath.startswith(".."):
        return None

    current = cwd
    for part in relpath.split(os.sep):
        current = os.path.join(current, part)
        if not os.path.exists(current):
            return current
    return None

seen = set()
for candidate in mandatory_deny_paths + deny_write:
    normalized = normalize(candidate)
    missing = first_missing_component(normalized)
    if missing is None or missing in seen:
        continue
    seen.add(missing)
    print(missing)
PY
  )
}

merge_settings_files() {
  local home_settings=$1
  local local_settings=$2
  local merged_settings

  require_python3

  merged_settings="$(mktemp)"
  TEMP_SETTINGS_FILES+=("$merged_settings")

  python3 - "$home_settings" "$local_settings" "$merged_settings" <<'PY'
import json
import sys

home_path, local_path, out_path = sys.argv[1:]

with open(home_path, "r", encoding="utf-8") as f:
    home_data = json.load(f)

with open(local_path, "r", encoding="utf-8") as f:
    local_data = json.load(f)

merged = dict(home_data)

if "enabled" in local_data and local_data["enabled"] is not None:
    merged["enabled"] = local_data["enabled"]

if isinstance(local_data.get("network"), dict):
    merged["network"] = {
        **(home_data.get("network") if isinstance(home_data.get("network"), dict) else {}),
        **local_data["network"],
    }

if isinstance(local_data.get("filesystem"), dict):
    merged["filesystem"] = {
        **(home_data.get("filesystem") if isinstance(home_data.get("filesystem"), dict) else {}),
        **local_data["filesystem"],
    }

if isinstance(local_data.get("ignoreViolations"), dict):
    merged["ignoreViolations"] = local_data["ignoreViolations"]

if "enableWeakerNestedSandbox" in local_data and local_data["enableWeakerNestedSandbox"] is not None:
    merged["enableWeakerNestedSandbox"] = local_data["enableWeakerNestedSandbox"]

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(merged, f)
PY

  SETTINGS_FILE="$merged_settings"
}

patch_settings_for_proj_dir() {
  local source_settings=$1
  local patched_settings

  require_python3

  patched_settings="$(mktemp)"
  TEMP_SETTINGS_FILES+=("$patched_settings")

  python3 - "$source_settings" "$patched_settings" "$PROJ_DIR" <<'PY'
import json
import os
import sys

source_path, out_path, proj_dir = sys.argv[1:]

with open(source_path, "r", encoding="utf-8") as f:
    data = json.load(f)

filesystem = data.get("filesystem")
if not isinstance(filesystem, dict):
    filesystem = {}
    data["filesystem"] = filesystem

allow_write = filesystem.get("allowWrite")
if not isinstance(allow_write, list):
    allow_write = []
    filesystem["allowWrite"] = allow_write

if proj_dir not in allow_write:
    allow_write.append(proj_dir)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f)
PY

  SETTINGS_FILE="$patched_settings"
}

if [[ -n "$SETTINGS_FILE" ]]; then
  if [[ ! -f "$SETTINGS_FILE" ]]; then
    echo "Error: settings file not found: $SETTINGS_FILE" >&2
    exit 1
  fi
else
  HOME_SETTINGS="$HOME/.sandbox/default.json"
  LOCAL_SETTINGS="$PWD/.sandbox/default.json"
  FOUND_SETTINGS=()

  [[ -f "$HOME_SETTINGS" ]] && FOUND_SETTINGS+=("$HOME_SETTINGS")
  [[ -f "$LOCAL_SETTINGS" ]] && FOUND_SETTINGS+=("$LOCAL_SETTINGS")

  if [[ ${#FOUND_SETTINGS[@]} -eq 0 ]]; then
    echo "Error: no sandbox settings file found." >&2
    echo "Checked: $HOME_SETTINGS and $LOCAL_SETTINGS" >&2
    exit 1
  fi

  if [[ ${#FOUND_SETTINGS[@]} -eq 1 ]]; then
    SETTINGS_FILE="${FOUND_SETTINGS[0]}"
  else
    merge_settings_files "$HOME_SETTINGS" "$LOCAL_SETTINGS"
  fi
fi

if [[ $PATCH_SETTINGS -eq 1 && -n "${PROJ_DIR:-}" ]]; then
  patch_settings_for_proj_dir "$SETTINGS_FILE"
fi

track_bwrap_artifacts

set +e
srt --settings "$SETTINGS_FILE" -- "$@"
status=$?
set -e

exit "$status"
