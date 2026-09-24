"""The SRT sandbox backend: settings resolution, merging, and project-write patching."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from agm.config.sandbox.srt import (
    JsonDict,
    load_settings,
    merge_settings_chain,
    patch_for_proj_dir,
    sandbox_settings_candidates,
    track_bwrap_artifacts,
)
from agm.core.fs import is_file
from agm.core.path import display_path
from agm.sandbox.backend import ResolvedSettings, SandboxSettingsError, SandboxUnavailableError
from agm.sandbox.request import DRY_RUN_SETTINGS_PLACEHOLDER, SandboxRequest, cleanup_artifacts


def _write_json_temp(data: JsonDict, temp_files: list[Path]) -> Path:
    with NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        json.dump(data, handle)
        path = Path(handle.name)
    temp_files.append(path)
    return path


def _load_settings_or_raise(path: Path) -> JsonDict:
    """`load_settings`, wrapped so a malformed or unreadable file raises SandboxSettingsError."""

    try:
        return load_settings(path)
    except (OSError, json.JSONDecodeError) as error:
        raise SandboxSettingsError(f"failed to load sandbox settings {path}: {error}") from error


class SrtBackend:
    """The SRT sandbox-runtime backend."""

    def require_available(self, env: dict[str, str]) -> None:
        if shutil.which("srt", path=env.get("PATH")) is not None:
            return
        raise SandboxUnavailableError(
            "srt is not installed or not in PATH.",
            detail="Install it with: npm install -g @anthropic-ai/sandbox-runtime",
        )

    def settings_candidates(self, request: SandboxRequest) -> list[Path]:
        return sandbox_settings_candidates(
            cwd=request.config_cwd,
            home=request.home,
            proj_dir=request.proj_dir,
            command_name=request.spec.profile_name,
            alias_command_name=request.alias_name,
            env=request.env,
        )

    def resolve_settings(self, request: SandboxRequest) -> ResolvedSettings:
        spec = request.spec
        temp_files: list[Path] = []
        try:
            data: JsonDict | None = None
            write_settings = False
            if spec.settings_file is not None:
                selected = spec.settings_file
                if not selected.is_absolute():
                    selected = request.config_cwd / selected
                if not is_file(selected):
                    raise SandboxSettingsError(
                        f"settings file not found: "
                        f"{display_path(selected, cwd=request.config_cwd)}",
                        path=selected,
                    )
            else:
                candidates = self.settings_candidates(request)
                found = [path for path in candidates if is_file(path)]
                if not found:
                    raise SandboxSettingsError(
                        "no sandbox settings file found. Checked: "
                        + ", ".join(
                            display_path(path, cwd=request.config_cwd) for path in candidates
                        ),
                        candidates=tuple(candidates),
                    )
                selected = found[0]
                if len(found) > 1:
                    settings_data = [_load_settings_or_raise(path) for path in found]
                    data = merge_settings_chain(settings_data)
                    write_settings = True

            if spec.patch and request.proj_dir is not None:
                selected_data = data if data is not None else _load_settings_or_raise(selected)
                data = patch_for_proj_dir(selected_data, request.proj_dir)
                write_settings = True

            if data is None:
                data = _load_settings_or_raise(selected)
            if write_settings:
                selected = _write_json_temp(data, temp_files)

            tracked_artifacts = track_bwrap_artifacts(data, request.cwd)
        except OSError as error:
            # A temp-settings write failure (disk full, permissions) is a
            # reachable settings-resolution failure, not a host crash.
            cleanup_artifacts(temp_files, [])
            raise SandboxSettingsError(f"failed to write sandbox settings: {error}") from error
        except Exception:
            # Clean up a merge/patch temp file already written before the
            # failure, so a partial preparation never leaks it.
            cleanup_artifacts(temp_files, [])
            raise

        return ResolvedSettings(
            path=selected,
            temp_files=tuple(temp_files),
            tracked_artifacts=tuple(tracked_artifacts),
        )

    def wrap(self, request: SandboxRequest, resolved: ResolvedSettings) -> list[str]:
        return ["srt", "--settings", str(resolved.path), "--"]

    def dry_run_wrap(self, request: SandboxRequest) -> list[str]:
        return ["srt", "--settings", DRY_RUN_SETTINGS_PLACEHOLDER, "--"]

    def prepare_env(self, request: SandboxRequest, env: dict[str, str]) -> dict[str, str]:
        # Node.js's built-in fetch() does not respect HTTP_PROXY/HTTPS_PROXY
        # environment variables by default (unlike curl, wget, etc.). Inside
        # the SRT sandbox all DNS resolution must go through the proxy bridges
        # (bwrap --unshare-net removes network access), so without this flag
        # any Node.js process using globalThis.fetch() will fail with
        # getaddrinfo EAI_AGAIN. NODE_USE_ENV_PROXY=1 tells Node.js to honour
        # the proxy env vars that SRT injects via bwrap --setenv. This is a
        # no-op outside the sandbox (no proxy vars are set) and safe to always
        # include when running under SRT.
        resolved_env = dict(env)
        resolved_env.setdefault("NODE_USE_ENV_PROXY", "1")
        return resolved_env


SRT_BACKEND = SrtBackend()
