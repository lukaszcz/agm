"""Host-supplied AgL source overrides for engine-setting defaults.

:class:`SettingOverride` pairs an AgL expression's source text with a short
label identifying where it came from (a CLI flag, a config key, ...), so a
diagnostic produced while compiling the override can name its origin. It is a
pure data leaf with no other ``agm`` imports, consumed by
``PipelineDriver.prepare_program``'s ``setting_overrides`` parameter (see
``agm.agl.pipeline``).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SettingOverride"]


@dataclass(frozen=True, slots=True)
class SettingOverride:
    """One engine-setting default supplied as AgL source text.

    ``source`` is parsed as a single AgL expression and spliced in as the
    named ``builtin var`` declaration's default in the loaded ``std/config``
    module, so it is resolved, type-checked, and constant-checked by the
    program's own pipeline rather than a separate one.

    ``origin`` is a short, human-readable label for where the override came
    from (e.g. ``"--agent"``, ``"[exec] default-agent"``), used to name the
    source in diagnostics produced while compiling it.

    ``required`` records the override's PROVENANCE, not its content: it is
    ``True`` for an explicit per-run request (a CLI flag such as ``--agent``)
    that the host cannot silently drop, and ``False`` for a value sourced from
    ambient configuration (e.g. ``[exec] default-agent``) that is simply
    inert when the target ``std/config`` module never loads. A graph with no
    loaded ``std/config`` still produces a diagnostic for a ``required``
    override, naming its origin; a non-``required`` override is skipped
    silently in that case instead. Defaults to ``True`` (a request), since
    that was every override's behavior before this field existed.
    """

    source: str
    origin: str
    required: bool = True
