"""Reserved structural names in the AGM TOML config schema.

A pure data leaf (no ``agm`` imports) so any layer — including the AgL
semantics leaf that guards ``program NAME`` declarations — can depend on it
without pulling in the config-loading machinery.
"""

from __future__ import annotations

# Structural config-section / sub-table names reserved by the config schema
# rather than being workflow-command overrides: ``[exec.agents]`` holds the
# per-agent runner map and ``params`` addresses program param tables.  These
# own a fixed ``[NAME]`` section, so a ``program NAME`` declaration may not
# reuse them.
RESERVED_CONFIG_SECTIONS: frozenset[str] = frozenset({"agents", "params"})
