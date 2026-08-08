"""Reserved structural names in the AGM TOML config schema.

A pure data leaf (no ``agm`` imports) so any layer — including the AgL
semantics leaf that guards ``program NAME`` declarations — can depend on it
without pulling in the config-loading machinery.
"""

from __future__ import annotations

# ``params`` addresses program parameter tables and owns a fixed ``[NAME]``
# section, so a ``program NAME`` declaration may not reuse it.
RESERVED_CONFIG_SECTIONS: frozenset[str] = frozenset({"params"})
