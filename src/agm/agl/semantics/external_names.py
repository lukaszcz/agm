"""The external spellings a field or record declaration carries."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["NO_EXTERNAL_NAME", "ExternalName"]


@dataclass(frozen=True, slots=True)
class ExternalName:
    """The ``@name`` and ``@json-name`` arguments of one field or record declaration.

    ``name`` is an alternative value-syntax spelling and the JSON default;
    ``json_name`` overrides the JSON spelling only.
    """

    name: str | None = None
    json_name: str | None = None

    def json(self, declared: str) -> str:
        """Return the JSON spelling of a declaration named *declared*."""
        if self.json_name is not None:
            return self.json_name
        return declared if self.name is None else self.name

    def value_names(self, declared: str) -> tuple[str, ...]:
        """Return every value-syntax spelling of a declaration named *declared*."""
        if self.name is None or self.name == declared:
            return (declared,)
        return (declared, self.name)

    def alias(self, declared: str) -> str | None:
        """Return the ``@name`` spelling if it differs from *declared*, else ``None``.

        The single rule for a decode descriptor's optional alias field:
        derived from :meth:`value_names` so every caller (record, enum member,
        and field descriptors alike) computes it the same way.
        """
        names = self.value_names(declared)
        return names[-1] if len(names) > 1 else None


#: The external spellings of a declaration carrying neither attribute.
NO_EXTERNAL_NAME = ExternalName()
