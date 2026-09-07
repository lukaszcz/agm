"""Shared errors for package store operations."""


class PackageInstallError(ValueError):
    """Raised when package installation or removal cannot safely proceed."""


class DisciplineError(ValueError):
    """Raised when a package directory violates package discipline."""


class FetchError(ValueError):
    """Raised when a package archive cannot be fetched or verified."""
