"""TypeSafe System One questions for ``sysone/jev``, over the official ``typesafe-sdk``."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from typesafe_sdk import RetryPolicy, TypeSafeClient

if TYPE_CHECKING:
    import httpx2
    from agl import TypeContract


@dataclass(frozen=True)
class Settings:
    """The pool key of one client; a ``None`` field leaves the SDK default."""

    api_key: str | None
    base_url: str | None
    max_retries: int


def open_client(
    settings: Settings, *, transport: httpx2.BaseTransport | None = None
) -> TypeSafeClient:
    """Build the SDK client for *settings*; tests replace this seam to inject *transport*."""
    return TypeSafeClient(
        api_key=settings.api_key,
        base_url=settings.base_url,
        retry=RetryPolicy(max_retries=settings.max_retries),
        transport=transport,
    )


def system_one(
    state: object,
    questions: object,
    model: str | None,
    timeout: str | None,
    api_key: str | None,
    base_url: str | None,
    max_retries: int,
) -> object:
    """Ask *questions* about *state* in one request."""
    raise NotImplementedError("sysone/jev::system-one")


def ask_noul(
    instructions: str,
    state: object,
    criteria: object,
    model: str | None,
    timeout: str | None,
    api_key: str | None,
    base_url: str | None,
    max_retries: int,
) -> object:
    """Ask one noul question."""
    raise NotImplementedError("sysone/jev::ask-noul")


def ask_choice(
    target: TypeContract,
    instructions: str,
    state: object,
    model: str | None,
    timeout: str | None,
    api_key: str | None,
    base_url: str | None,
    max_retries: int,
) -> object:
    """Ask one choice question over the members of *target*'s choice type."""
    raise NotImplementedError("sysone/jev::ask-choice")


def ask_score(
    target: TypeContract,
    instructions: str,
    state: object,
    model: str | None,
    timeout: str | None,
    api_key: str | None,
    base_url: str | None,
    max_retries: int,
) -> object:
    """Ask one score question over the rubric of *target*'s level type."""
    raise NotImplementedError("sysone/jev::ask-score")


def ask(
    target: TypeContract,
    instructions: str,
    state: object,
    model: str | None,
    timeout: str | None,
    api_key: str | None,
    base_url: str | None,
    max_retries: int,
    noul_threshold: Decimal,
) -> object:
    """Ask one question whose kind *target* selects."""
    raise NotImplementedError("sysone/jev::ask")


def ask_many(
    target: TypeContract,
    state: object,
    model: str | None,
    timeout: str | None,
    api_key: str | None,
    base_url: str | None,
    max_retries: int,
    noul_threshold: Decimal,
) -> object:
    """Ask one question per field of the record *target* in one request."""
    raise NotImplementedError("sysone/jev::ask-many")
