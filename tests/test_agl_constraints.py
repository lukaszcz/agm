"""Tests for the structural constraint-kind vocabulary in `agm.agl.constraints`."""

from __future__ import annotations

from agm.agl.constraints import CONSTRAINT_SPELLINGS, ConstraintKind, close_constraints


def test_constraint_spellings_is_a_closed_set() -> None:
    assert CONSTRAINT_SPELLINGS == {"Eq": ConstraintKind.EQ, "Hashable": ConstraintKind.HASHABLE}


def test_close_constraints_hashable_implies_eq() -> None:
    assert close_constraints(frozenset({ConstraintKind.HASHABLE})) == frozenset(
        {ConstraintKind.HASHABLE, ConstraintKind.EQ}
    )


def test_close_constraints_eq_alone_stays_eq() -> None:
    assert close_constraints(frozenset({ConstraintKind.EQ})) == frozenset({ConstraintKind.EQ})


def test_close_constraints_empty_stays_empty() -> None:
    assert close_constraints(frozenset()) == frozenset()
