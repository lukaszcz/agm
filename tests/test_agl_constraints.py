"""Tests for the structural constraint-kind vocabulary in `agm.agl.constraints`."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from agm.agl.constraints import (
    CONSTRAINT_SPELLINGS,
    ConstraintKind,
    close_constraints,
    strongest_constraint,
)


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


_UNBOUNDED_MERGE_PROGRAM = """\
def f[K, V](a: dict[K, V], b: dict[K, V]) -> int = a.merge(b).size()

program def main() -> unit =
  print(f({1: 1}, {2: 2}))
"""

_DIAGNOSTICS_SCRIPT = """\
import sys
from agm.agl import PipelineDriver

result = PipelineDriver(resolve_agent_spec=None, get_sandbox_context=None).run(sys.argv[1])
print(" | ".join(d.message for d in result.diagnostics))
"""


@pytest.mark.parametrize("hash_seed", ["0", "2"])
def test_unmet_bound_reports_strongest_kind_under_any_hash_seed(hash_seed: str) -> None:
    """Seeds 0 and 2 give the two frozenset orders; the strongest unmet kind is named in both."""
    done = subprocess.run(
        [sys.executable, "-c", _DIAGNOSTICS_SCRIPT, _UNBOUNDED_MERGE_PROGRAM],
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Hashable" in done.stdout


def test_strongest_constraint_prefers_hashable_over_eq() -> None:
    both = close_constraints(frozenset({ConstraintKind.HASHABLE}))
    assert strongest_constraint(both) is ConstraintKind.HASHABLE
    assert strongest_constraint(frozenset({ConstraintKind.EQ})) is ConstraintKind.EQ
