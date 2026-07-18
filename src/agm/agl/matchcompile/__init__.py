"""Public match-compilation artifacts, diagnostics, and decision contract.

Matrix construction, normalization, allocation, provenance, and compiler
helpers are implementation details available only from their defining
submodules.
"""

from .compiler import CompiledCase
from .diagnostics import (
    BoolWitness,
    EnumWitness,
    EnumWitnessQualification,
    LiteralWitness,
    MatchIssue,
    MatchWitness,
    NonExhaustiveIssue,
    OpenComplementWitness,
    RedundantArmIssue,
    WildcardWitness,
    WitnessField,
    render_witness,
)
from .model import (
    BoolConstructor,
    Constructor,
    Decision,
    DecisionLeaf,
    DecisionSwitch,
    EnumConstructor,
    FieldOccurrenceProvenance,
    LiteralKind,
    Occurrence,
    OccurrenceId,
)
from .stage import (
    MatchCompilationResult,
    MatchCompiledArtifact,
    MatchCompiledModule,
    MatchCompiledProgram,
    compile_module_matches,
    compile_program_matches,
    diagnostic_from_match_issue,
    diagnostics_from_match_issues,
    validate_match_compiled_module,
    validate_match_compiled_program,
)

__all__ = [
    "BoolConstructor",
    "BoolWitness",
    "Constructor",
    "CompiledCase",
    "Decision",
    "DecisionLeaf",
    "DecisionSwitch",
    "EnumConstructor",
    "EnumWitness",
    "FieldOccurrenceProvenance",
    "EnumWitnessQualification",
    "LiteralKind",
    "LiteralWitness",
    "MatchCompilationResult",
    "MatchCompiledArtifact",
    "MatchCompiledProgram",
    "MatchCompiledModule",
    "MatchIssue",
    "MatchWitness",
    "NonExhaustiveIssue",
    "Occurrence",
    "OccurrenceId",
    "OpenComplementWitness",
    "RedundantArmIssue",
    "WildcardWitness",
    "WitnessField",
    "compile_program_matches",
    "compile_module_matches",
    "diagnostic_from_match_issue",
    "diagnostics_from_match_issues",
    "render_witness",
    "validate_match_compiled_program",
    "validate_match_compiled_module",
]
