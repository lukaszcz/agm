"""Tests for ``TypeEnvironment``'s own-facts journal: recording and replay.

Covers ``begin_own_facts``/``own_facts``/``seal``/``replay`` and each of the
nine journaled mutators' typed fact. ``TestFactScenarios`` parametrizes one
scenario per mutator: record a fact, replay it onto a freshly prepared
environment, and check the queries that mutator's fact drives agree with the
source and differ from an environment that was never replayed. One test
drives a real module through the pipeline's module path to confirm
``check_program`` never starts a journal there.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable

import pytest

from agm.agl import artifact_serialization
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.semantics.types import (
    EnumType,
    IntType,
    RecordType,
    TextType,
    TypeTemplate,
)
from agm.agl.syntax.types import IntT, NameT
from agm.agl.typecheck.env import (
    AliasFact,
    BindingTypeFact,
    ConstructorSignature,
    EnvironmentFacts,
    FunctionSignature,
    GenericTypeDef,
    TypeEnvironment,
    TypeFact,
)
from agm.agl.zones import ParamZone
from tests._agl_helpers import dummy_span
from tests.agl.module_graph import check_resolved, resolve_inline_entry

# ---------------------------------------------------------------------------
# Fact scenarios: one per journaled mutator, record + query
# ---------------------------------------------------------------------------

_HELPER_SIG = FunctionSignature(params=(), result=IntType())
_SCOPED_ENUM_TYPE = EnumType(name="Color")
_WIDGET_GENERIC_DEF = GenericTypeDef(
    kind="record", type_params=("T",), template=RecordType(name="Widget")
)
_ALIAS_TARGET = IntT(span=dummy_span(), node_id=1)
_BOX2_RECORD_TYPE = RecordType(name="Box2")
_BOX2_CTOR_SIG = ConstructorSignature(
    owner_name="Box2",
    field_names=("value",),
    field_templates=(IntType(),),
    result_template=_BOX2_RECORD_TYPE,
    type_params=(),
)
_FIELD_KINDS = (("value", ParamZone.STANDARD),)


def _record_set_binding_type(env: TypeEnvironment) -> None:
    env.set_binding_type(101, IntType())


def _query_set_binding_type(env: TypeEnvironment) -> object:
    return env.get_binding_type(101)


def _record_register_function_signature(env: TypeEnvironment) -> None:
    env.register_function_signature("helper", _HELPER_SIG)


def _query_register_function_signature(env: TypeEnvironment) -> object:
    return env.all_function_signatures().get("helper")


def _record_register_function_signature_by_node_id(env: TypeEnvironment) -> None:
    env.register_function_signature_by_node_id(202, _HELPER_SIG)


def _query_register_function_signature_by_node_id(env: TypeEnvironment) -> object:
    return env.get_function_signature_by_node_id(202)


def _record_register_extern_node_id(env: TypeEnvironment) -> None:
    env.register_extern_node_id(303)


def _query_register_extern_node_id(env: TypeEnvironment) -> object:
    return env.is_extern_node_id(303)


def _record_register_type(env: TypeEnvironment) -> None:
    env.register_type("Shapes::Color", _SCOPED_ENUM_TYPE)


def _query_register_type(env: TypeEnvironment) -> object:
    with env.type_scope(("Shapes",)):
        resolved = env.resolve_named_type("Color")
    # blocked_enum_variants() needs module routes only a program-context env has,
    # so it is trivially equal here; program-level parity covers it.
    return (resolved, env.enum_owner_forms())


def _record_register_generic_type(env: TypeEnvironment) -> None:
    env.register_generic_type("Widget", _WIDGET_GENERIC_DEF)


def _query_register_generic_type(env: TypeEnvironment) -> object:
    return (env.all_generic_types(), env.resolve_named_type("Widget"))


def _record_register_alias(env: TypeEnvironment) -> None:
    env.register_alias("Alias", _ALIAS_TARGET, type_params=())


def _query_register_alias(env: TypeEnvironment) -> object:
    return (
        env.source_type_template_qname(ENTRY_ID, "Alias"),
        env.resolve_named_type("Alias"),
    )


def _record_register_constructor_signature(env: TypeEnvironment) -> None:
    env.register_constructor_signature(_BOX2_CTOR_SIG)


def _query_register_constructor_signature(env: TypeEnvironment) -> object:
    return env.get_constructor_signature("Box2")


def _record_register_constructor_field_kinds(env: TypeEnvironment) -> None:
    env.register_constructor_field_kinds(
        "Widget", _FIELD_KINDS, scope_path=("Scoped",), module_id=ENTRY_ID, decl_id=707
    )


def _query_register_constructor_field_kinds(env: TypeEnvironment) -> object:
    return (
        env.get_constructor_field_kinds("Widget", scope_path=("Scoped",)),
        env.get_constructor_field_kinds_for_type(RecordType(name="Widget", decl_id=707), "Widget"),
    )


@dataclasses.dataclass(frozen=True)
class _Scenario:
    """One mutator's record call plus the query that observes its fact."""

    label: str
    record: Callable[[TypeEnvironment], None]
    query: Callable[[TypeEnvironment], object]


_SCENARIOS: tuple[_Scenario, ...] = (
    _Scenario("set_binding_type", _record_set_binding_type, _query_set_binding_type),
    _Scenario(
        "register_function_signature",
        _record_register_function_signature,
        _query_register_function_signature,
    ),
    _Scenario(
        "register_function_signature_by_node_id",
        _record_register_function_signature_by_node_id,
        _query_register_function_signature_by_node_id,
    ),
    _Scenario(
        "register_extern_node_id", _record_register_extern_node_id, _query_register_extern_node_id
    ),
    _Scenario("register_type", _record_register_type, _query_register_type),
    _Scenario("register_generic_type", _record_register_generic_type, _query_register_generic_type),
    _Scenario("register_alias", _record_register_alias, _query_register_alias),
    _Scenario(
        "register_constructor_signature",
        _record_register_constructor_signature,
        _query_register_constructor_signature,
    ),
    _Scenario(
        "register_constructor_field_kinds",
        _record_register_constructor_field_kinds,
        _query_register_constructor_field_kinds,
    ),
)

_SCENARIO_IDS = [scenario.label for scenario in _SCENARIOS]


# ---------------------------------------------------------------------------
# Per-mutator replay parity and idempotence
# ---------------------------------------------------------------------------


class TestFactScenarios:
    @pytest.mark.parametrize("scenario", _SCENARIOS, ids=_SCENARIO_IDS)
    def test_replay_reproduces_source_and_differs_from_a_never_replayed_environment(
        self, scenario: _Scenario
    ) -> None:
        source = TypeEnvironment()
        source.begin_own_facts()
        scenario.record(source)
        facts = source.own_facts()

        target = TypeEnvironment()
        with target.type_scope(("Elsewhere",)):
            target.replay(facts)

        fresh = TypeEnvironment()

        assert scenario.query(target) == scenario.query(source)
        assert scenario.query(target) != scenario.query(fresh)

    @pytest.mark.parametrize("scenario", _SCENARIOS, ids=_SCENARIO_IDS)
    def test_replay_is_idempotent(self, scenario: _Scenario) -> None:
        source = TypeEnvironment()
        source.begin_own_facts()
        scenario.record(source)
        facts = source.own_facts()
        expected = scenario.query(source)

        replayed_twice = TypeEnvironment()
        replayed_twice.replay(facts)
        replayed_twice.replay(facts)
        assert scenario.query(replayed_twice) == expected

        already_holding = TypeEnvironment()
        scenario.record(already_holding)
        already_holding.replay(facts)
        assert scenario.query(already_holding) == expected


# ---------------------------------------------------------------------------
# Constructor field kinds: module-id keying
# ---------------------------------------------------------------------------


class TestConstructorFieldKindsModuleId:
    def test_replay_records_the_resolved_module_id_for_cross_module_lookup(self) -> None:
        lib_id = ModuleId(("lib",))
        source = TypeEnvironment(module_id=lib_id)
        source.begin_own_facts()
        source.register_constructor_field_kinds("Widget", _FIELD_KINDS, scope_path=("Scoped",))
        facts = source.own_facts()

        target = TypeEnvironment()
        target.replay(facts)

        assert (
            target.get_constructor_field_kinds("Widget", scope_path=("Scoped",), module_id=lib_id)
            == _FIELD_KINDS
        )
        assert target.get_constructor_field_kinds("Widget", scope_path=("Scoped",)) is None


# ---------------------------------------------------------------------------
# Journal lifecycle
# ---------------------------------------------------------------------------


class TestJournalLifecycle:
    def test_mutations_are_recorded_only_from_begin_own_facts_onward_in_order(self) -> None:
        env = TypeEnvironment()
        env.register_type("Foo", RecordType(name="Foo"))
        env.begin_own_facts()
        assert env.own_facts().entries == ()

        env.register_type("Foo", RecordType(name="Foo"))
        env.set_binding_type(801, IntType())
        entries = env.own_facts().entries
        assert entries == (
            TypeFact(name="Foo", typ=RecordType(name="Foo")),
            BindingTypeFact(node_id=801, typ=IntType()),
        )

    def test_own_facts_before_begin_own_facts_raises(self) -> None:
        env = TypeEnvironment()
        with pytest.raises(AssertionError):
            env.own_facts()

    def test_begin_own_facts_twice_raises(self) -> None:
        env = TypeEnvironment()
        env.begin_own_facts()
        with pytest.raises(AssertionError):
            env.begin_own_facts()

    def test_begin_own_facts_on_a_sealed_environment_raises(self) -> None:
        env = TypeEnvironment()
        env.seal()
        with pytest.raises(AssertionError):
            env.begin_own_facts()

    def test_own_facts_is_stable_after_seal(self) -> None:
        env = TypeEnvironment()
        env.begin_own_facts()
        env.register_type("Foo", RecordType(name="Foo"))
        env.seal()
        facts = env.own_facts()
        assert env.own_facts() == facts

    def test_begin_replay_seal_reproduces_facts_exactly(self) -> None:
        target_expr = NameT(name="A", span=dummy_span(), node_id=901)
        env = TypeEnvironment()
        env.register_alias("A", target_expr)
        env.freeze_alias("A", TextType())
        facts = EnvironmentFacts(
            entries=(
                BindingTypeFact(node_id=902, typ=IntType()),
                AliasFact(name="A", target_expr=target_expr, type_params=()),
            )
        )

        env.begin_own_facts()
        env.replay(facts)
        env.seal()

        assert env.own_facts() == facts

    def test_replay_of_empty_facts_onto_a_sealed_environment_raises(self) -> None:
        target = TypeEnvironment()
        target.seal()
        with pytest.raises(AssertionError):
            target.replay(EnvironmentFacts())


# ---------------------------------------------------------------------------
# Alias replay skip
# ---------------------------------------------------------------------------


class TestAliasReplaySkip:
    def test_replay_skips_a_duplicate_alias_registration_preserving_the_frozen_template(
        self,
    ) -> None:
        """A self-referencing alias would cycle if re-resolved; the frozen
        template must survive an identical replayed registration."""
        target_expr = NameT(name="A", span=dummy_span(), node_id=1001)
        target = TypeEnvironment()
        target.register_alias("A", target_expr)
        target.freeze_alias("A", IntType())
        facts = EnvironmentFacts(
            entries=(AliasFact(name="A", target_expr=target_expr, type_params=()),)
        )

        target.replay(facts)

        assert target.source_type_template_qname(ENTRY_ID, "A") == TypeTemplate(IntType())

    @pytest.mark.parametrize(
        "differing_target_expr,differing_type_params",
        [
            pytest.param(IntT(span=dummy_span(), node_id=1102), (), id="different_target"),
            pytest.param(
                NameT(name="A", span=dummy_span(), node_id=1101),
                ("T",),
                id="same_target_different_type_params",
            ),
        ],
    )
    def test_replay_does_not_skip_a_differing_alias_registration(
        self, differing_target_expr: object, differing_type_params: tuple[str, ...]
    ) -> None:
        target = TypeEnvironment()
        target.register_alias("A", NameT(name="A", span=dummy_span(), node_id=1101))
        target.freeze_alias("A", TextType())
        facts = EnvironmentFacts(
            entries=(
                AliasFact(
                    name="A",
                    target_expr=differing_target_expr,
                    type_params=differing_type_params,
                ),
            )
        )

        target.replay(facts)

        # A non-identical re-registration is not skipped: the fact's target/params
        # replace the original registration rather than being discarded.
        assert target.has_alias_registration("A", differing_target_expr, differing_type_params)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestArtifactSerialization:
    def test_environment_facts_round_trip_through_artifact_serialization(self) -> None:
        source = TypeEnvironment()
        source.begin_own_facts()
        for scenario in _SCENARIOS:
            scenario.record(source)
        facts = source.own_facts()
        assert len(facts.entries) == len(_SCENARIOS)

        key = os.urandom(16)
        artifact_serialization.save(key, "test-environment-facts-round-trip", facts)
        restored = artifact_serialization.load(key, "test-environment-facts-round-trip")

        assert restored == facts


# ---------------------------------------------------------------------------
# Module path
# ---------------------------------------------------------------------------


class TestModulePathNeverJournals:
    def test_module_path_checking_never_starts_a_journal(self) -> None:
        """A real single-module check never opens the journal window."""
        checked = check_resolved(resolve_inline_entry("def value() -> int = 1"))
        with pytest.raises(AssertionError):
            checked.type_env.own_facts()
