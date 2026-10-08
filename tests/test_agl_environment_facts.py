"""Tests for ``TypeEnvironment``'s own-facts journal: recording and replay.

Covers ``begin_facts``/``end_facts``/``own_facts``/``seal``/``replay`` and
each journaled mutator's typed fact. ``TestFactScenarios`` parametrizes one
scenario per mutator: set up whatever state the mutator acts on, record one
fact, replay it onto an identically set-up environment, and check the queries
that mutator's fact drives agree with the source and differ from an
environment that was never replayed. One test drives a real module through
the pipeline's module path to confirm ``check_program`` never starts a
journal there.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
from collections.abc import Callable

import pytest

from agm.agl import artifact_serialization
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.semantics.type_table import MethodDef
from agm.agl.semantics.types import (
    EnumType,
    FunctionType,
    IntType,
    RecordType,
    TextType,
    Type,
    TypeTemplate,
)
from agm.agl.syntax.types import IntT, NameT
from agm.agl.typecheck import env as env_module
from agm.agl.typecheck.checker import _check_prepared_module
from agm.agl.typecheck.env import (
    AliasFact,
    BindingTypeFact,
    EnvironmentFact,
    EnvironmentFacts,
    FunctionSignature,
    GenericTypeDef,
    TypeEnvironment,
    TypeFact,
)
from tests._agl_helpers import dummy_span
from tests.agl.ir_harness import base_caps
from tests.agl.module_graph import check_resolved, resolve_and_check_entry, resolve_inline_entry

# ---------------------------------------------------------------------------
# Fact scenarios: one per journaled mutator, record + query
# ---------------------------------------------------------------------------


def test_persistent_environment_checks_each_type_identity_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = IntType()
    second = IntType()
    env = TypeEnvironment()
    env.register_type("First", first)
    env.register_type("Second", second)
    env.set_binding_type(1, first)
    env.register_function_signature("helper", FunctionSignature(params=(), result=second))
    scans: list[Type] = []
    original = env_module.contains_inference_var

    def scan(typ: Type) -> bool:
        scans.append(typ)
        return original(typ)

    monkeypatch.setattr(env_module, "contains_inference_var", scan)
    env.assert_closed()
    assert sum(typ is first for typ in scans) == 1
    assert sum(typ is second for typ in scans) == 1


def test_checked_module_seals_local_types_without_repeated_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = resolve_inline_entry("()", default_stdlib=False)
    env = TypeEnvironment()
    typ = RecordType("Marker", decl_id=900018, type_args=(IntType(),))
    env.set_binding_type(900019, typ)
    scans = 0
    original = env_module.contains_inference_var

    def scan(candidate: Type) -> bool:
        nonlocal scans
        if candidate is typ:
            scans += 1
        return original(candidate)

    monkeypatch.setattr(env_module, "contains_inference_var", scan)
    checked = _check_prepared_module(
        resolved,
        base_caps(),
        env=env,
        prepare_headers=False,
        infer_candidates=False,
    )
    assert checked.type_env.binding_type_of(900019) is typ
    assert scans == 1


def test_program_validates_shared_declarations_once(monkeypatch: pytest.MonkeyPatch) -> None:
    validations = 0
    original = TypeEnvironment.assert_shared_tables_closed

    def validate(env: TypeEnvironment) -> None:
        nonlocal validations
        validations += 1
        original(env)

    monkeypatch.setattr(TypeEnvironment, "assert_shared_tables_closed", validate)
    checked = resolve_and_check_entry("type Number = int", base_caps(), default_stdlib=False)
    assert checked.type_env.declared_type_template(ENTRY_ID, "Number") == TypeTemplate(IntType())
    assert validations == 1


def test_program_validates_sealed_local_state_once(monkeypatch: pytest.MonkeyPatch) -> None:
    validations = 0
    original = TypeEnvironment.assert_closed

    def validate(env: TypeEnvironment) -> None:
        nonlocal validations
        validations += 1
        original(env)

    monkeypatch.setattr(TypeEnvironment, "assert_closed", validate)
    checked = resolve_and_check_entry("type Number = int", base_caps(), default_stdlib=False)
    assert checked.type_env.declared_type_template(ENTRY_ID, "Number") == TypeTemplate(IntType())
    assert validations == 1


_HELPER_SIG = FunctionSignature(params=(), result=IntType())
_SCOPED_ENUM_TYPE = EnumType(name="Color")
_WIDGET_GENERIC_DEF = GenericTypeDef(
    kind="record", type_params=("T",), template=RecordType(name="Widget")
)
_ALIAS_TARGET = IntT(span=dummy_span(), node_id=1)
_METHOD_OWNER = RecordType(name="Box3", decl_id=909)
_OWNED_METHOD = MethodDef(
    module_id=ENTRY_ID,
    scope_path=(),
    name="doubled",
    decl_node_id=910,
    signature=FunctionType(params=(_METHOD_OWNER,), result=IntType()),
    receiver_type_param_arity=0,
    type_params=(),
)
_BUILTIN_METHOD = MethodDef(
    module_id=ENTRY_ID,
    scope_path=(),
    name="tripled",
    decl_node_id=911,
    signature=FunctionType(params=(IntType(),), result=IntType()),
    receiver_type_param_arity=0,
    type_params=(),
)


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
    resolved = env.get_type("Shapes::Color")
    return resolved


def _record_register_generic_type(env: TypeEnvironment) -> None:
    env.register_generic_type("Widget", _WIDGET_GENERIC_DEF)


def _query_register_generic_type(env: TypeEnvironment) -> object:
    return (env.all_generic_types(), env.get_generic_type("Widget"))


def _record_register_alias(env: TypeEnvironment) -> None:
    env.register_alias("Alias", _ALIAS_TARGET, type_params=(), declaration_span=dummy_span())


def _query_register_alias(env: TypeEnvironment) -> object:
    registered = env.has_alias_registration("Alias", _ALIAS_TARGET, ())
    # ``declared_type_template`` assumes the caller already knows the type is
    # declared; a never-mutated comparison environment has no "Alias" at all.
    template = env.declared_type_template(ENTRY_ID, "Alias") if registered else None
    return (template, registered)


def _setup_unregister_name(env: TypeEnvironment) -> None:
    env.register_generic_type("Retired", _WIDGET_GENERIC_DEF)


def _record_unregister_name(env: TypeEnvironment) -> None:
    env.unregister_name("Retired")


def _query_unregister_name(env: TypeEnvironment) -> object:
    return env.all_generic_types()


def _setup_freeze_alias(env: TypeEnvironment) -> None:
    env.register_alias("Frozen", _ALIAS_TARGET, type_params=(), declaration_span=dummy_span())


def _record_freeze_alias(env: TypeEnvironment) -> None:
    env.freeze_alias("Frozen", TextType())


def _query_freeze_alias(env: TypeEnvironment) -> object:
    return env.declared_type_template(ENTRY_ID, "Frozen")


def _record_register_method_def(env: TypeEnvironment) -> None:
    env.register_method_def(_METHOD_OWNER, _OWNED_METHOD)


def _query_register_method_def(env: TypeEnvironment) -> object:
    return env.type_table.method_candidates(_METHOD_OWNER, "doubled")


def _record_register_builtin_method_def(env: TypeEnvironment) -> None:
    env.register_method_def("int", _BUILTIN_METHOD)


def _query_register_builtin_method_def(env: TypeEnvironment) -> object:
    return env.type_table.method_candidates(IntType(), "tripled")


def _no_setup(env: TypeEnvironment) -> None:
    """Most mutators act on an empty environment and need no prior state."""


@dataclasses.dataclass(frozen=True)
class _Scenario:
    """One mutator's record call plus the query that observes its fact.

    ``setup`` runs before the journal opens, for a mutator whose effect is
    only visible against state it did not record itself (a name it retires, an
    alias whose resolved template it replaces).
    """

    label: str
    record: Callable[[TypeEnvironment], None]
    query: Callable[[TypeEnvironment], object]
    setup: Callable[[TypeEnvironment], None] = _no_setup


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
        "unregister_name",
        _record_unregister_name,
        _query_unregister_name,
        setup=_setup_unregister_name,
    ),
    _Scenario("freeze_alias", _record_freeze_alias, _query_freeze_alias, setup=_setup_freeze_alias),
    _Scenario("register_method_def", _record_register_method_def, _query_register_method_def),
    _Scenario(
        "register_method_def_builtin",
        _record_register_builtin_method_def,
        _query_register_builtin_method_def,
    ),
)

_SCENARIO_IDS = [scenario.label for scenario in _SCENARIOS]


def _prepared(scenario: _Scenario) -> TypeEnvironment:
    """A fresh environment carrying whatever state this scenario acts on."""
    env = TypeEnvironment()
    scenario.setup(env)
    return env


# ---------------------------------------------------------------------------
# Per-mutator replay parity and idempotence
# ---------------------------------------------------------------------------


class TestFactScenarios:
    @pytest.mark.parametrize("scenario", _SCENARIOS, ids=_SCENARIO_IDS)
    def test_replay_reproduces_source_and_differs_from_a_never_replayed_environment(
        self, scenario: _Scenario
    ) -> None:
        source = _prepared(scenario)
        source.begin_facts()
        scenario.record(source)
        facts = source.own_facts()

        target = _prepared(scenario)
        target.replay(facts)

        fresh = _prepared(scenario)

        assert scenario.query(target) == scenario.query(source)
        assert scenario.query(target) != scenario.query(fresh)

    @pytest.mark.parametrize("scenario", _SCENARIOS, ids=_SCENARIO_IDS)
    def test_replay_is_idempotent(self, scenario: _Scenario) -> None:
        source = _prepared(scenario)
        source.begin_facts()
        scenario.record(source)
        facts = source.own_facts()
        expected = scenario.query(source)

        replayed_twice = _prepared(scenario)
        replayed_twice.replay(facts)
        replayed_twice.replay(facts)
        assert scenario.query(replayed_twice) == expected

        already_holding = _prepared(scenario)
        scenario.record(already_holding)
        already_holding.replay(facts)
        assert scenario.query(already_holding) == expected


# ---------------------------------------------------------------------------
# Fact-to-mutator forwarding contract
# ---------------------------------------------------------------------------


def _fact_classes() -> list[type[EnvironmentFact]]:
    """Every journaled fact ``typecheck.env`` declares."""
    return [
        value
        for value in vars(env_module).values()
        if isinstance(value, type)
        and issubclass(value, EnvironmentFact)
        and dataclasses.is_dataclass(value)
    ]


class TestFactForwarding:
    def test_each_fact_supplies_its_mutator_by_parameter_name(self) -> None:
        """A fact replays as a keyword call, so its fields are that method's parameters.

        Renaming a mutator, a parameter, or a fact field, or giving a mutator
        a new required parameter, breaks replay without a type error --
        ``_MUTATOR`` names the method as text.
        """
        facts = _fact_classes()
        assert len(facts) == len({fact.__name__ for fact in facts}) > 0
        for fact in facts:
            mutator = getattr(TypeEnvironment, fact._MUTATOR)
            parameters = inspect.signature(mutator).parameters
            for name in fact.__match_args__:
                assert parameters[name].kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ), (fact.__name__, name)
            required = {
                name
                for name, parameter in parameters.items()
                if name != "self" and parameter.default is inspect.Parameter.empty
            }
            assert required <= set(fact.__match_args__), fact.__name__

    def test_every_fact_has_a_replay_scenario(self) -> None:
        assert {fact._MUTATOR for fact in _fact_classes()} <= set(_SCENARIO_IDS)


# ---------------------------------------------------------------------------
# Journal lifecycle
# ---------------------------------------------------------------------------


class TestJournalLifecycle:
    def test_mutations_are_recorded_only_from_begin_facts_onward_in_order(self) -> None:
        env = TypeEnvironment()
        env.register_type("Foo", RecordType(name="Foo"))
        env.begin_facts()
        assert env.own_facts().entries == ()

        env.register_type("Foo", RecordType(name="Foo"))
        env.set_binding_type(801, IntType())
        entries = env.own_facts().entries
        assert entries == (
            TypeFact(name="Foo", typ=RecordType(name="Foo")),
            BindingTypeFact(node_id=801, typ=IntType()),
        )

    def test_end_facts_closes_one_window_and_leaves_the_next_free_to_open(self) -> None:
        """Header preparation and the body check journal the same environment in
        turn, so what the first window took must not reach the second's facts."""
        env = TypeEnvironment()
        env.begin_facts()
        env.register_type("Header", RecordType(name="Header"))
        header_facts = env.end_facts()

        env.begin_facts()
        env.set_binding_type(804, IntType())
        env.seal()

        assert header_facts.entries == (TypeFact(name="Header", typ=RecordType(name="Header")),)
        assert env.own_facts().entries == (BindingTypeFact(node_id=804, typ=IntType()),)

    def test_own_facts_is_stable_after_seal(self) -> None:
        env = TypeEnvironment()
        env.begin_facts()
        env.register_type("Foo", RecordType(name="Foo"))
        env.seal()
        facts = env.own_facts()
        assert env.own_facts() == facts

    def test_begin_replay_seal_reproduces_facts_exactly(self) -> None:
        target_expr = NameT(name="A", span=dummy_span(), node_id=901)
        env = TypeEnvironment()
        env.register_alias("A", target_expr, declaration_span=dummy_span())
        env.freeze_alias("A", TextType())
        facts = EnvironmentFacts(
            entries=(
                BindingTypeFact(node_id=902, typ=IntType()),
                AliasFact(
                    name="A", target_expr=target_expr, type_params=(), declaration_span=dummy_span()
                ),
            )
        )

        env.begin_facts()
        env.replay(facts)
        env.seal()

        assert env.own_facts() == facts


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
        target.register_alias("A", target_expr, declaration_span=dummy_span())
        target.freeze_alias("A", IntType())
        facts = EnvironmentFacts(
            entries=(
                AliasFact(
                    name="A", target_expr=target_expr, type_params=(), declaration_span=dummy_span()
                ),
            )
        )

        target.replay(facts)

        assert target.declared_type_template(ENTRY_ID, "A") == TypeTemplate(IntType())

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
        target.register_alias(
            "A", NameT(name="A", span=dummy_span(), node_id=1101), declaration_span=dummy_span()
        )
        target.freeze_alias("A", TextType())
        facts = EnvironmentFacts(
            entries=(
                AliasFact(
                    name="A",
                    target_expr=differing_target_expr,
                    type_params=differing_type_params,
                    declaration_span=dummy_span(),
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
        for scenario in _SCENARIOS:
            scenario.setup(source)
        source.begin_facts()
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
        assert checked.type_env.own_facts() == EnvironmentFacts()
