"""Tests for module-parameter config routing without a command host."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from agm.agl.attributes import ProgramOptionSpec
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.runtime.types import ParamBindingInfo, ProgramDeclInfo, ProgramParamInfo
from agm.agl.semantics.types import BoolType, IntType, TextType, Type
from agm.agl.syntax.spans import SourceSpan
from agm.agl.zones import ParamZone
from agm.cli_support.param_config import (
    ParamValueTiers,
    RouteReport,
    _report_undeclared_config_keys,
    resolve_param_values,
)
from agm.cli_support.param_surface import ParamSurface, build_param_surface
from agm.config.general import GeneralConfig
from agm.config.qualified_keys import QualifiedConfigLookupError
from agm.core.toml import toml_dict

_SPAN = SourceSpan(1, 1, 1, 2, 0, 1)


def _config(*layers: Mapping[str, object]) -> GeneralConfig:
    return GeneralConfig.from_layers(toml_dict(layer) for layer in layers)


def _option(name: str) -> ProgramOptionSpec:
    return ProgramOptionSpec(name=name, short=None, env=None, metavar=None, hidden=False, doc=None)


def _binding(
    module: str | ModuleId,
    name: str,
    type_: Type = BoolType(),
    *,
    external: str | None = None,
    scope_path: tuple[str, ...] = (),
) -> ParamBindingInfo:
    return ParamBindingInfo(
        module=ModuleId.from_path(module) if isinstance(module, str) else module,
        scope_path=scope_path,
        name=name,
        node_id=1,
        span=_SPAN,
        type=type_,
        mutable=False,
        cli=_option(name if external is None else external),
        doc=None,
    )


def _program(
    *,
    module: str | ModuleId = "app/main",
    name: str = "run",
    closure: tuple[ModuleId, ...] = (),
    parameters: tuple[ProgramParamInfo, ...] = (),
) -> ProgramDeclInfo:
    module_id = ModuleId.from_path(module) if isinstance(module, str) else module
    return ProgramDeclInfo(
        module=module_id,
        scope_path=(),
        name=name,
        node_id=1,
        span=_SPAN,
        parameters=parameters,
        is_entry=module_id.is_entry,
        doc=None,
        closure=closure or (module_id,),
    )


def _surface(program: ProgramDeclInfo, bindings: tuple[ParamBindingInfo, ...]) -> ParamSurface:
    return build_param_surface(frozenset(), program, bindings)


def _resolve(
    config: GeneralConfig,
    program: ProgramDeclInfo,
    bindings: tuple[ParamBindingInfo, ...],
    params: Mapping[tuple[ModuleId, tuple[str, ...], str], object] = {},
) -> tuple[dict[tuple[ModuleId, tuple[str, ...], str], object], list[RouteReport]]:
    tiers, reports = resolve_param_values(
        config,
        program,
        params,
        entry_segments=("workflow",),
        command_paths=(),
        surface=_surface(program, bindings),
    )
    return {**tiers.lower, **tiers.upper}, reports


def test_resolves_a_module_route() -> None:
    binding = _binding("A/logging", "verbose")
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    values, _reports = _resolve(_config({"A": {"logging": {"verbose": True}}}), program, (binding,))

    assert values == {binding.key: True}


def test_resolves_a_scope_region_table() -> None:
    binding = _binding("A/logging", "trace", scope_path=("debug",))
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    values, _reports = _resolve(
        _config({"A": {"logging": {"debug": {"trace": True}}}}), program, (binding,)
    )

    assert values == {binding.key: True}


def test_program_route_overrides_a_module_route_from_any_layer() -> None:
    binding = _binding("A/logging", "verbose")
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    values, _reports = _resolve(
        _config(
            {"A": {"logging": {"verbose": False}}},
            {"workflow": {"run": {"verbose": True}}},
        ),
        program,
        (binding,),
    )

    assert values == {binding.key: True}


def test_a_lower_layer_program_route_overrides_a_higher_layer_module_route() -> None:
    binding = _binding("A/logging", "verbose")
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    values, _reports = _resolve(
        _config(
            {"workflow": {"run": {"verbose": True}}},
            {"A": {"logging": {"verbose": False}}},
        ),
        program,
        (binding,),
    )

    assert values == {binding.key: True}


def test_rejects_a_leaf_shared_by_a_module_and_program_route() -> None:
    imported = _binding("workflow/main", "verbose")
    own = _binding(ENTRY_ID, "verbose")
    program = _program(module=ENTRY_ID, name="main", closure=(ENTRY_ID, imported.module))

    with pytest.raises(QualifiedConfigLookupError) as exc_info:
        _resolve(_config({"workflow": {"main": {"verbose": True}}}), program, (imported, own))

    assert imported.declaration_path in str(exc_info.value)
    assert own.declaration_path in str(exc_info.value)


def test_distinct_module_and_program_tables_do_not_collide() -> None:
    imported = _binding("A/logging", "verbose")
    own = _binding(ENTRY_ID, "verbose")
    program = _program(module=ENTRY_ID, closure=(ENTRY_ID, imported.module))

    values, _reports = _resolve(
        _config({"A": {"logging": {"verbose": False}}}), program, (imported, own)
    )

    assert values == {imported.key: False}


def test_option_name_is_used_for_module_and_program_routes_but_keeps_the_declared_key() -> None:
    binding = _binding("A/logging", "verbose", external="chatty")
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    values, _reports = _resolve(
        _config(
            {"A": {"logging": {"chatty": False}}},
            {"workflow": {"run": {"chatty": True}}},
        ),
        program,
        (binding,),
    )

    assert values == {binding.key: True}


def test_resolves_a_quoted_module_anchor() -> None:
    binding = _binding("A/logging", "trace", scope_path=("debug",))
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    values, _reports = _resolve(
        _config({"A/logging": {"debug": {"trace": True}}}), program, (binding,)
    )

    assert values == {binding.key: True}


def test_rejects_an_ambiguous_program_table_leaf_when_configured() -> None:
    first = _binding("first/logging", "verbose")
    second = _binding("second/logging", "verbose")
    program = _program(
        closure=(ModuleId.from_path("app/main"), first.module, second.module),
    )

    with pytest.raises(QualifiedConfigLookupError) as exc_info:
        _resolve(_config({"workflow": {"run": {"verbose": True}}}), program, (first, second))

    assert first.declaration_path in str(exc_info.value)
    assert second.declaration_path in str(exc_info.value)


def test_rejects_a_configured_duplicate_external_name_in_one_module_route() -> None:
    first = _binding("A/logging", "verbose", external="level")
    second = _binding("A/logging", "detail", external="level")
    program = _program(closure=(ModuleId.from_path("app/main"), first.module))

    with pytest.raises(QualifiedConfigLookupError) as exc_info:
        _resolve(_config({"A": {"logging": {"level": True}}}), program, (first, second))

    assert first.declaration_path in str(exc_info.value)
    assert second.declaration_path in str(exc_info.value)


def test_an_inactive_duplicate_external_name_in_one_module_route_does_not_error() -> None:
    first = _binding("A/logging", "verbose", external="level")
    second = _binding("A/logging", "detail", external="level")
    program = _program(closure=(ModuleId.from_path("app/main"), first.module))

    values, _reports = _resolve(_config(), program, (first, second))

    assert values == {}


def test_duplicate_external_names_at_distinct_module_scopes_remain_independent() -> None:
    root = _binding("A/logging", "verbose", external="level")
    scoped = _binding("A/logging", "detail", external="level", scope_path=("debug",))
    program = _program(closure=(ModuleId.from_path("app/main"), root.module))

    values, _reports = _resolve(
        _config({"A": {"logging": {"level": True, "debug": {"level": False}}}}),
        program,
        (root, scoped),
    )

    assert values == {root.key: True, scoped.key: False}


def test_an_inactive_ambiguous_program_leaf_does_not_prevent_resolution() -> None:
    first = _binding("first/logging", "verbose")
    second = _binding("second/logging", "verbose")
    program = _program(
        closure=(ModuleId.from_path("app/main"), first.module, second.module),
    )

    values, _reports = _resolve(_config(), program, (first, second))

    assert values == {}


def test_program_route_accepts_a_qualified_leaf() -> None:
    binding = _binding("A/logging", "verbose")
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    values, _reports = _resolve(
        _config({"workflow": {"run": {"logging.verbose": True}}}), program, (binding,)
    )

    assert values == {binding.key: True}


def test_program_route_accepts_the_longest_qualified_leaf() -> None:
    binding = _binding("A/logging", "trace", scope_path=("debug",))
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    values, _reports = _resolve(
        _config({"workflow": {"run": {"A.logging.debug.trace": True}}}), program, (binding,)
    )

    assert values == {binding.key: True}


def test_a_qualified_leaf_reaches_a_param_whose_bare_name_a_signature_claims() -> None:
    binding = _binding("A/logging", "verbose")
    signature = ProgramParamInfo(
        name="verbose",
        kind=ParamZone.NAMED_ONLY,
        type=BoolType(),
        has_default=False,
        span=_SPAN,
        cli=_option("verbose"),
    )
    program = _program(
        closure=(ModuleId.from_path("app/main"), binding.module),
        parameters=(signature,),
    )

    values, _reports = _resolve(
        _config({"workflow": {"run": {"logging.verbose": True}}}), program, (binding,)
    )

    assert values == {binding.key: True}


def test_a_qualified_leaf_reaches_a_param_named_after_an_engine_key() -> None:
    binding = _binding("A/logging", "timeout", TextType())
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    values, _reports = _resolve(
        _config({"workflow": {"run": {"logging.timeout": "30s"}}}), program, (binding,)
    )

    assert values == {binding.key: "30s"}


def test_a_bare_engine_key_leaf_still_never_reaches_a_param() -> None:
    binding = _binding("A/logging", "timeout", TextType())
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    values, _reports = _resolve(
        _config({"workflow": {"run": {"timeout": "30s"}}}), program, (binding,)
    )

    assert values == {}


def test_rejects_two_spellings_of_one_param_in_one_program_table() -> None:
    binding = _binding("A/logging", "verbose")
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    with pytest.raises(QualifiedConfigLookupError) as exc_info:
        _resolve(
            _config({"workflow": {"run": {"verbose": True, "logging.verbose": False}}}),
            program,
            (binding,),
        )

    assert "verbose" in str(exc_info.value)
    assert "logging.verbose" in str(exc_info.value)


def test_a_later_layer_qualified_leaf_overrides_an_earlier_bare_leaf() -> None:
    binding = _binding("A/logging", "verbose")
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    values, _reports = _resolve(
        _config(
            {"workflow": {"run": {"verbose": False}}},
            {"workflow": {"run": {"logging.verbose": True}}},
        ),
        program,
        (binding,),
    )

    assert values == {binding.key: True}


def test_a_qualified_program_leaf_beats_a_module_route() -> None:
    binding = _binding("A/logging", "verbose")
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    values, _reports = _resolve(
        _config(
            {"workflow": {"run": {"logging.verbose": True}}},
            {"A": {"logging": {"verbose": False}}},
        ),
        program,
        (binding,),
    )

    assert values == {binding.key: True}


def test_a_qualified_program_leaf_is_declared_and_draws_no_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    binding = _binding("A/logging", "verbose")
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))
    config = _config({"workflow": {"run": {"logging.verbose": True}}})

    _values, reports = _resolve(config, program, (binding,))
    _report_undeclared_config_keys(config, reports)

    assert "logging.verbose" not in capsys.readouterr().err


def test_non_engine_key_leaf_forms_a_program_route_for_a_param() -> None:
    binding = _binding("A/logging", "max-iters", IntType())
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    values, _reports = _resolve(
        _config({"workflow": {"run": {"max-iters": 4}}}), program, (binding,)
    )

    assert values == {binding.key: 4}


def test_anonymous_entry_params_do_not_read_a_module_route() -> None:
    binding = _binding(ENTRY_ID, "verbose")
    program = _program(module=ENTRY_ID)

    tiers, _reports = resolve_param_values(
        _config({"workflow": {"run": {"verbose": True}}}),
        program,
        {},
        entry_segments=(),
        command_paths=(),
        surface=_surface(program, (binding,)),
    )

    assert {**tiers.lower, **tiers.upper} == {}


def test_cli_or_environment_values_are_not_overwritten() -> None:
    binding = _binding("A/logging", "verbose")
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    values, _reports = _resolve(
        _config({"A": {"logging": {"verbose": False}}}),
        program,
        (binding,),
        {binding.key: True},
    )

    assert values == {binding.key: True}


def test_reports_each_module_route_and_the_program_route() -> None:
    own = _binding("app/main", "quiet")
    imported = _binding("A/logging", "verbose")
    signature = ProgramParamInfo(
        name="label",
        kind=ParamZone.NAMED_ONLY,
        type=TextType(),
        has_default=False,
        span=_SPAN,
        cli=_option("label"),
    )
    program = _program(
        closure=(own.module, imported.module),
        parameters=(signature,),
    )

    _values, reports = _resolve(_config(), program, (own, imported))

    assert reports == [
        RouteReport(("app", "main"), (), (), frozenset({"quiet"}), frozenset()),
        RouteReport(("A", "logging"), (), (), frozenset({"verbose"}), frozenset()),
        RouteReport(
            ("workflow",),
            ("run",),
            (),
            frozenset(
                {
                    "label",
                    "quiet",
                    "main.quiet",
                    "app.main.quiet",
                    "verbose",
                    "logging.verbose",
                    "A.logging.verbose",
                    "default-agent",
                    "default-sandbox",
                    "trace",
                    "trace-file",
                    "strict-json",
                    "timeout",
                }
            ),
            frozenset(),
        ),
    ]


def test_reports_a_module_root_before_its_scope_routes_even_without_root_params() -> None:
    scoped = _binding("A/logging", "trace", scope_path=("debug",))
    program = _program(closure=(ModuleId.from_path("app/main"), scoped.module))

    _values, reports = _resolve(_config(), program, (scoped,))

    assert reports[:3] == [
        RouteReport(("app", "main"), (), (), frozenset(), frozenset()),
        RouteReport(("A", "logging"), (), (), frozenset(), frozenset()),
        RouteReport(("A", "logging"), ("debug",), (), frozenset({"trace"}), frozenset()),
    ]


def test_program_report_marks_positional_only_signature_leaves() -> None:
    positional = ProgramParamInfo(
        name="input",
        kind=ParamZone.POSITIONAL_ONLY,
        type=TextType(),
        has_default=False,
        span=_SPAN,
        cli=_option("input"),
    )
    program = _program(parameters=(positional,))

    _values, reports = _resolve(_config(), program, ())

    assert reports[-1].positional_only == frozenset({"input"})
    assert "input" in reports[-1].declared_leaves


def test_undeclared_key_reporting_uses_every_returned_route(
    capsys: pytest.CaptureFixture[str],
) -> None:
    own = _binding("app/main", "quiet")
    imported = _binding("A/logging", "verbose")
    program = _program(closure=(own.module, imported.module))
    config = _config(
        {"A": {"logging": {"verbose": True, "misspelled": True}}},
        {"workflow": {"run": {"unexpected": True}}},
    )

    _values, reports = _resolve(config, program, (own, imported))
    _report_undeclared_config_keys(config, reports)

    reported = capsys.readouterr().err
    assert "misspelled" in reported
    assert "unexpected" in reported


def test_a_bare_leaf_a_signature_claims_draws_no_undeclared_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    binding = _binding("A/logging", "verbose")
    signature = ProgramParamInfo(
        name="verbose",
        kind=ParamZone.NAMED_ONLY,
        type=BoolType(),
        has_default=False,
        span=_SPAN,
        cli=_option("verbose"),
    )
    program = _program(
        closure=(ModuleId.from_path("app/main"), binding.module),
        parameters=(signature,),
    )
    config = _config({"workflow": {"run": {"verbose": True}}})

    _values, reports = _resolve(config, program, (binding,))
    _report_undeclared_config_keys(config, reports)

    assert capsys.readouterr().err == ""


def test_rejects_an_ambiguous_qualified_program_leaf_when_configured() -> None:
    first = _binding("first/logging", "verbose")
    second = _binding("second/logging", "verbose")
    program = _program(closure=(ModuleId.from_path("app/main"), first.module, second.module))

    with pytest.raises(QualifiedConfigLookupError) as exc_info:
        _resolve(
            _config({"workflow": {"run": {"logging.verbose": True}}}), program, (first, second)
        )

    assert first.declaration_path in str(exc_info.value)
    assert second.declaration_path in str(exc_info.value)


def test_positional_only_route_leaf_gets_its_distinct_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config({"workflow": {"run": {"positional": "value"}}})

    _report_undeclared_config_keys(
        config,
        (
            RouteReport(
                ("workflow",),
                ("run",),
                (),
                frozenset({"positional"}),
                frozenset({"positional"}),
            ),
        ),
    )

    assert "positional-only" in capsys.readouterr().err


def test_reports_surface_modules_missing_from_the_recorded_closure() -> None:
    binding = _binding("A/logging", "verbose")
    program = _program(closure=(ModuleId.from_path("app/main"),))

    _values, reports = _resolve(_config(), program, (binding,))

    assert RouteReport(("A", "logging"), (), (), frozenset({"verbose"}), frozenset()) in reports


def _resolve_tiers(
    config: GeneralConfig,
    program: ProgramDeclInfo,
    bindings: tuple[ParamBindingInfo, ...],
    params: Mapping[tuple[ModuleId, tuple[str, ...], str], object] = {},
) -> ParamValueTiers:
    tiers, _reports = resolve_param_values(
        config,
        program,
        params,
        entry_segments=("workflow",),
        command_paths=(),
        surface=_surface(program, bindings),
    )
    return tiers


def test_a_module_route_only_value_is_in_the_lower_tier() -> None:
    binding = _binding("A/logging", "verbose")
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    tiers = _resolve_tiers(_config({"A": {"logging": {"verbose": True}}}), program, (binding,))

    assert tiers.lower == {binding.key: True}
    assert tiers.upper == {}


def test_a_program_route_value_is_in_the_upper_tier_and_absent_from_lower() -> None:
    binding = _binding("A/logging", "verbose")
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    tiers = _resolve_tiers(
        _config({"workflow": {"run": {"logging.verbose": True}}}), program, (binding,)
    )

    assert tiers.upper == {binding.key: True}
    assert tiers.lower == {}


def test_a_program_route_value_shadows_a_configured_module_route_in_lower() -> None:
    binding = _binding("A/logging", "verbose")
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    tiers = _resolve_tiers(
        _config(
            {"workflow": {"run": {"logging.verbose": True}}},
            {"A": {"logging": {"verbose": False}}},
        ),
        program,
        (binding,),
    )

    assert tiers.upper == {binding.key: True}
    assert binding.key not in tiers.lower


def test_a_cli_supplied_value_is_in_the_upper_tier_and_shadows_a_module_route_in_lower() -> None:
    binding = _binding("A/logging", "verbose")
    program = _program(closure=(ModuleId.from_path("app/main"), binding.module))

    tiers = _resolve_tiers(
        _config({"A": {"logging": {"verbose": False}}}),
        program,
        (binding,),
        {binding.key: True},
    )

    assert tiers.upper == {binding.key: True}
    assert tiers.lower == {}


def test_supplied_and_program_route_tiers_combine_over_module_route() -> None:
    """``upper`` (program route) and ``lower`` (module route) partition disjoint
    keys here; the full precedence chain, including where ``@config`` ranks
    between them, is ``PipelineDriver.preflight_arguments``'s own merge."""
    module_only = _binding("A/logging", "verbose")
    program_routed = _binding("B/logging", "trace")
    program = _program(
        closure=(ModuleId.from_path("app/main"), module_only.module, program_routed.module)
    )

    tiers = _resolve_tiers(
        _config(
            {"A": {"logging": {"verbose": True}}},
            {"workflow": {"run": {"logging.trace": True}}},
        ),
        program,
        (module_only, program_routed),
    )

    assert tiers.upper == {program_routed.key: True}
    assert tiers.lower == {module_only.key: True}
