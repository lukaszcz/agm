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
    closure: tuple[ModuleId, ...] = (),
    parameters: tuple[ProgramParamInfo, ...] = (),
) -> ProgramDeclInfo:
    module_id = ModuleId.from_path(module) if isinstance(module, str) else module
    return ProgramDeclInfo(
        module=module_id,
        scope_path=(),
        name="run",
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
    return resolve_param_values(
        config,
        program,
        params,
        entry_segments=("workflow",),
        command_paths=(),
        surface=_surface(program, bindings),
    )


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


def test_ignores_a_nonbare_ambiguous_spelling_in_the_program_table() -> None:
    first = _binding("first/logging", "verbose")
    second = _binding("second/logging", "verbose")
    program = _program(
        closure=(ModuleId.from_path("app/main"), first.module, second.module),
    )

    values, _reports = _resolve(
        _config({"workflow": {"run": {"logging.verbose": True}}}), program, (first, second)
    )

    assert values == {}


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

    values, _reports = resolve_param_values(
        _config({"workflow": {"run": {"verbose": True}}}),
        program,
        {},
        entry_segments=(),
        command_paths=(),
        surface=_surface(program, (binding,)),
    )

    assert values == {}


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
                    "verbose",
                    "default-agent",
                    "log",
                    "log-file",
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
