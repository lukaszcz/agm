"""Tests for declaration-attribute recognition in the AgL scope pass.

Every attribute a declaration carries is validated against the built-in
attribute catalog while names are resolved, and the surviving attributes are
turned into the resolved program's fact tables: ``param_zones`` from the
``@arg-*`` attributes, ``program_options`` and ``params`` from the host-facing
parameter attributes, and ``docs`` from ``@doc``. These tests assert on those
tables' contents and pin the phase the rejections come from; the rejection fixtures under
``tests/agl/rejections/scope/`` cover the same diagnostics as whole programs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.attributes import ProgramCommandSpec, ProgramOptionSpec
from agm.agl.parser.parser import parse_program_seeded
from agm.agl.scope import AglScopeError, ModuleResolution, recognize_program_command
from agm.agl.scope.attributes import recognize_attributes
from agm.agl.syntax.nodes import (
    Attribute,
    BuiltinVarDecl,
    EnumDef,
    ExceptionDef,
    FuncDef,
    Lambda,
    LetDecl,
    Param,
    RecordDef,
    ScopeRegion,
    TypeAlias,
    VarDecl,
    VariantDef,
    static_function_items,
)
from agm.agl.syntax.visitor import walk
from agm.agl.zones import ParamZone
from tests.agl.module_graph import resolve_entry

POSITIONAL_ONLY = ParamZone.POSITIONAL_ONLY
STANDARD = ParamZone.STANDARD
NAMED_ONLY = ParamZone.NAMED_ONLY
AGL_FIXTURES = Path(__file__).parent / "agl"


def _entries(resolution: ModuleResolution, owner_name: str) -> tuple[Param, ...]:
    """Return the parameter or field list of the named declaration."""
    found: list[tuple[Param, ...]] = []

    def visit(node: object) -> None:
        if isinstance(node, FuncDef) and node.name == owner_name:
            found.append(node.params)
        elif isinstance(node, (RecordDef, ExceptionDef, VariantDef)) and node.name == owner_name:
            found.append(node.fields)

    walk(resolution.program, visit)
    assert len(found) == 1, f"expected exactly one declaration named {owner_name!r}"
    return found[0]


def _program(resolution: ModuleResolution, name: str) -> FuncDef:
    """Return the uniquely named function declaration of a resolved module."""
    found: list[FuncDef] = []

    def visit(node: object) -> None:
        if isinstance(node, FuncDef) and node.name == name:
            found.append(node)

    walk(resolution.program, visit)
    assert len(found) == 1, f"expected exactly one declaration named {name!r}"
    return found[0]


def _zones(resolution: ModuleResolution, owner_name: str) -> dict[str, ParamZone]:
    """Return the resolved zone of every entry of the named declaration."""
    return {
        entry.name: resolution.attributes.param_zones[entry.node_id]
        for entry in _entries(resolution, owner_name)
    }


def _lambda_zones(resolution: ModuleResolution) -> dict[str, ParamZone]:
    """Return the resolved zones of the single lambda in the program."""
    lambdas: list[Lambda] = []

    def visit(node: object) -> None:
        if isinstance(node, Lambda):
            lambdas.append(node)

    walk(resolution.program, visit)
    assert len(lambdas) == 1
    return {p.name: resolution.attributes.param_zones[p.node_id] for p in lambdas[0].params}


def _option(resolution: ModuleResolution, owner_name: str, param_name: str) -> ProgramOptionSpec:
    """Return the option spec recognized for one program parameter."""
    entries = {entry.name: entry for entry in _entries(resolution, owner_name)}
    return resolution.attributes.program_options[entries[param_name].node_id]


def _static_binding(resolution: ModuleResolution, name: str) -> LetDecl | VarDecl:
    """Return the uniquely named static binding, including scope-region members."""
    found: list[LetDecl | VarDecl] = []

    def visit_items(items: tuple[object, ...]) -> None:
        for item in items:
            if isinstance(item, ScopeRegion):
                visit_items(item.items)
            elif isinstance(item, VarDecl) and item.name == name:
                found.append(item)
            elif isinstance(item, LetDecl):
                if item.name == name:
                    found.append(item)

    visit_items(resolution.program.body.items)
    assert len(found) == 1, f"expected exactly one static binding named {name!r}"
    return found[0]


def _binding_node_id(binding: LetDecl | VarDecl) -> int:
    """Return the static binding identity used by scope and typecheck."""
    return binding.node_id


class TestParameterZones:
    """Per-entry ``@arg-*`` attributes and the form defaults they override."""

    def test_a_function_defaults_every_parameter_to_standard(self) -> None:
        resolution = resolve_entry("def f(a: int, b: int) -> int = a + b\n")

        assert _zones(resolution, "f") == {"a": STANDARD, "b": STANDARD}

    def test_a_parameter_attribute_sets_that_entrys_zone(self) -> None:
        resolution = resolve_entry(
            "def f(@arg-pos a: int, b: int, @arg-named c: int) -> int = a + b + c\n"
        )

        assert _zones(resolution, "f") == {
            "a": POSITIONAL_ONLY,
            "b": STANDARD,
            "c": NAMED_ONLY,
        }

    def test_a_declaration_attribute_sets_the_list_default(self) -> None:
        resolution = resolve_entry("@arg-named\ndef f(a: int, b: int) -> int = a + b\n")

        assert _zones(resolution, "f") == {"a": NAMED_ONLY, "b": NAMED_ONLY}

    def test_an_entry_attribute_overrides_the_declaration_default(self) -> None:
        resolution = resolve_entry("@arg-named\ndef f(@arg-pos a: int, b: int) -> int = a + b\n")

        assert _zones(resolution, "f") == {"a": POSITIONAL_ONLY, "b": NAMED_ONLY}

    def test_a_program_defaults_its_parameters_to_named_only(self) -> None:
        resolution = resolve_entry("program def main(a: int) -> unit = print a\n")

        assert _zones(resolution, "main") == {"a": NAMED_ONLY}

    def test_a_program_declaration_attribute_overrides_the_named_only_default(self) -> None:
        resolution = resolve_entry("@arg-pos\nprogram def main(a: int) -> unit = print a\n")

        assert _zones(resolution, "main") == {"a": POSITIONAL_ONLY}

    def test_a_receiver_is_positional_only(self) -> None:
        resolution = resolve_entry(
            "record Point\n  x: int\n\ndef Point::shifted(self, by: int) -> int = self.x + by\n"
        )

        assert _zones(resolution, "shifted") == {"self": POSITIONAL_ONLY, "by": STANDARD}

    def test_a_receiver_sits_ahead_of_a_positional_only_parameter(self) -> None:
        resolution = resolve_entry(
            "record Point\n"
            "  x: int\n"
            "\n"
            "def Point::shifted(self, @arg-pos by: int) -> int = self.x + by\n"
        )

        assert _zones(resolution, "shifted") == {
            "self": POSITIONAL_ONLY,
            "by": POSITIONAL_ONLY,
        }

    def test_a_zone_attribute_is_found_past_an_unrelated_one(self) -> None:
        resolution = resolve_entry('def f(@doc("a") @arg-named a: int) -> int = a\n')

        assert _zones(resolution, "f") == {"a": NAMED_ONLY}

    def test_a_lambda_parameter_carries_its_own_zone(self) -> None:
        resolution = resolve_entry(
            "def f() -> int =\n"
            "  let g = fn(a: int, @arg-named b: int) -> int => a + b\n"
            "  g(1, b = 2)\n"
        )

        assert _lambda_zones(resolution) == {"a": STANDARD, "b": NAMED_ONLY}


class TestFieldZones:
    """Record, exception and enum-member field lists follow the same rules."""

    def test_record_fields_default_to_standard(self) -> None:
        resolution = resolve_entry("record R\n  x: int\n  y: int\n")

        assert _zones(resolution, "R") == {"x": STANDARD, "y": STANDARD}

    def test_a_record_attribute_sets_its_field_default(self) -> None:
        resolution = resolve_entry("@arg-named\nrecord R\n  @arg-std x: int\n  y: int\n")

        assert _zones(resolution, "R") == {"x": STANDARD, "y": NAMED_ONLY}

    def test_an_exception_attribute_sets_its_field_default(self) -> None:
        resolution = resolve_entry(
            "@arg-named\nexception Failure extends Exception\n  detail: text\n"
        )

        assert _zones(resolution, "Failure") == {"detail": NAMED_ONLY}

    def test_an_enum_member_attribute_sets_its_field_default(self) -> None:
        resolution = resolve_entry(
            "enum Shape =\n  | @arg-named Circle(radius: int)\n  | Square(side: int)\n"
        )

        assert _zones(resolution, "Circle") == {"radius": NAMED_ONLY}
        assert _zones(resolution, "Square") == {"side": STANDARD}


class TestAttributeDiagnostics:
    """Diagnostics that need no whole-program pipeline to observe."""

    def test_an_unknown_attribute_is_rejected(self) -> None:
        with pytest.raises(AglScopeError, match="nonesuch"):
            resolve_entry("@nonesuch\ndef f() -> int = 1\n")

    def test_an_attribute_on_a_disallowed_target_is_rejected(self) -> None:
        with pytest.raises(AglScopeError, match="opt-name"):
            resolve_entry('@opt-name("x")\nrecord R\n  x: int\n')

    def test_an_enum_declaration_admits_a_doc_attribute(self) -> None:
        resolution = resolve_entry('@doc("shapes")\nenum Shape =\n  | Dot\n  | Dash\n')

        enums = [item for item in resolution.program.body.items if isinstance(item, EnumDef)]
        assert [enum.name for enum in enums] == ["Shape"]
        assert resolution.attributes.docs[enums[0].node_id] == "shapes"

    def test_a_zone_attribute_on_an_enum_declaration_is_rejected(self) -> None:
        with pytest.raises(AglScopeError, match="arg-named"):
            resolve_entry("@arg-named\nenum Shape =\n  | Dot\n  | Dash\n")

    def test_a_binding_admits_a_doc_attribute(self) -> None:
        resolution = resolve_entry('@doc("the answer")\nlet answer = 42\n')

        lets = [item for item in resolution.program.body.items if isinstance(item, LetDecl)]
        assert len(lets) == 1
        assert resolution.attributes.docs[lets[0].node_id] == "the answer"

    def test_a_builtin_var_admits_a_doc_attribute(self) -> None:
        program, _next_id = parse_program_seeded(
            '@doc("the setting")\nbuiltin var setting: int = 1\n',
            start_id=0,
            resolve_infix=False,
        )
        declaration = program.body.items[0]
        assert isinstance(declaration, BuiltinVarDecl)

        facts = recognize_attributes(program, declares_receiver=lambda _node: False)

        assert facts.docs[declaration.node_id] == "the setting"

    @pytest.mark.parametrize(
        ("fixture", "attribute_name", "occurrence"),
        [
            ("rejections/scope/param_nested_binder.agl", "param", 0),
            ("rejections/scope/param_wildcard.agl", "param", 0),
            (
                "program_modules/param_on_builtin_var_stdlib/src/prelude.agl",
                "param",
                0,
            ),
            ("rejections/scope/opt_name_without_param.agl", "opt-name", 0),
            ("rejections/scope/param_duplicate_attribute.agl", "param", 1),
        ],
        ids=(
            "nested-binder",
            "wildcard",
            "builtin-var",
            "option-without-param",
            "duplicate-param",
        ),
    )
    def test_param_rejections_point_at_the_exact_attribute_span(
        self,
        fixture: str,
        attribute_name: str,
        occurrence: int,
    ) -> None:
        source = (AGL_FIXTURES / fixture).read_text(encoding="utf-8")
        program, _next_id = parse_program_seeded(source, start_id=0, resolve_infix=False)
        attributes: list[Attribute] = []

        def collect(node: object) -> None:
            if isinstance(node, Attribute) and node.name == attribute_name:
                attributes.append(node)

        walk(program, collect)

        with pytest.raises(AglScopeError) as raised:
            recognize_attributes(program, declares_receiver=lambda _node: False)

        assert raised.value.span == attributes[occurrence].span

    def test_a_zone_attribute_on_a_binding_is_rejected(self) -> None:
        with pytest.raises(AglScopeError, match="arg-pos"):
            resolve_entry("@arg-pos\nlet answer = 42\n")

    def test_a_zone_attribute_out_of_order_is_rejected(self) -> None:
        with pytest.raises(AglScopeError, match="ordered"):
            resolve_entry("def f(a: int, @arg-pos b: int) -> int = a + b\n")

    def test_a_zone_attribute_on_a_receiver_is_rejected(self) -> None:
        with pytest.raises(AglScopeError, match="shifted"):
            resolve_entry(
                "record Point\n"
                "  x: int\n"
                "\n"
                "def Point::shifted(@arg-std self, by: int) -> int = self.x + by\n"
            )

    def test_a_type_alias_admits_a_doc_attribute(self) -> None:
        resolution = resolve_entry('@doc("a count")\ntype Count = int\n')

        aliases = [item for item in resolution.program.body.items if isinstance(item, TypeAlias)]
        assert len(aliases) == 1
        assert resolution.attributes.docs[aliases[0].node_id] == "a count"
        assert resolution.attributes.param_zones == {}


class TestProgramOptions:
    """The command-line presentation the ``@opt-*`` attributes describe."""

    def test_a_parameter_without_attributes_is_addressed_by_its_declared_name(self) -> None:
        resolution = resolve_entry("program def main(count: int = 0) -> unit = print count\n")

        assert _option(resolution, "main", "count") == ProgramOptionSpec(name="count")

    def test_every_option_attribute_reaches_the_spec(self) -> None:
        resolution = resolve_entry(
            "program def main(\n"
            '  @doc("how many runs")\n'
            '  @opt-name("total-runs")\n'
            '  @opt-short("r")\n'
            '  @opt-env("TOTAL_RUNS")\n'
            '  @opt-metavar("N")\n'
            "  @opt-hidden\n"
            "  runs: int = 0,\n"
            ") -> unit = print runs\n"
        )

        assert _option(resolution, "main", "runs") == ProgramOptionSpec(
            name="total-runs",
            short="r",
            env="TOTAL_RUNS",
            metavar="N",
            hidden=True,
            doc="how many runs",
        )

    def test_a_positional_only_parameter_admits_a_metavar_and_documentation(self) -> None:
        resolution = resolve_entry(
            "program def main(\n"
            '  @arg-pos @opt-metavar("N") @doc("how many runs") runs: int,\n'
            ") -> unit = print runs\n"
        )

        assert _option(resolution, "main", "runs") == ProgramOptionSpec(
            name="runs", metavar="N", doc="how many runs"
        )

    @pytest.mark.parametrize(
        "attribute",
        ['@opt-name("total")', '@opt-short("r")', '@opt-env("RUNS")', "@opt-hidden"],
    )
    def test_a_name_addressed_attribute_on_a_positional_only_parameter_is_rejected(
        self, attribute: str
    ) -> None:
        with pytest.raises(AglScopeError, match="positional-only"):
            resolve_entry(
                f"program def main(@arg-pos {attribute} runs: int) -> unit = print runs\n"
            )

    @pytest.mark.parametrize("short", ["", "rr", "1", "-", " "])
    def test_a_short_option_that_is_not_one_letter_is_rejected(self, short: str) -> None:
        with pytest.raises(AglScopeError, match="opt-short"):
            resolve_entry(
                f'program def main(@opt-short("{short}") runs: int = 0) -> unit = print runs\n'
            )

    @pytest.mark.parametrize(
        "name",
        ["", "--runs", "-runs", "-", "total runs", "runs=1", "a.b", "runs-", "runs--", "to--tal"],
    )
    def test_an_option_name_that_is_not_a_flag_word_is_rejected(self, name: str) -> None:
        with pytest.raises(AglScopeError, match="flag word"):
            resolve_entry(
                f'program def main(@opt-name("{name}") runs: int = 0) -> unit = print runs\n'
            )

    @pytest.mark.parametrize("name", ["runs", "total-runs", "R", "2nd-run", "a-b-c"])
    def test_a_flag_word_of_letters_digits_and_hyphens_is_accepted(self, name: str) -> None:
        resolution = resolve_entry(
            f'program def main(@opt-name("{name}") runs: int = 0) -> unit = print runs\n'
        )

        assert _option(resolution, "main", "runs").name == name

    def test_an_ordinary_functions_parameter_admits_no_option_attribute(self) -> None:
        with pytest.raises(AglScopeError, match="opt-name"):
            resolve_entry(
                'def f(@opt-name("value") a: int) -> int = a\n'
                "\n"
                "program def main() -> unit = print f(1)\n"
            )

    def test_only_program_parameters_have_option_specs(self) -> None:
        resolution = resolve_entry(
            "def f(a: int) -> int = a\n\nprogram def main() -> unit = print f(1)\n"
        )

        assert resolution.attributes.program_options == {}


class TestParamBindings:
    """Host presentation facts for static bindings carrying ``@param``."""

    def test_a_param_defaults_to_its_declared_name(self) -> None:
        resolution = resolve_entry("@param let count = 1\n")
        binding = _static_binding(resolution, "count")

        assert resolution.attributes.params[_binding_node_id(binding)] == ProgramOptionSpec(
            name="count"
        )

    def test_every_presentation_attribute_reaches_the_param_spec(self) -> None:
        resolution = resolve_entry(
            '@param @doc("how many runs") @opt-name("total-runs") '
            '@opt-short("r") @opt-env("TOTAL_RUNS") @opt-metavar("N") '
            "@opt-hidden let runs = 0\n"
        )
        binding = _static_binding(resolution, "runs")

        assert resolution.attributes.params[_binding_node_id(binding)] == ProgramOptionSpec(
            name="total-runs",
            short="r",
            env="TOTAL_RUNS",
            metavar="N",
            hidden=True,
            doc="how many runs",
        )
        assert resolution.attributes.docs[binding.node_id] == "how many runs"

    def test_a_scope_region_param_is_recorded(self) -> None:
        resolution = resolve_entry(
            "scope logging\n"
            "\n"
            "  scope debug\n"
            "    @param let trace = false\n"
            "  end debug\n"
            "end logging\n"
        )
        binding = _static_binding(resolution, "trace")

        assert resolution.attributes.params[_binding_node_id(binding)] == ProgramOptionSpec(
            name="trace"
        )

    def test_a_var_param_is_recorded(self) -> None:
        resolution = resolve_entry("@param var level = 1\n")
        binding = _static_binding(resolution, "level")

        assert resolution.attributes.params[_binding_node_id(binding)] == ProgramOptionSpec(
            name="level"
        )

    def test_an_ordinary_binding_has_no_param_fact(self) -> None:
        resolution = resolve_entry("let count = 1\n")

        assert resolution.attributes.params == {}


class TestDocumentationTexts:
    """``@doc`` text, keyed by the declaration that carries it."""

    def test_a_function_and_its_parameter_are_documented_independently(self) -> None:
        resolution = resolve_entry(
            '@doc("adds two numbers")\ndef add(@doc("the first") a: int, b: int) -> int = a + b\n'
        )

        functions = [item for item in resolution.program.body.items if isinstance(item, FuncDef)]
        entries = {entry.name: entry for entry in _entries(resolution, "add")}
        assert resolution.attributes.docs[functions[0].node_id] == "adds two numbers"
        assert resolution.attributes.docs[entries["a"].node_id] == "the first"
        assert entries["b"].node_id not in resolution.attributes.docs

    def test_a_record_field_is_documented(self) -> None:
        resolution = resolve_entry('record R\n  @doc("across")\n  x: int\n')

        entries = {entry.name: entry for entry in _entries(resolution, "R")}
        assert resolution.attributes.docs[entries["x"].node_id] == "across"

    def test_an_undocumented_program_has_no_entry(self) -> None:
        resolution = resolve_entry("program def main() -> unit = ()\n")

        assert resolution.attributes.docs == {}


class TestCommandRegistrations:
    """``@command``/``@description``/``@help``, keyed by the program they register."""

    def test_every_command_attribute_reaches_the_registration(self) -> None:
        resolution = resolve_entry(
            '@command("devel review")\n'
            '@description("Review changes")\n'
            '@help("This program reviews changes")\n'
            '@doc("Change review")\n'
            "program def main() -> unit = ()\n"
        )

        program = _program(resolution, "main")
        assert resolution.attributes.command_registrations[program.node_id] == ProgramCommandSpec(
            path="devel review",
            description="Review changes",
            help="This program reviews changes",
        )

    def test_documentation_stays_separate_from_the_registration(self) -> None:
        resolution = resolve_entry(
            '@command("audit")\n@doc("Change review")\nprogram def main() -> unit = ()\n'
        )

        program = _program(resolution, "main")
        assert resolution.attributes.docs[program.node_id] == "Change review"
        assert resolution.attributes.command_registrations[program.node_id].description is None

    def test_a_command_without_prose_carries_only_its_path(self) -> None:
        resolution = resolve_entry('@command("audit")\nprogram def main() -> unit = ()\n')

        program = _program(resolution, "main")
        assert resolution.attributes.command_registrations[program.node_id] == ProgramCommandSpec(
            path="audit"
        )

    def test_an_unregistered_program_has_no_entry(self) -> None:
        resolution = resolve_entry("program def main() -> unit = ()\n")

        assert resolution.attributes.command_registrations == {}

    def test_a_program_in_a_scope_region_is_registered(self) -> None:
        resolution = resolve_entry(
            "scope Tools\n"
            "\n"
            '  @command("devel review")\n'
            "  program def run() -> unit = ()\n"
            "end Tools\n"
            "\n"
            "program def main() -> unit = ()\n"
        )

        program = _program(resolution, "run")
        registration = resolution.attributes.command_registrations[program.node_id]
        assert registration.path == "devel review"

    @pytest.mark.parametrize("path", ("", "   ", "devel  review", "devel\treview"))
    def test_a_path_that_is_not_space_separated_words_is_rejected(self, path: str) -> None:
        with pytest.raises(AglScopeError, match="Command path"):
            resolve_entry(f'@command("{path}")\nprogram def main() -> unit = ()\n')

    def test_a_path_beginning_with_a_reserved_command_is_rejected(self) -> None:
        with pytest.raises(AglScopeError, match="reserved"):
            resolve_entry('@command("exec review")\nprogram def main() -> unit = ()\n')

    @pytest.mark.parametrize("attribute", ("description", "help"))
    def test_prose_without_a_command_is_rejected(self, attribute: str) -> None:
        with pytest.raises(AglScopeError, match="command"):
            resolve_entry(f'@{attribute}("text")\nprogram def main() -> unit = ()\n')

    def test_an_ordinary_function_admits_no_command_attribute(self) -> None:
        with pytest.raises(AglScopeError, match="command"):
            resolve_entry(
                '@command("audit")\ndef helper() -> unit = ()\nprogram def main() -> unit = ()\n'
            )


class TestParseOnlyCommandRecognition:
    """``recognize_program_command`` validates a declaration header alone.

    Package command discovery (``agm.packages.source_commands``) calls it on
    a parsed-but-unresolved program, so it must raise the exact diagnostics
    the full scope walk raises for the same source — the two share one
    implementation of the command-registration rules.
    """

    @staticmethod
    def _program(source: str, name: str = "main") -> FuncDef:
        program, _next_id = parse_program_seeded(source, start_id=0, resolve_infix=False)
        (function,) = (
            item for item in static_function_items(program.body.items) if item.name == name
        )
        return function

    def test_recognizes_a_full_registration_without_resolving_the_program(self) -> None:
        function = self._program(
            '@command("devel review")\n'
            '@description("Review changes")\n'
            '@help("This program reviews changes")\n'
            "program def main() -> unit = ()\n"
        )

        assert recognize_program_command(function) == ProgramCommandSpec(
            path="devel review",
            description="Review changes",
            help="This program reviews changes",
        )

    def test_returns_none_for_an_unregistered_program(self) -> None:
        function = self._program("program def main() -> unit = ()\n")

        assert recognize_program_command(function) is None

    def test_rejects_a_malformed_path_like_full_scope_resolution(self) -> None:
        function = self._program('@command(" bad")\nprogram def main() -> unit = ()\n')

        with pytest.raises(AglScopeError, match="Command path"):
            recognize_program_command(function)

    def test_rejects_prose_without_command_like_full_scope_resolution(self) -> None:
        function = self._program('@description("text")\nprogram def main() -> unit = ()\n')

        with pytest.raises(AglScopeError, match="command"):
            recognize_program_command(function)
