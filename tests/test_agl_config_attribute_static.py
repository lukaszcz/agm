"""Static semantics of the ``@config`` program attribute.

``@config`` entries resolve their keys and values in the scope that declares
the ``program def`` (before its own parameter scope opens), then type check:
each key must name a root ``std/config`` engine setting or a whole-program
``@param`` binding, no two entries may name the same target, and each value
must be a constant expression assignable to the target's type.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.scope import AglScopeError
from agm.agl.scope.program import resolve_program
from agm.agl.typecheck.env import AglTypeError
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_file_graph_from_files
from tests.agl.module_graph import resolve_and_check_entry


def _check(source: str) -> None:
    """Resolve and type-check a single-module *source*, raising on any static error."""
    resolve_and_check_entry(source, base_caps())


def _check_multi(tmp_path: Path, modules: dict[str, str]) -> None:
    """Resolve and type-check a multi-module program, raising on any static error."""
    graph = make_file_graph_from_files(tmp_path, modules)
    resolved_program = resolve_program(graph)
    check_program(resolved_program, base_caps())


class TestAcceptedConfigTargets:
    def test_engine_setting_bool(self) -> None:
        _check(
            "import std/config\n\n@config(config::trace = true)\nprogram def main() -> unit = ()\n"
        )

    def test_engine_setting_option_text(self) -> None:
        _check(
            "import std/config\n\n"
            '@config(std/config::timeout = Some("30m"))\n'
            "program def main() -> unit = ()\n"
        )

    def test_engine_setting_agent(self) -> None:
        _check(
            "import std/config\n\n"
            '@config(config::default-agent = AgentClaude("opus", "high"))\n'
            "program def main() -> unit = ()\n"
        )

    def test_own_module_param_bare_name(self) -> None:
        _check("@param let count: int = 1\n\n@config(count = 2)\nprogram def main() -> unit = ()\n")

    def test_value_naming_this_modules_own_constant(self) -> None:
        _check(
            "@param let count: int = 1\n"
            "let other: int = 2\n\n"
            "@config(count = other)\n"
            "program def main() -> unit = ()\n"
        )

    def test_value_interpolating_this_modules_own_constant(self) -> None:
        _check(
            '@param let label: text = "plain"\n'
            'let suffix = "tail"\n\n'
            '@config(label = "head-%{suffix}")\n'
            "program def main() -> unit = ()\n"
        )

    def test_own_module_param_current_module_anchored_name(self) -> None:
        _check(
            "@param let count: int = 1\n\n@config(::count = 2)\nprogram def main() -> unit = ()\n"
        )

    def test_own_module_param_declared_name_not_opt_name(self) -> None:
        _check(
            '@param @opt-name("threshold") let count: int = 1\n\n'
            "@config(count = 2)\n"
            "program def main() -> unit = ()\n"
        )

    def test_own_module_param_declared_later_in_the_module(self) -> None:
        """A ``@param let`` is a static-root module-level binding, visible
        regardless of textual order like any other, so a ``@config`` entry
        may reference one declared after the ``program def``."""
        _check("@config(count = 1)\nprogram def main() -> unit = ()\n\n@param let count: int = 0\n")

    def test_scoped_param(self) -> None:
        _check(
            "scope Logging\n"
            "  @param let level: int = 1\n"
            "end Logging\n\n"
            "@config(Logging::level = 2)\n"
            "program def main() -> unit = ()\n"
        )

    def test_program_inside_a_scope_region(self) -> None:
        _check(
            "@param let count: int = 1\n\n"
            "scope Tools\n"
            "\n"
            "  @config(count = 2)\n"
            "  program def run() -> unit = ()\n"
            "end Tools\n\n"
            "program def main() -> unit = ()\n"
        )

    def test_enum_constructor_value(self) -> None:
        _check(
            "import std/agent\n\n"
            "@param let policy: ParsePolicy = ParsePolicy::Abort\n\n"
            "@config(policy = ParsePolicy::Retry(n = 3))\n"
            "program def main() -> unit = ()\n"
        )

    def test_negative_number_value(self) -> None:
        _check(
            "@param let offset: int = 0\n\n@config(offset = -5)\nprogram def main() -> unit = ()\n"
        )

    def test_array_value(self) -> None:
        _check(
            "@param let names: array[text] = []\n\n"
            '@config(names = ["a", "b"])\n'
            "program def main() -> unit = ()\n"
        )

    def test_dict_value(self) -> None:
        _check(
            "@param let counts: dict[text, int] = {}\n\n"
            '@config(counts = {"a": 1})\n'
            "program def main() -> unit = ()\n"
        )

    def test_imported_module_param_via_suffix_route(self, tmp_path: Path) -> None:
        _check_multi(
            tmp_path,
            {
                "entry": (
                    "import helper\n\n@config(helper::value = 2)\nprogram def main() -> unit = ()\n"
                ),
                "helper": "@param let value: int = 1\n",
            },
        )

    def test_imported_module_param_via_anchored_route(self, tmp_path: Path) -> None:
        _check_multi(
            tmp_path,
            {
                "entry": (
                    "import helper\n\n"
                    "@config(/helper::value = 2)\n"
                    "program def main() -> unit = ()\n"
                ),
                "helper": "@param let value: int = 1\n",
            },
        )

    def test_own_module_param_var(self) -> None:
        _check("@param var count: int = 1\n\n@config(count = 2)\nprogram def main() -> unit = ()\n")

    def test_program_names_its_own_scope_regions_param_bare(self) -> None:
        _check(
            "scope Tools\n"
            "  @param let level: int = 1\n\n"
            "  @config(level = 2)\n"
            "  program def run() -> unit = ()\n"
            "end Tools\n"
        )

    def test_imported_module_param_via_rename_spelling(self, tmp_path: Path) -> None:
        _check_multi(
            tmp_path,
            {
                "entry": (
                    "import helper::{value as v}\n\n"
                    "@config(v = 2)\n"
                    "program def main() -> unit = ()\n"
                ),
                "helper": "@param let value: int = 1\n",
            },
        )


class TestRejectedConfigTargets:
    def test_a_plain_var_is_not_a_legal_target(self) -> None:
        with pytest.raises(AglTypeError):
            _check("let count: int = 1\n\n@config(count = 2)\nprogram def main() -> unit = ()\n")

    def test_a_function_is_not_a_legal_target(self) -> None:
        with pytest.raises(AglTypeError):
            _check(
                "def helper() -> int = 1\n\n@config(helper = 2)\nprogram def main() -> unit = ()\n"
            )

    def test_a_constructor_is_not_a_legal_target(self) -> None:
        with pytest.raises(AglTypeError):
            _check("@config(Some = 2)\nprogram def main() -> unit = ()\n")

    def test_the_program_own_signature_parameter_is_unresolved(self) -> None:
        with pytest.raises(AglScopeError):
            _check("@config(count = 2)\nprogram def main(count: int) -> unit = ()\n")

    def test_an_undefined_key_is_unresolved(self) -> None:
        with pytest.raises(AglScopeError):
            _check("@config(unknown-name = 2)\nprogram def main() -> unit = ()\n")

    def test_an_opt_name_spelling_does_not_name_the_param(self) -> None:
        with pytest.raises(AglScopeError):
            _check(
                '@param @opt-name("threshold") let count: int = 1\n\n'
                "@config(threshold = 2)\n"
                "program def main() -> unit = ()\n"
            )

    def test_duplicate_target_via_two_spellings_is_rejected(self) -> None:
        with pytest.raises(AglTypeError):
            _check(
                "@param let count: int = 1\n\n"
                "@config(count = 2, ::count = 3)\n"
                "program def main() -> unit = ()\n"
            )

    def test_a_non_constant_call_value_is_rejected(self) -> None:
        with pytest.raises(AglTypeError):
            _check(
                "@param let count: int = 1\n"
                "def helper() -> int = 2\n\n"
                "@config(count = helper())\n"
                "program def main() -> unit = ()\n"
            )

    def test_a_reference_to_a_non_constant_binding_is_rejected(self) -> None:
        with pytest.raises(AglTypeError):
            _check(
                "@param let count: int = 1\n"
                "def helper() -> int = 2\n"
                "let other: int = helper()\n\n"
                "@config(count = other)\n"
                "program def main() -> unit = ()\n"
            )

    def test_a_wrong_typed_value_is_rejected(self) -> None:
        with pytest.raises(AglTypeError):
            _check(
                "@param let count: int = 1\n\n"
                '@config(count = "not a number")\n'
                "program def main() -> unit = ()\n"
            )

    def test_an_engine_setting_wrong_typed_value_is_rejected(self) -> None:
        with pytest.raises(AglTypeError):
            _check(
                "import std/config\n\n@config(config::trace = 1)\nprogram def main() -> unit = ()\n"
            )

    def test_a_non_engine_builtin_var_is_not_a_legal_target(self) -> None:
        with pytest.raises(AglTypeError):
            _check("import std/env\n\n@config(env::environ = 0)\nprogram def main() -> unit = ()\n")

    def test_a_non_engine_builtin_var_is_not_a_legal_target_via_use_bare(self) -> None:
        with pytest.raises(AglTypeError):
            _check(
                "import std/env\nuse std/env::environ\n\n"
                "@config(environ = 0)\n"
                "program def main() -> unit = ()\n"
            )

    def test_duplicate_target_via_two_engine_spellings_is_rejected(self) -> None:
        with pytest.raises(AglTypeError):
            _check(
                "import std/config\nuse std/config::trace\n\n"
                "@config(trace = true, config::trace = false)\n"
                "program def main() -> unit = ()\n"
            )

    def test_duplicate_cross_module_target_via_two_spellings_is_rejected(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(AglTypeError):
            _check_multi(
                tmp_path,
                {
                    "entry": (
                        "import helper\n"
                        "import helper::{value as v}\n\n"
                        "@config(helper::value = 2, v = 3)\n"
                        "program def main() -> unit = ()\n"
                    ),
                    "helper": "@param let value: int = 1\n",
                },
            )

    def test_a_body_local_let_is_not_in_scope_for_a_key(self) -> None:
        with pytest.raises(AglScopeError):
            _check(
                "def helper() -> int =\n"
                "  let x: int = 1\n"
                "  x\n\n"
                "@config(x = 2)\n"
                "program def main() -> unit = ()\n"
            )

    def test_a_type_name_is_not_a_legal_key(self) -> None:
        with pytest.raises(AglScopeError):
            _check(
                "enum Color =\n"
                "  | Red\n"
                "  | Green\n\n"
                "@config(Color = 2)\n"
                "program def main() -> unit = ()\n"
            )
