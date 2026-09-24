"""Tests for ``CheckedModuleImage``: image/rehydrate parity and journal scope.

Builds one multi-module program (generics, aliases, an enum with a method, a
scoped record and function, extern declarations (one type-directed), a self-method, a
cross-module candidate-function dependency, and a static ``let``/``var`` pair with
an unannotated ``let``), compiles it once, then checks that turning each
non-entry module into a ``CheckedModuleImage``
and rehydrating it onto a freshly prepared environment reproduces every query
answer the original module gives -- and that a program reassembled from
rehydrated modules lowers to the identical ``ExecutableProgram``.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

from agm.agl import artifact_serialization
from agm.agl.artifact_cache import clear_retained_artifacts
from agm.agl.capabilities import HostCapabilities
from agm.agl.lower.program import lower_program
from agm.agl.matchcompile import compile_program_matches
from agm.agl.modules.ids import ModuleId
from agm.agl.scope.program import ResolvedProgram, resolve_program
from agm.agl.semantics.types import Type
from agm.agl.syntax.nodes import (
    LetDecl,
    VarDecl,
    static_binding_node_id,
    static_function_items,
    static_items,
    static_type_items,
)
from agm.agl.typecheck import env as env_module
from agm.agl.typecheck.env import (
    ArgumentBindings,
    CheckedModule,
    CheckedModuleImage,
    EnvironmentFacts,
    ModuleTypeInterface,
    PublishedModuleSurface,
    TypeEnvironment,
)
from agm.agl.typecheck.function_inference import FunctionSignatureRecord
from agm.agl.typecheck.program import CheckedProgram, _prepare_program, check_program
from tests.agl.ir_harness import (
    base_caps,
    extern_caps,
    make_file_graph_from_files,
    write_companion_file,
)
from tests.agl.module_graph import resolve_and_check_repl_entry

_GENERICS_SRC = """\
record Box[T]
  value: T

def make-box[T](value: T) -> Box[T] = Box(value = value)

def print-box-value(value: int) -> unit = print::[int](value)
"""

_ALIAS_SRC = """\
type Seq[A] = array[A]
type Count = int

def first-of[A](values: Seq[A]) -> A = values[0]

def wrap-count(value: int) -> Count = value
"""

_ENUM_SRC = """\
enum Signal
  | idle
  | number(value: int)

def Signal::describe(self) -> text =
  case self of
    | idle => "idle"
    | number(value) => "number %{value}"

def Signal::is-idle(self) -> bool = self is idle

record SignalBox
  signal: Signal

def unbox-label(box: SignalBox) -> text =
  case box of
    | SignalBox(signal = idle) => "idle"
    | SignalBox(signal = number(value)) => "number %{value}"
"""

_SCOPE_SRC = """\
scope Region
  record Marker
    value: int
  def scoped-value(marker: Marker) -> int = marker.value
end Region

def region-example() -> int = Region::scoped-value(Region::Marker(value = 5))

def region-exec-example() -> text = exec("ls", on-parse-error = Abort())
"""

_EXTERN_SRC = """\
extern def double_value(value: int) -> int

def quadruple(value: int) -> int = double_value(double_value(value))

def quadruple-fn() -> (int) -> int = quadruple(?)

extern def convert[T](value: int) -> T

def convert-to-text(value: int) -> text = convert(value)
"""

_EXTERN_COMPANION_SRC = (
    "def double_value(value):\n    return value * 2\ndef convert(value):\n    return str(value)\n"
)

_METHODS_SRC = """\
record Point
  x: int
  y: int

def Point::sum(self) -> int = self.x + self.y

def sum-of-origin() -> int = Point(x = 0, y = 0).sum()
"""

_CANDIDATE_CALLEE_SRC = "def factorial(n: int) = if n <= 1 => 1 else => n * factorial(n - 1)\n"

_CANDIDATE_CONSUMER_SRC = """\
import lib_candidate_callee::*

def increment-factorial(n: int) -> int = factorial(n) + 1
"""

_STATIC_LET_SRC = """\
let default-count: int = 7
var default-attempts: int = 0
let default-label = "starter"

def default-count-plus-one() -> int = default-count + 1

def default-count-as-decimal() -> decimal = default-count as decimal

def bump-attempts() -> int =
  default-attempts := default-attempts + 1
  default-attempts

def shout-default-label() -> text = default-label

@param let default-threshold: int = 5

@config(default-threshold = 9)
program def bump-threshold() -> unit = ()
"""

_ENTRY_SRC = """\
import lib_generics::*
import lib_alias::*
import lib_enum::*
import lib_scope::*
import lib_extern::*
import lib_methods::*
import lib_candidate_consumer::*
import lib_static_let::*

def summarize() -> int =
  let box = make-box(3)
  let seq: Seq[int] = [1, 2, 3]
  let first = first-of(seq)
  let signal: Signal = Signal::number(value = 2)
  let _ = signal.describe()
  let scoped = region-example()
  let point = Point(x = 1, y = 2)
  let total = point.sum()
  let inc = increment-factorial(4)
  let base = default-count-plus-one()
  box.value + first + scoped + total + inc + base
"""

_MODULES = {
    "entry": _ENTRY_SRC,
    "lib_generics": _GENERICS_SRC,
    "lib_alias": _ALIAS_SRC,
    "lib_enum": _ENUM_SRC,
    "lib_scope": _SCOPE_SRC,
    "lib_extern": _EXTERN_SRC,
    "lib_methods": _METHODS_SRC,
    "lib_candidate_callee": _CANDIDATE_CALLEE_SRC,
    "lib_candidate_consumer": _CANDIDATE_CONSUMER_SRC,
    "lib_static_let": _STATIC_LET_SRC,
}


@dataclasses.dataclass(frozen=True, slots=True)
class _Compiled:
    resolved: ResolvedProgram
    checked: CheckedProgram
    caps: HostCapabilities


@pytest.fixture(scope="module")
def compiled(tmp_path_factory: pytest.TempPathFactory) -> _Compiled:
    """Compile the fixture program fresh, with no retained-artifact reuse."""
    tmp_path = tmp_path_factory.mktemp("checked_module_image")
    write_companion_file(tmp_path / "root", "lib_extern", _EXTERN_COMPANION_SRC)
    caps = dataclasses.replace(extern_caps(), supports_shell_exec=True)
    clear_retained_artifacts()
    graph = make_file_graph_from_files(tmp_path, _MODULES, default_stdlib=True)
    resolved = resolve_program(graph)
    checked = check_program(resolved, caps)
    return _Compiled(resolved=resolved, checked=checked, caps=caps)


@pytest.fixture(scope="module")
def rehydrated(compiled: _Compiled) -> dict[ModuleId, CheckedModule]:
    """Rehydrate every non-entry module onto a second, freshly prepared pass.

    ``_prepare_program`` is called with no cached modules, so every returned
    environment is exactly what a cache-free compilation would have prepared
    -- the seam :meth:`CheckedModuleImage.rehydrate` is meant to be used from.
    """
    prepared = _prepare_program(compiled.resolved, compiled.caps)
    result: dict[ModuleId, CheckedModule] = {}
    for mid, rmod in compiled.resolved.modules.items():
        if mid == compiled.resolved.entry_id:
            continue
        image = compiled.checked.modules[mid].image()
        result[mid] = image.rehydrate(rmod, prepared.module_envs[mid])
    return result


_FIXTURE_LIB_IDS = {ModuleId.from_path(name) for name in _MODULES if name != "entry"}


def _non_entry_module_ids(compiled: _Compiled) -> list[ModuleId]:
    """This fixture's own dependency modules, excluding the entry and the stdlib.

    ``default_stdlib=True`` (required for ``exec``, see ``compiled``) pulls the
    whole standard library into ``compiled.resolved.modules`` too; scoping to
    the modules this file declares keeps parity/serialization loops cheap and
    decoupled from unrelated stdlib content.
    """
    return [mid for mid in compiled.resolved.modules if mid in _FIXTURE_LIB_IDS]


class TestRehydrationParity:
    """Every declaration query a rehydrated module answers matches the original."""

    def test_function_signature_and_extern_queries_match(
        self, compiled: _Compiled, rehydrated: dict[ModuleId, CheckedModule]
    ) -> None:
        for mid in _non_entry_module_ids(compiled):
            rehydrated_module = rehydrated[mid]
            cm = compiled.checked.modules[mid]
            for item in static_function_items(cm.resolved.program.body.items):
                assert rehydrated_module.type_env.get_function_signature_by_node_id(
                    item.node_id
                ) == cm.type_env.get_function_signature_by_node_id(item.node_id)
                assert rehydrated_module.type_env.is_extern_node_id(
                    item.node_id
                ) == cm.type_env.is_extern_node_id(item.node_id)
                for param in item.params:
                    assert rehydrated_module.type_env.get_binding_type(
                        param.node_id
                    ) == cm.type_env.get_binding_type(param.node_id)

    def test_named_type_and_template_queries_match(
        self, compiled: _Compiled, rehydrated: dict[ModuleId, CheckedModule]
    ) -> None:
        for mid in _non_entry_module_ids(compiled):
            rehydrated_module = rehydrated[mid]
            cm = compiled.checked.modules[mid]
            for item in static_type_items(cm.resolved.program.body.items):
                scope_path = tuple(segment.name for segment in item.scope_path)
                with rehydrated_module.type_env.type_scope(scope_path):
                    re_named = rehydrated_module.type_env.resolve_named_type(item.name)
                with cm.type_env.type_scope(scope_path):
                    cm_named = cm.type_env.resolve_named_type(item.name)
                assert re_named == cm_named
                assert rehydrated_module.type_env.source_type_template_qname(
                    mid, item.name, scope_path=scope_path
                ) == cm.type_env.source_type_template_qname(mid, item.name, scope_path=scope_path)

    def test_static_let_and_var_binding_types_match(
        self, compiled: _Compiled, rehydrated: dict[ModuleId, CheckedModule]
    ) -> None:
        for mid in _non_entry_module_ids(compiled):
            rehydrated_module = rehydrated[mid]
            cm = compiled.checked.modules[mid]
            for item in static_items(cm.resolved.program.body.items):
                if not isinstance(item, (LetDecl, VarDecl)):
                    continue
                node_id = static_binding_node_id(item)
                assert rehydrated_module.type_env.get_binding_type(
                    node_id
                ) == cm.type_env.get_binding_type(node_id)

    def test_enum_and_generic_whole_environment_queries_match(
        self, compiled: _Compiled, rehydrated: dict[ModuleId, CheckedModule]
    ) -> None:
        for mid in _non_entry_module_ids(compiled):
            rehydrated_module = rehydrated[mid]
            cm = compiled.checked.modules[mid]
            assert rehydrated_module.type_env.enum_owner_forms() == cm.type_env.enum_owner_forms()
            assert (
                rehydrated_module.type_env.blocked_enum_variants()
                == cm.type_env.blocked_enum_variants()
            )
            assert rehydrated_module.type_env.all_generic_types() == cm.type_env.all_generic_types()

    def test_rehydration_reseals_a_complete_image(
        self, compiled: _Compiled, rehydrated: dict[ModuleId, CheckedModule]
    ) -> None:
        """A rehydrated module is sealed and its own image equals the original.

        ``rehydrate`` ends with ``seal()``: the environment is frozen, and a
        further ``image()`` call on the rehydrated module -- covering every
        field, including ``interface`` -- reproduces the image it was built
        from.
        """
        for mid in _non_entry_module_ids(compiled):
            rehydrated_module = rehydrated[mid]
            cm = compiled.checked.modules[mid]
            assert rehydrated_module.type_env.is_sealed
            assert rehydrated_module.image() == cm.image()


class TestRehydratedProgramLowering:
    def test_program_assembled_from_rehydrated_modules_lowers_identically(
        self, compiled: _Compiled, rehydrated: dict[ModuleId, CheckedModule]
    ) -> None:
        """Lowering is deterministic over its checked input, and every field on
        the reassembled program besides ``modules`` is untouched, so comparing
        the linked ``ExecutableProgram``s with plain dataclass ``==`` is exact:
        node ids, ``NominalId`` values, and every table are drawn from the same
        underlying ``resolved`` AST objects on both sides (rehydration keeps
        ``resolved`` identical), so no identity-sensitive field can differ.
        """
        original_executable = lower_program(compile_program_matches(compiled.checked).compiled)

        reassembled_modules = dict(compiled.checked.modules)
        reassembled_modules.update(rehydrated)
        reassembled = dataclasses.replace(compiled.checked, modules=reassembled_modules)
        rehydrated_executable = lower_program(compile_program_matches(reassembled).compiled)

        assert rehydrated_executable == original_executable


class TestImageArtifactSerialization:
    def test_image_round_trips_through_artifact_serialization(self, compiled: _Compiled) -> None:
        """Every non-entry module's image round-trips, not just one.

        Covers the allow-list's harder corners: the enum module's
        ``TypeDef``/``NominalId``/``ConstructorRef``/``BindingRef``, and the
        generics module's ``GenericTypeDef``/``ConstructorSignature``.
        """
        for mid in _non_entry_module_ids(compiled):
            image = compiled.checked.modules[mid].image()

            key = os.urandom(16)
            artifact_serialization.save(key, "test-checked-module-image-round-trip", image)
            restored = artifact_serialization.load(key, "test-checked-module-image-round-trip")

            assert restored == image


def _is_vacuous(value: object) -> bool:
    """Whether *value* is an image field's "nothing recorded" state.

    Generic over field shape so the guard test below needs no per-field
    knowledge: every ``CheckedModuleImage`` field is a mapping, a tuple, an
    optional value, or one of the three small aggregate types the image
    carries.
    """
    if value is None:
        return True
    if isinstance(value, (dict, tuple)):
        return len(value) == 0
    if isinstance(value, ArgumentBindings):
        return not (
            value.function_calls
            or value.function_param_types
            or value.constructor_calls
            or value.constructor_patterns
        )
    if isinstance(value, ModuleTypeInterface):
        return not (
            value.types
            or value.generics
            or value.aliases
            or value.constructors
            or value.field_kinds
            or value.definitions
        )
    if isinstance(value, EnvironmentFacts):
        return len(value.entries) == 0
    return False


class TestImageFieldsAreNonVacuous:
    def test_every_image_field_is_populated_by_some_non_entry_module(
        self, compiled: _Compiled
    ) -> None:
        """No ``CheckedModuleImage`` field is vacuous across the whole fixture.

        Drops a field silently (a missing keyword in ``CheckedModule.image``)
        would otherwise pass every other test here, since a rehydrated
        module's non-env fields are the same objects the image carried.
        Iterating ``dataclasses.fields`` picks up a newly added field for
        free; the failure names the vacuous one.
        """
        images = [compiled.checked.modules[mid].image() for mid in _non_entry_module_ids(compiled)]
        for image_field in dataclasses.fields(CheckedModuleImage):
            assert any(not _is_vacuous(getattr(image, image_field.name)) for image in images), (
                image_field.name
            )


class TestRecordsPickleByName:
    """Every record this module persists restores through its field names."""

    def test_every_record_declares_its_fields_as_match_args(self) -> None:
        """``__match_args__`` is the state both pickling and ``image`` read.

        A record whose fields it does not name -- one declared ``kw_only``,
        or with an ``init=False`` field -- would silently lose that field on
        a round trip, and one declared without ``_pickles_by_name`` would keep
        the per-object field-tuple rebuild those hooks exist to avoid.
        """
        records = _env_records()
        assert CheckedModuleImage in records
        for record in records:
            assert record.__match_args__ == tuple(f.name for f in dataclasses.fields(record)), (
                record.__name__
            )
            assert record.__getstate__ is env_module._Record.__getstate__, record.__name__
            assert record.__setstate__ is env_module._Record.__setstate__, record.__name__


class TestImageMirrorsItsModule:
    """``image``/``rehydrate`` move fields by name, so the names must line up."""

    def test_every_image_field_names_a_member_of_the_checked_module(self) -> None:
        for image_field in dataclasses.fields(CheckedModuleImage):
            assert hasattr(CheckedModule, image_field.name) or image_field.name in {
                f.name for f in dataclasses.fields(CheckedModule)
            }, image_field.name

    def test_the_module_fields_an_image_cannot_carry_are_the_live_ones(self) -> None:
        """``rehydrate`` supplies exactly these from the current compilation."""
        image_fields = {f.name for f in dataclasses.fields(CheckedModuleImage)}
        module_fields = {f.name for f in dataclasses.fields(CheckedModule)}
        assert module_fields - image_fields == {
            "resolved",
            "type_env",
            "import_env",
            "source_text",
        }


def _env_records() -> list[type[object]]:
    """Every frozen data record ``typecheck.env`` declares."""
    return [
        value
        for value in vars(env_module).values()
        if isinstance(value, type)
        and issubclass(value, env_module._Record)
        and dataclasses.is_dataclass(value)
    ]


def _published_surface(
    surface: PublishedModuleSurface,
) -> tuple[ModuleTypeInterface, dict[int, FunctionSignatureRecord] | None, dict[int, Type] | None]:
    """Read a module's published surface the way the pre-pass tables do."""
    return surface.interface, surface.published_signatures, surface.published_binding_types


class TestPublishedModuleSurface:
    def test_an_image_publishes_the_same_surface_as_its_module(self, compiled: _Compiled) -> None:
        """An image stands in for its module wherever the pre-passes read that surface.

        ``_build_program_type_table`` and ``_build_program_func_sig_table``
        reconstruct a reused module's declarations from exactly these two
        members, so an image may replace a live ``CheckedModule`` there only
        if it publishes the identical pair.
        """
        for mid in _non_entry_module_ids(compiled):
            cm = compiled.checked.modules[mid]
            assert _published_surface(cm.image()) == _published_surface(cm)


class TestJournalNeverStartsOutsideProgramChecking:
    def test_repl_seed_environment_is_never_journaled(self) -> None:
        """A REPL session's seed env is only ever copied from, never mutated.

        ``seed_from`` requires a sealed source; a real session env is sealed
        by the ``check_program`` run that produced it. Sealing it directly
        here (skipping ``begin_facts``) is the minimal environment that
        satisfies that precondition without itself having journaled.
        """
        seed_env = TypeEnvironment()
        seed_env.seal()
        resolve_and_check_repl_entry(
            "def value() -> int = 1", base_caps(), seed_env=seed_env, default_stdlib=False
        )
        with pytest.raises(AssertionError):
            seed_env.own_facts()
