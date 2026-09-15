"""Exception behavior at the AgL/Python extern boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import pytest

from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS
from agm.agl.ir.ids import FunctionId, NominalId
from agm.agl.ir.program import NominalDescriptor, NominalKind, ValueDescriptors
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.runtime.boundary import AglException
from agm.agl.runtime.externs import AglCallableProxy, ExternCallWindow, ExternRegistry
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import ExceptionValue, IrClosureValue, TextValue, Value

_NO_DESCRIPTORS = ValueDescriptors(nominals={}, functions={})


class _NominalConstructor(Protocol):
    def __call__(self, **fields: object) -> object: ...


def _problem() -> ExceptionValue:
    return ExceptionValue(
        NominalId(9_000_001),
        {"message": TextValue("callback"), "detail": TextValue("original")},
    )


def _raising_proxy(window: ExternCallWindow, problem: ExceptionValue) -> AglCallableProxy:
    def invoke(args: tuple[Value, ...]) -> Value:
        assert args == ()
        raise AglRaise(problem)

    return AglCallableProxy(
        arity=0,
        closure=IrClosureValue(FunctionId(1), ()),
        require_active_window=window.require_active,
        invoke=invoke,
    )


def test_callback_exception_returns_as_the_same_agl_exception_value() -> None:
    registry = ExternRegistry()
    window = ExternCallWindow()
    problem = _problem()
    callback = _raising_proxy(window, problem)

    def invoke_callback(*args: object) -> object:
        assert args == ()
        return callback()

    with window.active(), pytest.raises(AglRaise) as excinfo:
        registry.invoke(
            "invoke",
            invoke_callback,
            (),
            nominals=NO_BUILTIN_DECLARATIONS,
            descriptors=_NO_DESCRIPTORS,
        )

    assert excinfo.value.exc is problem


def test_companion_can_catch_and_reraise_a_callback_exception_transparently() -> None:
    registry = ExternRegistry()
    window = ExternCallWindow()
    problem = _problem()
    callback = _raising_proxy(window, problem)

    def catch_and_reraise(*args: object) -> object:
        assert args == ()
        try:
            return callback()
        except AglException as error:
            raise error

    with window.active(), pytest.raises(AglRaise) as excinfo:
        registry.invoke(
            "catch_and_reraise",
            catch_and_reraise,
            (),
            nominals=NO_BUILTIN_DECLARATIONS,
            descriptors=_NO_DESCRIPTORS,
        )

    assert excinfo.value.exc is problem


def test_companion_can_raise_a_synthesized_exception_through_the_carrier() -> None:
    problem_id = NominalId(9_000_002)
    registry = ExternRegistry()
    registry.set_nominals(
        {
            problem_id: NominalDescriptor(
                nominal=problem_id,
                module_id=ENTRY_ID,
                scope_path=(),
                declared_name="Problem",
                kind=NominalKind.EXCEPTION,
                fields=("message", "detail"),
            )
        },
    )
    problem_class = cast(_NominalConstructor, registry._nominal_classes[problem_id])

    def raise_problem(*args: object) -> object:
        assert args == ()
        raise AglException(problem_class(message="companion", detail="initiated"))

    with pytest.raises(AglRaise) as excinfo:
        registry.invoke(
            "raise_problem",
            raise_problem,
            (),
            nominals=NO_BUILTIN_DECLARATIONS,
            descriptors=_NO_DESCRIPTORS,
        )

    assert excinfo.value.exc == ExceptionValue(
        problem_id,
        {"message": TextValue("companion"), "detail": TextValue("initiated")},
    )


@pytest.mark.parametrize(
    "value",
    ["not an exception", object(), [], {}, lambda: None],
)
def test_carrier_rejects_every_non_exception_python_value(value: object) -> None:
    with pytest.raises(TypeError):
        AglException(value)


def test_carrier_name_is_reserved_when_an_agl_exception_uses_it(tmp_path: Path) -> None:
    exception_id = NominalId(9_000_003)
    registry = ExternRegistry()
    registry.set_nominals(
        {
            exception_id: NominalDescriptor(
                nominal=exception_id,
                module_id=ENTRY_ID,
                scope_path=(),
                declared_name="AglException",
                kind=NominalKind.EXCEPTION,
                fields=("message",),
            )
        },
    )
    companion = tmp_path / "companion.py"
    companion.write_text(
        "from agl import AglException, nominals\n"
        "def raise_named_exception():\n"
        "    raise AglException(nominals.entry.AglException(message='companion'))\n"
    )
    module = registry.load_companion(ENTRY_ID, companion)

    with pytest.raises(AglRaise) as excinfo:
        registry.invoke(
            "raise_named_exception",
            module.raise_named_exception,
            (),
            nominals=NO_BUILTIN_DECLARATIONS,
            descriptors=_NO_DESCRIPTORS,
        )

    assert excinfo.value.exc == ExceptionValue(exception_id, {"message": TextValue("companion")})


def test_ordinary_python_exceptions_become_extern_error_and_base_exceptions_propagate() -> None:
    registry = ExternRegistry()

    def raise_ordinary(*args: object) -> object:
        assert args == ()
        raise RuntimeError("boom")

    def raise_interrupt(*args: object) -> object:
        assert args == ()
        raise KeyboardInterrupt

    with pytest.raises(AglRaise) as ordinary:
        registry.invoke(
            "ordinary",
            raise_ordinary,
            (),
            nominals=NO_BUILTIN_DECLARATIONS,
            descriptors=_NO_DESCRIPTORS,
        )
    assert ordinary.value.exc.nominal == NO_BUILTIN_DECLARATIONS.resolve("ExternError").nominal
    assert ordinary.value.exc.fields["python-type"] == TextValue("RuntimeError")

    with pytest.raises(KeyboardInterrupt):
        registry.invoke(
            "interrupt",
            raise_interrupt,
            (),
            nominals=NO_BUILTIN_DECLARATIONS,
            descriptors=_NO_DESCRIPTORS,
        )


def test_str_of_agl_exception_is_the_message() -> None:
    """A companion's ``str(AglException(...))`` is the AgL exception's message."""
    problem = _problem()

    assert str(AglException(problem)) == "callback"
