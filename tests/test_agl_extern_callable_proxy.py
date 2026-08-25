"""Callable proxies passed from AgL to extern companions."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from agm.agl.ir.ids import FunctionId, NominalId
from agm.agl.runtime.boundary import BoundaryTypeError, BoundaryViolation, encode_boundary_value
from agm.agl.runtime.externs import (
    AglCallableProxy,
    CallableProxyError,
    ExternCallWindow,
    ExternRegistry,
)
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import ConstructorValue, IntValue, IrClosureValue, TextValue, Value
from tests.agl.ir_harness import evaluate_ir_raises_with_externs, evaluate_ir_with_externs


def _proxy(window: ExternCallWindow) -> AglCallableProxy:
    def invoke(args: tuple[Value, ...]) -> Value:
        assert args == (IntValue(2),)
        return IntValue(3)

    return AglCallableProxy(
        arity=1,
        closure=IrClosureValue(FunctionId(1), ()),
        require_active_window=window.require_active,
        invoke=invoke,
        encode=encode_boundary_value,
    )


def test_proxy_converts_arguments_and_result_on_the_owner_thread_during_an_extern_call() -> None:
    registry = ExternRegistry()
    window = ExternCallWindow()
    proxy = _proxy(window)

    with window.active():
        assert registry.invoke("apply", lambda: proxy(2), ()) == IntValue(3)


def test_proxy_rejects_invocation_after_the_extern_call_window_closes() -> None:
    window = ExternCallWindow()
    proxy = _proxy(window)

    with pytest.raises(CallableProxyError):
        proxy(2)


def test_proxy_rejects_worker_registry_window_bypass() -> None:
    registry = ExternRegistry()
    window = ExternCallWindow()
    proxy = _proxy(window)
    failures: list[Exception] = []

    def invoke_from_worker() -> None:
        def call_proxy() -> object:
            try:
                proxy(2)
            except Exception as exc:
                failures.append(exc)
            return None

        registry.invoke("worker", call_proxy, ())

    with window.active():
        worker = threading.Thread(target=invoke_from_worker)
        worker.start()
        worker.join()

    assert len(failures) == 1
    assert isinstance(failures[0], CallableProxyError)


def test_window_rejects_another_thread_while_its_interpreter_is_active() -> None:
    window = ExternCallWindow()
    failures: list[Exception] = []

    def open_from_worker() -> None:
        try:
            with window.active():
                pass
        except Exception as exc:
            failures.append(exc)

    with window.active():
        worker = threading.Thread(target=open_from_worker)
        worker.start()
        worker.join()

    assert len(failures) == 1
    assert isinstance(failures[0], CallableProxyError)


def test_proxy_remains_valid_for_nested_and_later_owner_extern_calls() -> None:
    registry = ExternRegistry()
    window = ExternCallWindow()
    proxy = _proxy(window)

    def invoke_nested() -> object:
        with window.active():
            nested = registry.invoke("nested", lambda: proxy(2), ())
        assert isinstance(nested, IntValue)
        return nested.value

    with window.active():
        assert registry.invoke("outer", invoke_nested, ()) == IntValue(3)
    with window.active():
        assert registry.invoke("later", lambda: proxy(2), ()) == IntValue(3)


def test_shared_registry_keeps_interpreter_callback_windows_independent() -> None:
    registry = ExternRegistry()
    first_window = ExternCallWindow()
    second_window = ExternCallWindow()
    first_proxy = _proxy(first_window)
    second_proxy = _proxy(second_window)
    second_started = threading.Event()
    second_results: list[object] = []

    def invoke_second() -> object:
        with second_window.active():
            second_started.set()
            return second_proxy(2)

    def invoke_first() -> object:
        worker = threading.Thread(
            target=lambda: second_results.append(registry.invoke("second", invoke_second, ()))
        )
        worker.start()
        assert second_started.wait(timeout=5)
        first_result = first_proxy(2)
        worker.join()
        return first_result

    with first_window.active():
        assert registry.invoke("first", invoke_first, ()) == IntValue(3)
    assert second_results == [IntValue(3)]


def test_proxy_rejects_keyword_arguments() -> None:
    registry = ExternRegistry()
    window = ExternCallWindow()
    proxy = _proxy(window)

    def invoke_with_keyword() -> object:
        try:
            proxy(value=2)
        except TypeError:
            return 1
        return 0

    with window.active():
        assert registry.invoke("apply", invoke_with_keyword, ()) == IntValue(1)


def test_proxy_rejects_the_wrong_number_of_positional_arguments() -> None:
    registry = ExternRegistry()
    window = ExternCallWindow()
    proxy = _proxy(window)

    def invoke_without_an_argument() -> object:
        try:
            proxy()
        except TypeError:
            return 1
        return 0

    with window.active():
        assert registry.invoke("apply", invoke_without_an_argument, ()) == IntValue(1)


def test_proxy_rejects_unsupported_python_arguments() -> None:
    registry = ExternRegistry()
    window = ExternCallWindow()
    proxy = _proxy(window)

    def invoke_with_bad_argument() -> object:
        try:
            proxy(object())
        except BoundaryTypeError:
            return 1
        return 0

    with window.active():
        assert registry.invoke("apply", invoke_with_bad_argument, ()) == IntValue(1)


def test_encoding_a_function_requires_an_interpreter_callback_factory() -> None:
    closure = IrClosureValue(FunctionId(1), ())

    with pytest.raises(BoundaryViolation):
        encode_boundary_value(closure)


def test_python_callable_return_remains_a_boundary_error() -> None:
    registry = ExternRegistry()

    with pytest.raises(AglRaise):
        registry.invoke("build", lambda: lambda value: value, ())


def test_unencodable_argument_remains_an_extern_error() -> None:
    registry = ExternRegistry()

    with pytest.raises(AglRaise):
        registry.invoke("take", lambda value: value, (ConstructorValue(NominalId(1), "Box"),))


def test_extern_returning_a_callback_round_trips_its_agl_closure(tmp_path: Path) -> None:
    result, _ = evaluate_ir_with_externs(
        "extern def relay(f: (int) -> int) -> (int) -> int\n"
        "extern def increment(value: int) -> int\n"
        "let callback = relay(increment)\n"
        "let result = callback(2)\n",
        "def relay(f): f(2); return f\ndef increment(value): return value + 1\n",
        tmp_path,
    )

    assert result["result"] == IntValue(3)


def test_returning_a_python_callable_into_agl_raises_extern_error(tmp_path: Path) -> None:
    exc = evaluate_ir_raises_with_externs(
        "extern def build() -> (int) -> int\nlet callback = build()\ncallback(1)\n",
        "def build(): return lambda value: value\n",
        tmp_path,
    )

    assert exc.display_name == "ExternError"
    assert exc.fields["python_type"] == TextValue("")
