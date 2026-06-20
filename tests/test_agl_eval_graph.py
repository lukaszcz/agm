"""Tests for M5a: multi-module eval graph execution (execute_graph).

All tests run the full pipeline:
    load_graph → resolve_graph → check_graph → execute_graph

Helper ``_run_graph`` builds module files in ``tmp_path`` and returns the
entry-frame snapshot after execution.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.eval.exceptions import AglRaise
from agm.agl.eval.interpreter import execute_graph
from agm.agl.eval.values import (
    DecimalValue,
    EnumValue,
    IntValue,
    RecordValue,
    TextValue,
    Value,
)
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.loader import load_graph
from agm.agl.modules.roots import RootSet
from agm.agl.runtime.agents import AgentRegistry
from agm.agl.runtime.codec import TextCodec
from agm.agl.runtime.contract import materialize_contract
from agm.agl.scope.graph import resolve_graph
from agm.agl.typecheck.graph import check_graph

_CAPS = HostCapabilities(
    agent_names=frozenset(),
    has_default_agent=False,
    supports_shell_exec=False,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset(
            {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
        ),
    },
)


def _roots(*paths: Path) -> RootSet:
    return RootSet(roots=frozenset(paths))


def _write_module(root: Path, dotted: str, source: str) -> Path:
    mid = ModuleId.from_dotted(dotted)
    p = root / mid.relpath().replace("/", os.sep)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(source)
    return p


def _run_graph(
    entry_source: str,
    modules: dict[str, str],
    tmp_path: Path,
) -> dict[str, Value]:
    """Run a multi-module program and return the entry frame snapshot."""
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)

    for dotted, source in modules.items():
        _write_module(root, dotted, source)

    mg = load_graph(entry_source, entry_path=None, roots=_roots(root))
    rg = resolve_graph(mg)
    cg = check_graph(rg, _CAPS)

    codecs = {"text": TextCodec()}
    contracts = {}
    for _mid, cm in cg.modules.items():
        for node_id, spec in cm.contract_specs.items():
            contracts[node_id] = materialize_contract(spec, codecs)

    registry = AgentRegistry(named={}, default_agent=None)

    return execute_graph(
        cg,
        registry,
        contracts,
        loop_limit=100,
        strict_json=False,
    )


def test_single_program_path_unchanged(tmp_path: Path) -> None:
    """Existing single-module execute() path still works (no regression)."""
    from agm.agl.eval.interpreter import Interpreter
    from agm.agl.eval.scope import Scope
    from agm.agl.parser import parse_program
    from agm.agl.runtime.agents import AgentRegistry
    from agm.agl.scope import resolve
    from agm.agl.typecheck import check

    # Single-module: function call + let binding at top level.
    source = "def add(a: int, b: int) -> int = a + b\nlet x = add(40, 2)\nx"
    program = parse_program(source)
    resolved = resolve(program)
    caps = HostCapabilities(
        agent_names=frozenset(),
        has_default_agent=False,
        supports_shell_exec=False,
        codec_kinds={"text": frozenset({"text"})},
    )
    checked = check(resolved, caps)
    contracts: dict[int, object] = {}
    registry = AgentRegistry(named={}, default_agent=None)
    root_scope = Scope(parent=None)
    interp = Interpreter(
        checked=checked,
        registry=registry,
        contracts=contracts,
        type_env=checked.type_env,
        loop_limit=100,
        strict_json=False,
    )
    interp.execute(root_scope)
    snap = root_scope.snapshot()
    assert snap["x"] == IntValue(42)


def test_cross_module_simple_call(tmp_path: Path) -> None:
    """Entry calls a function from an imported module (qualified)."""
    lib_source = "def double(n: int) -> int = n * 2"
    # Program must end with an expression (not a let/var decl).
    entry_source = "import lib\nlet result = lib::double(21)\nresult"
    snap = _run_graph(entry_source, {"lib": lib_source}, tmp_path)
    assert snap["result"] == IntValue(42)


def test_imported_function_executes_local_bindings_and_assignment(tmp_path: Path) -> None:
    """Library-local bindings retain their checker-selected runtime coercions."""
    lib_source = (
        "def compute(n: int) -> decimal =\n"
        "  let base: decimal = n\n"
        "  var total: decimal = base\n"
        "  total := total + 1\n"
        "  total"
    )
    entry_source = "import lib\nlet result = lib::compute(41)\nresult"

    snap = _run_graph(entry_source, {"lib": lib_source}, tmp_path)

    assert snap["result"] == DecimalValue(42)


def test_library_loop_error_uses_library_source_text(tmp_path: Path) -> None:
    """Runtime exception metadata slices the source identified by the library span."""
    lib_source = "def spin() -> unit =\n  do[1] ()\n  until false"
    entry_source = "import lib\nlib::spin()"

    with pytest.raises(AglRaise) as exc_info:
        _run_graph(entry_source, {"lib": lib_source}, tmp_path)

    assert exc_info.value.exc.type_name == "MaxIterationsExceeded"
    assert exc_info.value.exc.fields["condition"] == TextValue("false")


def test_open_import_call(tmp_path: Path) -> None:
    """Entry uses open import; calls function with bare name."""
    lib_source = 'def greet(name: text) -> text = "Hello, " + name'
    entry_source = 'import lib\nlet msg = greet("World")\nmsg'
    snap = _run_graph(entry_source, {"lib": lib_source}, tmp_path)
    assert snap["msg"] == TextValue("Hello, World")


def test_renamed_import_call(tmp_path: Path) -> None:
    """Entry does 'import lib using add as sum_it' and calls sum_it()."""
    lib_source = "def add(a: int, b: int) -> int = a + b"
    entry_source = "import lib using add as sum_it\nlet result = sum_it(10, 32)\nresult"
    snap = _run_graph(entry_source, {"lib": lib_source}, tmp_path)
    assert snap["result"] == IntValue(42)


def test_qualified_dispatch(tmp_path: Path) -> None:
    """Entry calls lib::add(x, y) with qualified syntax."""
    lib_source = "def add(x: int, y: int) -> int = x + y"
    entry_source = "import lib\nlet r = lib::add(20, 22)\nr"
    snap = _run_graph(entry_source, {"lib": lib_source}, tmp_path)
    assert snap["r"] == IntValue(42)


def test_three_module_pipeline(tmp_path: Path) -> None:
    """Entry imports two independent libraries and combines their results."""
    # math provides arithmetic, strings provides text ops — neither imports the other.
    math_source = "def double(n: int) -> int = n * 2"
    strings_source = 'def shout(s: text) -> text = s + "!"'
    entry_source = (
        "import math\n"
        "import strings\n"
        "let n = math::double(7)\n"
        'let s = strings::shout("hello")\n'
        "n"
    )
    snap = _run_graph(
        entry_source, {"math": math_source, "strings": strings_source}, tmp_path
    )
    assert snap["n"] == IntValue(14)
    assert snap["s"] == TextValue("hello!")


def test_module_qualified_record_constructor(tmp_path: Path) -> None:
    """Entry creates a cross-module record with qualified constructor."""
    lib_source = "record Point\n  x: int\n  y: int"
    entry_source = "import lib\nlet p = lib::Point(x: 10, y: 32)\np"
    snap = _run_graph(entry_source, {"lib": lib_source}, tmp_path)
    assert snap["p"] == RecordValue(
        type_name="Point", fields={"x": IntValue(10), "y": IntValue(32)}
    )


def test_module_qualified_enum_constructor(tmp_path: Path) -> None:
    """Entry uses a qualified enum variant constructor from an imported module."""
    lib_source = "enum Color\n  | Red\n  | Green\n  | Blue"
    # Qualified enum variant syntax: lib::Color.Red
    entry_source = "import lib\nlet c = lib::Color.Red\nc"
    snap = _run_graph(entry_source, {"lib": lib_source}, tmp_path)
    assert snap["c"] == EnumValue(type_name="Color", variant="Red", fields={})


def test_two_library_functions_same_name(tmp_path: Path) -> None:
    """Two modules each with a function named 'compute'; entry calls both qualified.

    Strengthened to use DIFFERENT signatures (different param types) and to
    also define the same name in the entry, actually exercising the collision
    path in the typecheck's function-signature lookup.

    Module 'a': compute(n: int) -> int = n + 1
    Module 'b': compute(s: text) -> text = s
    Entry:      compute(x: bool) -> bool = x  (must not pollute a:: or b:: lookup)

    Qualified calls a::compute(10) → 11, b::compute("hello") → "hello".
    """
    lib_a_source = "def compute(n: int) -> int = n + 1"
    lib_b_source = 'def compute(s: text) -> text = s'
    entry_source = (
        "import a qualified\nimport b qualified\n"
        "def compute(x: bool) -> bool = x\n"
        "let ra = a::compute(10)\n"
        'let rb = b::compute("hello")\n'
        "ra"
    )
    snap = _run_graph(
        entry_source, {"a": lib_a_source, "b": lib_b_source}, tmp_path
    )
    assert snap["ra"] == IntValue(11)
    assert snap["rb"] == TextValue("hello")


def test_entry_with_param_default(tmp_path: Path) -> None:
    """Entry has a param with a default value; execute_graph uses the default."""
    lib_source = "def inc(n: int) -> int = n + 1"
    # param with default — no value needs to be supplied externally.
    entry_source = "import lib\nparam base: int = 10\nlet result = lib::inc(base)\nresult"
    snap = _run_graph(entry_source, {"lib": lib_source}, tmp_path)
    assert snap["result"] == IntValue(11)


def test_entry_with_agent_declaration(tmp_path: Path) -> None:
    """Entry has an 'agent' declaration; execute_graph installs it in the entry frame."""
    from agm.agl.runtime.codec import TextCodec
    from agm.agl.runtime.contract import materialize_contract

    lib_source = "def compute(n: int) -> int = n * 3"
    # 'agent mybot' declaration — declare but do not call (so no agent registry needed).
    entry_source = "import lib\nagent mybot\nlet r = lib::compute(7)\nr"

    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    _write_module(root, "lib", lib_source)

    mg = load_graph(entry_source, entry_path=None, roots=_roots(root))
    rg = resolve_graph(mg)
    # Need has_default_agent=True so that agents in scope don't trigger an error.
    caps = HostCapabilities(
        agent_names=frozenset({"mybot"}),
        has_default_agent=True,
        supports_shell_exec=False,
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset(
                {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
            ),
        },
    )
    cg = check_graph(rg, caps)

    codecs = {"text": TextCodec()}
    contracts = {}
    for _mid, cm in cg.modules.items():
        for node_id, spec in cm.contract_specs.items():
            contracts[node_id] = materialize_contract(spec, codecs)

    registry = AgentRegistry(named={}, default_agent=None)

    snap = execute_graph(
        cg,
        registry,
        contracts,
        loop_limit=100,
        strict_json=False,
    )
    assert snap["r"] == IntValue(21)  # 7 * 3 = 21


def test_cross_file_mutual_recursion(tmp_path: Path) -> None:
    """Cross-file mutual recursion: entry's def calls a lib def, lib def is recursive.

    Module 'parity' exports is_even(n) which is self-recursive (calls itself).
    Entry imports parity and calls parity::is_even, demonstrating cross-module
    call dispatch where the imported function itself recurses.

    This tests that all frames are built before any closure executes (including
    the recursion from within the library module's own frame).
    """
    # The 'parity' module defines both is_even and is_odd (within one module)
    # using mutual recursion inside the library module itself.
    parity_source = (
        "def is_even(n: int) -> bool =\n"
        "  if n = 0 => true\n"
        "  | else => is_odd(n - 1)\n"
        "def is_odd(n: int) -> bool =\n"
        "  if n = 0 => false\n"
        "  | else => is_even(n - 1)"
    )
    entry_source = (
        "import parity\n"
        "let r4 = parity::is_even(4)\n"
        "let r3 = parity::is_odd(3)\n"
        "let r5 = parity::is_even(5)\n"
        "r5"
    )

    from agm.agl.eval.values import BoolValue

    snap = _run_graph(entry_source, {"parity": parity_source}, tmp_path)
    assert snap["r4"] == BoolValue(True)   # 4 is even
    assert snap["r3"] == BoolValue(True)   # 3 is odd
    assert snap["r5"] == BoolValue(False)  # 5 is not even


def test_agent_passed_to_imported_function(tmp_path: Path) -> None:
    """Entry declares an agent, passes it as a value to an imported function.

    The imported function receives an agent-typed argument and calls ask().
    This exercises D7/§8.2: agents belong to the entry; imported functions
    can receive agent-typed arguments.
    """
    from agm.agl.runtime.request import AgentRequest, AgentResponse

    lib_source = (
        "def ask_with(prompt: text, a: agent) -> text =\n"
        "  ask(prompt, agent: a)"
    )
    entry_source = (
        "import lib\n"
        "agent mybot\n"
        "let result = lib::ask_with(\"hello\", mybot)\n"
        "result"
    )

    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    _write_module(root, "lib", lib_source)

    mg = load_graph(entry_source, entry_path=None, roots=_roots(root))
    rg = resolve_graph(mg)
    caps = HostCapabilities(
        agent_names=frozenset({"mybot"}),
        has_default_agent=True,
        supports_shell_exec=False,
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset(
                {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
            ),
        },
    )
    cg = check_graph(rg, caps)

    codecs = {"text": TextCodec()}
    contracts = {}
    for _mid, cm in cg.modules.items():
        for node_id, spec in cm.contract_specs.items():
            contracts[node_id] = materialize_contract(spec, codecs)

    def mybot_fn(req: AgentRequest) -> AgentResponse | str:
        return "world"

    registry = AgentRegistry(named={"mybot": mybot_fn}, default_agent=None)

    snap = execute_graph(
        cg,
        registry,
        contracts,
        loop_limit=100,
        strict_json=False,
    )
    assert snap["result"] == TextValue("world")


def test_true_cross_file_mutual_recursion_qualified(tmp_path: Path) -> None:
    """True A↔B cross-file mutual recursion executes correctly (qualified calls).

    Module 'even' imports 'odd' and defines is_even calling odd::is_odd.
    Module 'odd' imports 'even' and defines is_odd calling even::is_even.
    Entry imports 'even' and calls even::is_even(10).

    This tests D8/§8.2: cyclic imports with cross-module mutual recursion must
    typecheck AND evaluate to the correct result.  It MUST FAIL before the
    function-signature pre-pass is added and MUST PASS after.
    """
    from agm.agl.eval.values import BoolValue

    even_source = (
        "import odd\n"
        "def is_even(n: int) -> bool =\n"
        "  if n = 0 => true\n"
        "  | else => odd::is_odd(n - 1)"
    )
    odd_source = (
        "import even\n"
        "def is_odd(n: int) -> bool =\n"
        "  if n = 0 => false\n"
        "  | else => even::is_even(n - 1)"
    )
    entry_source = (
        "import even\n"
        "let r10 = even::is_even(10)\n"
        "let r7  = even::is_even(7)\n"
        "r10"
    )

    snap = _run_graph(entry_source, {"even": even_source, "odd": odd_source}, tmp_path)
    assert snap["r10"] == BoolValue(True)   # 10 is even
    assert snap["r7"] == BoolValue(False)   # 7 is not even


def test_true_cross_file_mutual_recursion_open_import(tmp_path: Path) -> None:
    """True A↔B cross-file mutual recursion via open (unqualified) imports.

    Same as the qualified variant but uses open imports so calls are unqualified.
    """
    from agm.agl.eval.values import BoolValue

    even_source = (
        "import odd\n"
        "def is_even(n: int) -> bool =\n"
        "  if n = 0 => true\n"
        "  | else => is_odd(n - 1)"
    )
    odd_source = (
        "import even\n"
        "def is_odd(n: int) -> bool =\n"
        "  if n = 0 => false\n"
        "  | else => is_even(n - 1)"
    )
    entry_source = (
        "import even\n"
        "let r6  = is_even(6)\n"
        "let r5  = is_even(5)\n"
        "r6"
    )

    snap = _run_graph(entry_source, {"even": even_source, "odd": odd_source}, tmp_path)
    assert snap["r6"] == BoolValue(True)    # 6 is even
    assert snap["r5"] == BoolValue(False)   # 5 is not even


def test_same_name_functions_qualified_call_evaluates_correctly(tmp_path: Path) -> None:
    """Regression (Finding 1): qualified cross-module call resolves correct signature.

    Entry defines  helper(s: text) -> text.
    Lib defines    helper(n: int) -> int = n + 1.
    Entry calls    lib::helper(5).

    Before the fix: typecheck spuriously rejected this with 'Type mismatch:
    expected text, got int' because _check_declared_name_call fetched the
    name-keyed signature (entry's helper, text param) instead of lib's.
    After the fix: the call typechecks AND evaluates to IntValue(6).
    """
    lib_source = "def helper(n: int) -> int = n + 1"
    entry_source = (
        "import lib qualified\n"
        "def helper(s: text) -> text = s\n"
        "let result = lib::helper(5)\n"
        "result"
    )
    snap = _run_graph(entry_source, {"lib": lib_source}, tmp_path)
    assert snap["result"] == IntValue(6)


def test_two_library_functions_same_name_different_signatures(tmp_path: Path) -> None:
    """Regression (Finding 1): two libs with same-name fns with different signatures.

    Module 'a': helper(n: int) -> int = n + 1
    Module 'b': helper(s: text) -> text = s
    Entry also defines helper(x: bool) -> bool.

    Qualified calls a::helper(10) → 11 and b::helper("hi") → "hi" must
    both evaluate to the correct values.
    """
    lib_a_source = "def helper(n: int) -> int = n + 1"
    lib_b_source = 'def helper(s: text) -> text = s'
    entry_source = (
        "import a qualified\n"
        "import b qualified\n"
        "def helper(x: bool) -> bool = x\n"
        "let ra = a::helper(10)\n"
        'let rb = b::helper("hi")\n'
        "ra"
    )
    snap = _run_graph(entry_source, {"a": lib_a_source, "b": lib_b_source}, tmp_path)
    assert snap["ra"] == IntValue(11)
    assert snap["rb"] == TextValue("hi")


def test_self_qualifier_shadows_param_in_exec_mode(
    tmp_path: Path,
) -> None:
    """Regression (Finding 4): ``::x`` bypasses a same-named param (exec / graph path).

    Bind the result of ``shadow(7)`` to ``result`` so it appears in the snapshot.
    The program ends with an expression so the typechecker accepts it.
    """
    entry_source = (
        "let x = 100\n"
        "def shadow(x: int) -> int = ::x\n"
        "let result = shadow(7)\n"
        "result"
    )
    snap = _run_graph(entry_source, {}, tmp_path)
    assert snap["result"] == IntValue(100), f"Expected 100 (top-level x), got {snap['result']}"
