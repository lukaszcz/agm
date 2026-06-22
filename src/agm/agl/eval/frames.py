"""Frame and cell model for the AgL IR evaluator (D5).

Per-invocation frame
--------------------
A runtime frame is a ``dict[SymbolId, Slot]`` — one per function invocation.
The base frame holds module state and each function invocation adds one frame.

Slot kinds (D5):
- ``let`` (immutable) symbol → the slot holds a ``Value`` directly (by value).
  ``type alias Slot = Value | Cell``.
- ``var`` (mutable) symbol → the slot holds a ``Cell`` (a tiny mutable box).

``Cell`` wraps a ``Value`` and supports in-place update.  Closures capture a
``var`` by capturing the ``Cell`` reference so that mutations remain visible
through the closure.

``Slot`` is the type alias used by the frame dict values.  It is NOT a class —
use ``isinstance(slot, Cell)`` to discriminate.
"""

from __future__ import annotations

from dataclasses import dataclass

from agm.agl.eval.values import Value
from agm.agl.ir.ids import SymbolId

__all__ = ["Cell", "Frame", "Slot"]


@dataclass(slots=True)
class Cell:
    """A mutable box wrapping a ``Value``.

    Used as the slot for ``var`` (mutable) bindings in the per-invocation
    frame.  The cell itself is mutable (not frozen) so that ``IrAssign`` can
    update the contained value in place.

    Closures capture a ``var`` by capturing the ``Cell`` reference; the cell
    is allocated fresh each time ``IrBind`` executes for a ``var`` symbol (D5).
    """

    value: Value


#: A slot in the runtime frame is either a ``Value`` (for ``let`` bindings)
#: or a ``Cell`` (for ``var`` bindings).  Discriminate with ``isinstance``.
Slot = Value | Cell

#: Runtime frame type: maps each bound ``SymbolId`` to its slot.
Frame = dict[SymbolId, Slot]
