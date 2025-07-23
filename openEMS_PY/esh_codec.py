"""
esh_codec.py
============

Python counterpart of

    • io.openems.edge.energy.optimizer.EshCodec

OpenEMS stores a *schedule* (one GA chromosome) as a compact hexadecimal
string called **E S H** ( *Energy Schedule Handler* ).  Every element
(battery‑mode for one Period) is encoded as a single *nibble*
(4 bit → 0 … F); the Java helper converts back and forth.

The same convention is replicated here so that a schedule exported from
Python can be copy‑pasted into the original OpenEMS UI – and vice‑versa.
"""

from __future__ import annotations

from typing import Sequence


# -----------------------------------------------------------------------
#  Public helpers
# -----------------------------------------------------------------------

def encode(schedule: Sequence[int]) -> str:
    """
    Convert a list/tuple of mode integers (0‑15) to a compact hex string.

    Example
    -------
    >>> encode([2, 0, 4, 1])
    '2041'
    """
    try:
        return "".join(f"{m:x}" for m in schedule)
    except ValueError as err:
        raise ValueError(
            "Schedule must only contain integers 0 … 15 (one per Period)"
        ) from err


def decode(esh: str) -> list[int]:
    """
    Convert an ESH string back to a list of integers.

    Example
    -------
    >>> decode('2041')
    [2, 0, 4, 1]
    """
    try:
        return [int(ch, 16) for ch in esh.strip()]
    except ValueError as err:
        raise ValueError(
            "ESH string must be valid hexadecimal digits (0‑F)"
        ) from err


# -----------------------------------------------------------------------
#  Small self‑test when the module is executed directly
# -----------------------------------------------------------------------

if __name__ == "__main__":
    original = [0, 1, 2, 3, 4, 0, 2, 2, 1]
    encoded = encode(original)
    decoded = decode(encoded)
    assert decoded == original, (encoded, decoded)
    print("✔︎ round‑trip OK:", encoded)
