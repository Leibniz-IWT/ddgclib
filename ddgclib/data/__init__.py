"""Data handling: save/load simulation states and record time-series history."""

from ddgclib.data._io import save_state, load_state
from ddgclib.data._history import StateHistory
from ddgclib.data._conservation import (
    compute_conservation,
    as_jsonable,
    drift_fractions,
)

__all__ = [
    'save_state',
    'load_state',
    'StateHistory',
    'compute_conservation',
    'as_jsonable',
    'drift_fractions',
]
