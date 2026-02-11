"""Time-series recording of simulation state for post-processing.

StateHistory records snapshots during simulation via a callback and provides
query APIs for analysis.

Usage
-----
    from ddgclib.data import StateHistory

    history = StateHistory(fields=['u', 'P'], record_every=10)

    # Use as callback in integrator
    euler_velocity_only(HC, bV, dudt_fn, dt=1e-4, n_steps=1000,
                        callback=history.callback)

    # Query after simulation
    times, values = history.query_vertex(vertex_key, 'P')
    field_dict = history.query_field_at_time(0.5, 'P')
"""

import copy
from typing import Optional, Sequence

import numpy as np


class StateHistory:
    """Records simulation snapshots for post-processing.

    Parameters
    ----------
    fields : sequence of str
        Vertex attributes to record (default: ['u', 'P']).
    record_every : int
        Record a snapshot every N steps (default: 1).
    """

    def __init__(
        self,
        fields: Sequence[str] = ('u', 'P'),
        record_every: int = 1,
    ):
        self.fields = list(fields)
        self.record_every = record_every

        # Storage: list of (time, {vertex_key: {field: value}}, diagnostics)
        self._snapshots: list[tuple[float, dict, dict]] = []

    @property
    def times(self) -> list[float]:
        """List of recorded times."""
        return [s[0] for s in self._snapshots]

    @property
    def n_snapshots(self) -> int:
        return len(self._snapshots)

    def callback(self, step, t, HC, bV=None, diagnostics=None):
        """Callback suitable for integrators (supports both old and new signature).

        Can be passed directly as the ``callback`` argument to any integrator.
        """
        if step % self.record_every != 0:
            return

        snapshot = {}
        for v in HC.V:
            key = tuple(float(x) for x in v.x_a)
            vdata = {}
            for f in self.fields:
                val = getattr(v, f, None)
                if val is None:
                    continue
                if isinstance(val, np.ndarray):
                    vdata[f] = val.copy()
                else:
                    vdata[f] = val
            snapshot[key] = vdata

        diag = dict(diagnostics) if diagnostics else {}
        self._snapshots.append((float(t), snapshot, diag))

    def append(self, t: float, HC, diagnostics: Optional[dict] = None):
        """Manually record a snapshot (alternative to callback)."""
        snapshot = {}
        for v in HC.V:
            key = tuple(float(x) for x in v.x_a)
            vdata = {}
            for f in self.fields:
                val = getattr(v, f, None)
                if val is None:
                    continue
                if isinstance(val, np.ndarray):
                    vdata[f] = val.copy()
                else:
                    vdata[f] = val
            snapshot[key] = vdata

        diag = dict(diagnostics) if diagnostics else {}
        self._snapshots.append((float(t), snapshot, diag))

    def query_vertex(self, vertex_key: tuple, field: str):
        """Get time series of a field at a specific vertex.

        Parameters
        ----------
        vertex_key : tuple
            Vertex coordinate tuple (e.g. (0.5, 0.5)).
        field : str
            Field name (e.g. 'P', 'u').

        Returns
        -------
        times : list[float]
            Times at which the vertex was recorded.
        values : list
            Field values at each time.
        """
        times = []
        values = []
        for t, snapshot, _ in self._snapshots:
            if vertex_key in snapshot and field in snapshot[vertex_key]:
                times.append(t)
                val = snapshot[vertex_key][field]
                if isinstance(val, np.ndarray):
                    values.append(val.copy())
                else:
                    values.append(val)
        return times, values

    def query_field_at_time(self, t: float, field: str) -> dict:
        """Get field values at the snapshot closest to time t.

        Parameters
        ----------
        t : float
            Target time.
        field : str
            Field name.

        Returns
        -------
        dict
            Mapping vertex_key -> field value at the closest snapshot time.
        """
        if not self._snapshots:
            return {}

        # Find closest snapshot
        idx = min(range(len(self._snapshots)),
                  key=lambda i: abs(self._snapshots[i][0] - t))
        _, snapshot, _ = self._snapshots[idx]

        result = {}
        for key, vdata in snapshot.items():
            if field in vdata:
                result[key] = vdata[field]
        return result

    def query_diagnostics(self) -> list[tuple[float, dict]]:
        """Return all (time, diagnostics) pairs."""
        return [(t, d) for t, _, d in self._snapshots]

    def clear(self):
        """Remove all recorded snapshots."""
        self._snapshots.clear()
