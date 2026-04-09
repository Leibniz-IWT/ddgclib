"""Multiphase visualization helpers.

Provides ``record_multiphase_frame`` for capturing per-vertex multiphase
data, and ``dynamic_plot_multiphase`` as a convenience wrapper around
``dynamic_plot_fluid`` with phase/interface overlays enabled.

Usage
-----
    from ddgclib.visualization.multiphase import (
        record_multiphase_frame,
        dynamic_plot_multiphase,
    )

    # During simulation, record to StateHistory with phase fields:
    history = StateHistory(fields=['u', 'p', 'phase', 'is_interface'], ...)

    # After simulation, generate animation:
    dynamic_plot_multiphase(history, HC, save_path='fig/droplet.mp4')

    # Or use the standard dynamic_plot_fluid directly:
    from ddgclib.visualization import dynamic_plot_fluid
    dynamic_plot_fluid(history, HC, phase_field='phase',
                       interface_field='is_interface', reference_R=R0)
"""
from __future__ import annotations

import numpy as np


def record_multiphase_frame(HC, t: float, dim: int = 2) -> dict:
    """Capture per-vertex data for one animation frame.

    Records positions, pressure, velocity, phase assignment, and
    interface membership for every vertex.  Suitable for Lagrangian
    meshes where vertex count and connectivity may change each step.

    Parameters
    ----------
    HC : Complex
    t : float
        Simulation time.
    dim : int
        Spatial dimension.

    Returns
    -------
    dict
        Keys: ``'t'``, ``'x'``, ``'p'``, ``'u'``, ``'phase'``,
        ``'is_interface'``.  Values are numpy arrays.
    """
    xs, ps, us, phases, is_iface = [], [], [], [], []
    for v in HC.V:
        xs.append(v.x_a[:dim].copy())
        ps.append(float(v.p) if np.ndim(v.p) == 0 else float(v.p[0]))
        us.append(v.u[:dim].copy())
        phases.append(int(getattr(v, 'phase', 0)))
        is_iface.append(bool(getattr(v, 'is_interface', False)))
    return {
        't': t,
        'x': np.array(xs),
        'p': np.array(ps),
        'u': np.array(us),
        'phase': np.array(phases),
        'is_interface': np.array(is_iface),
    }


def dynamic_plot_multiphase(
    history,
    HC,
    bV=None,
    save_path: str = 'fig/multiphase.mp4',
    reference_R: float = None,
    fps: int = 20,
    dpi: int = 120,
    xlim: tuple = None,
    ylim: tuple = None,
    **kwargs,
):
    """Convenience wrapper: ``dynamic_plot_fluid`` with multiphase overlays.

    Enables ``phase_field='phase'`` and ``interface_field='is_interface'``
    automatically.  Requires the ``StateHistory`` to have recorded those
    fields (pass ``fields=['u', 'p', 'phase', 'is_interface']``).

    Parameters
    ----------
    history : StateHistory
    HC : Complex
    bV : set or None
    save_path : str
        Output video path (``.mp4`` recommended).
    reference_R : float or None
        Equilibrium radius for reference circle.
    fps, dpi : int
    xlim, ylim : tuple or None
    **kwargs
        Forwarded to ``dynamic_plot_fluid``.
    """
    from ddgclib.visualization.unified import dynamic_plot_fluid
    return dynamic_plot_fluid(
        history, HC, bV=bV,
        save_path=save_path,
        fps=fps, dpi=dpi,
        xlim=xlim, ylim=ylim,
        phase_field='phase',
        interface_field='is_interface',
        reference_R=reference_R,
        **kwargs,
    )
