"""Explicit method registry + solver config for the dynamic pipeline.

Why this exists
---------------
The dynamic (Lagrangian) pipeline makes ~20 method decisions per run:
time integrator, connectivity policy, conservative remap, projection
cadence, mass redistribution, per-phase volume split, curvature stencil,
dual volume source, ...  Before this module they were spread over
integrator kwargs, ``functools.partial`` bindings in setup helpers,
policy strings in ``_params.py`` files and dimension-gated branches,
with different names in every case.  ``METHODS.md`` (repo root) is the
human-readable companion: it embeds the tables generated from
:data:`AXES` and the per-case matrix.

Public API
----------
:class:`SolverMethods`
    Frozen, validated dataclass with one field per explicit axis.
    Builders: :meth:`~SolverMethods.retopologize_fn`,
    :meth:`~SolverMethods.dudt_fn`, :meth:`~SolverMethods.integrator_kwargs`,
    :meth:`~SolverMethods.integrate`.  Serialisation: ``to_dict`` /
    ``from_dict`` / ``to_json`` / ``from_json``.
:data:`AXES`
    The registry: allowed values, defaults, status, code anchors, evidence.
:data:`PRESETS`
    Named configs matching the shipped cases (bit-identical, tested).
:func:`effective_methods`
    Resolve the implicit (dim / cache gated) choices on a concrete mesh.
:func:`record_methods`
    Write ``methods.json`` (config + effective + git state) next to results.

Quick check from the shell::

    python -m ddgclib.methods                 # list axes and presets
    python -m ddgclib.methods --markdown      # tables for METHODS.md
    python -m ddgclib.methods --preset oscillating_droplet_2D
"""
from ddgclib.methods._axes import AXES, AXIS_GROUPS, MethodAxis, MethodOption, STATUSES
from ddgclib.methods._config import SolverMethods
from ddgclib.methods._effective import effective_methods, git_state, record_methods
from ddgclib.methods._presets import PRESETS, preset
from ddgclib.methods._report import axes_markdown, config_markdown, presets_markdown

__all__ = [
    'AXES', 'AXIS_GROUPS', 'MethodAxis', 'MethodOption', 'STATUSES',
    'SolverMethods',
    'effective_methods', 'git_state', 'record_methods',
    'PRESETS', 'preset',
    'axes_markdown', 'config_markdown', 'presets_markdown',
]
