"""Markdown views of the method registry and of preset configs.

``python -m ddgclib.methods --markdown`` regenerates the tables that
``METHODS.md`` embeds, so the document never drifts from the registry.
"""
from __future__ import annotations

from typing import Any, Iterable

from ddgclib.methods._axes import AXES, AXIS_GROUPS, MethodAxis

__all__ = ['axes_markdown', 'presets_markdown', 'config_markdown']


def _fmt(v: Any) -> str:
    if v is None:
        return '`None`'
    if isinstance(v, bool):
        return f'`{v}`'
    return f'`{v}`'


def _axis_markdown(ax: MethodAxis) -> str:
    kind = 'explicit' if ax.explicit else 'reported'
    scope = '' if ax.applies_to == 'both' else f', {ax.applies_to}phase only'
    lines = [
        f"### `{ax.name}` — {ax.title} ({kind}{scope})",
        '',
        f"Default: {_fmt(ax.default)}. Applied via: {ax.control}",
    ]
    if ax.notes:
        lines.append('')
        lines.append(ax.notes)
    lines += [
        '',
        '| value | status | what it does | code | evidence |',
        '|---|---|---|---|---|',
    ]
    for o in ax.options:
        dims = '' if o.dims == (1, 2, 3) else f" (dims {','.join(map(str, o.dims))})"
        lines.append(
            f"| {_fmt(o.key)}{dims} | {o.status} | {o.summary} | "
            f"`{o.where}` | {o.evidence} |")
    lines.append('')
    return '\n'.join(lines)


def axes_markdown(groups: Iterable[str] = AXIS_GROUPS) -> str:
    """One section per axis group, one table per axis."""
    out = []
    for g in groups:
        axes = [a for a in AXES.values() if a.group == g]
        if not axes:
            continue
        out.append(f"## {g}")
        out.append('')
        out.extend(_axis_markdown(a) for a in axes)
    return '\n'.join(out)


def config_markdown(methods) -> str:
    """One-line-per-axis table for a single config."""
    lines = ['| axis | value | status |', '|---|---|---|']
    for name, value in methods.explicit_items():
        ax = AXES[name]
        if ax.applies_to == 'multi' and methods.phases != 'multi':
            continue
        lines.append(f"| `{name}` | {_fmt(value)} | {methods.status_of(name)} |")
    return '\n'.join(lines)


def presets_markdown(presets: dict[str, Any]) -> str:
    """Matrix: one row per preset, one column per explicit axis that
    differs across presets (constant columns are listed once above)."""
    if not presets:
        return ''
    names = list(presets)
    axes = ['dim'] + [a.name for a in AXES.values() if a.explicit
                      and any(hasattr(p, a.name) for p in presets.values())]
    varying = [a for a in axes
               if len({repr(getattr(p, a)) for p in presets.values()}) > 1]
    constant = [a for a in axes if a not in varying]
    first = presets[names[0]]
    lines = []
    if constant:
        lines.append('Same in every preset: ' + ', '.join(
            f"`{a}={getattr(first, a)!r}`" for a in constant))
        lines.append('')
    lines.append('| preset | ' + ' | '.join(f'`{a}`' for a in varying) + ' | source |')
    lines.append('|---|' + '---|' * len(varying) + '---|')
    for n in names:
        p = presets[n]
        cells = [_fmt(getattr(p, a)) for a in varying]
        lines.append(f"| `{n}` | " + ' | '.join(cells) + f" | {p.label} |")
    return '\n'.join(lines)
