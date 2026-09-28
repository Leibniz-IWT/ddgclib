"""
fig_style.py
============

Shared Elsevier-like figure style for the equil_bubble single-bubble
manuscript.  Import and call ``apply_style()`` as the first non-import line of
every figure generator so that ALL data figures match:

  * serif body font (Times New Roman / Nimbus Roman if installed, else the
    guaranteed DejaVu Serif fallback),
  * STIX math font,
  * consistent label / tick / title / legend sizes,
  * embedded (Type-42) fonts in the vector PDF.

Capitalisation policy for every title, axis label, legend entry and
annotation is sentence case: capitalise only the FIRST word, lowercase the
rest EXCEPT proper nouns and math symbols/units (kept verbatim).  That policy
lives in the individual generators; this module only fixes fonts and sizes.
"""
from __future__ import annotations

import matplotlib as mpl


def apply_style() -> None:
    """Apply the shared manuscript figure style to the global rcParams."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman",
                       "Nimbus Roman No9 L", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "figure.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2.0,
    })
