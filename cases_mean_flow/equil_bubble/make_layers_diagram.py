"""
make_layers_diagram.py
======================

Overview schematic (Figure 1) for the coupled water-electrolysis bubble
detachment model.  Concise coupling diagram: physical sub-models drawn as
boxes, arrows annotated by the variable they carry.

Coupling structure
-------------------
    mixture thermodynamics (e-NRTL, Butler)      -> surface tension sigma_lv
    applied potential (Young-Lippmann)           -> contact angle theta   (parallel)
    concentration polarisation (Nernst layer)    -> surface molality m_s
    Young-Laplace shape + max-closable volume    -> detachment volume V_d, D_d
    Faradaic growth                              -> departure frequency f, rate

    bulk dielectrophoretic (Maxwell stress)      -> F_DEP, weighed against
                                                    buoyancy F_b, negligible

Output (vector PDF + PNG):
    manuscript/tex/figures/pipeline_layers.pdf
    manuscript/tex/figures/pipeline_layers.png
"""
from __future__ import annotations
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- shared figure style (Elsevier-like serif, applied to every figure) ----
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Nimbus Roman No9 L',
                   'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

_HERE = os.path.dirname(os.path.abspath(__file__))

# muted, print-friendly palette
C_THERMO = '#3b6ea5'   # mixture thermodynamics -> surface tension
C_FIELD  = '#8a5a9e'   # electrowetting -> contact angle
C_POL    = '#2e8b8b'   # concentration polarisation
C_SHAPE  = '#5a9e5a'   # Young-Laplace shape + detachment
C_RATE   = '#c46d2e'   # Faradaic growth -> rate
C_DEP    = '#9a9a9a'   # negligible bulk branch
INK      = '#1a1a1a'
HEAD     = 'white'


def _box(ax, x, y, w, h, color, header, body, out, fs_head=11):
    """Rounded sub-model box: header strip, model description, output variable."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.05',
        linewidth=1.5, edgecolor=color, facecolor=color + '20'))
    hh = 0.30 * h
    ax.add_patch(FancyBboxPatch(
        (x, y + h - hh), w, hh, boxstyle='round,pad=0.0,rounding_size=0.05',
        linewidth=0, facecolor=color))
    ax.text(x + w / 2, y + h - hh / 2, header, color=HEAD,
            fontsize=fs_head, fontweight='bold', va='center', ha='center')
    ax.text(x + w / 2, y + 0.40 * h, body, color='#2b2b2b',
            fontsize=9.3, va='center', ha='center')
    ax.text(x + w / 2, y + 0.13 * h, out, color=color,
            fontsize=10.0, fontweight='bold', va='center', ha='center')


def _tag(ax, x, y, text, color=INK):
    """Small input tag (left column)."""
    ax.text(x, y, text, color=color, fontsize=10.5, fontweight='bold',
            va='center', ha='center')


def _arrow(ax, p0, p1, color=INK, label='', lw=2.0, ls='-',
           conn='arc3,rad=0.0', dx=0.0, dy=0.18, ha='center'):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle='-|>', mutation_scale=16, linewidth=lw,
        color=color, linestyle=ls, shrinkA=3, shrinkB=3,
        connectionstyle=conn))
    if label:
        ax.text((p0[0] + p1[0]) / 2 + dx, (p0[1] + p1[1]) / 2 + dy, label,
                fontsize=9.2, color=color, va='center', ha=ha)


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 7.2), dpi=150)
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 7.2)
    ax.axis('off')

    # ---- title ----------------------------------------------------------
    ax.text(6.5, 6.85,
            'Coupled water electrolysis bubble detachment model',
            fontsize=15, fontweight='bold', ha='center', va='center', color=INK)
    ax.text(6.5, 6.42,
            r'Sub-models connected by the interfacial variables they set '
            r'($\sigma_{lv}$, $\theta$)',
            fontsize=10.5, ha='center', va='center', color='#555555',
            fontstyle='italic')

    # ---- sub-model boxes (reordered to avoid crossing couplings) --------
    sx, sw = 1.9, 3.15                     # sub-model column
    # concentration polarisation (top) -> feeds surface molality DOWN
    cy, ch = 4.95, 1.15
    _box(ax, sx, cy, sw, ch, C_POL, 'Concentration polarisation',
         '1D Nernst diffusion layer',
         r'$\rightarrow\ m_s$   (surface molality)')
    # mixture thermodynamics (middle)
    ty, th = 2.95, 1.25
    _box(ax, sx, ty, sw, th, C_THERMO, 'Mixture thermodynamics',
         'Electrolyte-NRTL activities,\nButler surface tension',
         r'$\rightarrow\ \sigma_{lv}(m)$   (mN m$^{-1}$)')
    # electrowetting (bottom)
    ey, eh = 0.80, 1.15
    _box(ax, sx, ey, sw, eh, C_FIELD, 'Electrowetting',
         'Young$-$Lippmann relation\n(saturating)',
         r'$\rightarrow\ \theta$   (contact angle, rad)')

    # box centre heights (for input arrows and couplings)
    cyc = cy + ch / 2      # concentration polarisation
    tyc = ty + th / 2      # mixture thermodynamics
    eyc = ey + eh / 2      # electrowetting

    # ---- input tags (left) ---------------------------------------------
    _tag(ax, 0.75, cyc, 'Current\ndensity $j$')
    _tag(ax, 0.75, tyc, 'Salt,\nmolality $m$')
    _tag(ax, 0.75, eyc, 'Applied\npotential $E$')

    # detachment (central, tall)
    dx0, dw = 6.15, 3.35
    dy0, dh = 2.55, 2.35
    _box(ax, dx0, dy0, dw, dh, C_SHAPE, 'Young$-$Laplace shape',
         'Pinned axisymmetric profile;\n'
         'detachment = maximum closable\n'
         'pinned volume (force balance:\n'
         r'buoyancy $F_b$ vs pinning $F_{pin}$)',
         r'$\rightarrow\ V_d,\ D_d$   (m$^3$, m)')

    # Faradaic growth (right)
    fx0, fw = 10.15, 2.55
    fy0, fh = 3.35, 1.35
    _box(ax, fx0, fy0, fw, fh, C_RATE, 'Faradaic growth',
         r'Gas flux $\propto j$;  $f = Q/V_d$',
         r'$\rightarrow\ f$ (Hz), rate', fs_head=11)

    # bulk DEP branch (bottom, negligible)
    gx0, gw = 6.15, 3.35
    gy0, gh = 0.45, 1.35
    ax.add_patch(FancyBboxPatch(
        (gx0, gy0), gw, gh, boxstyle='round,pad=0.0,rounding_size=0.05',
        linewidth=1.3, edgecolor=C_DEP, facecolor=C_DEP + '18',
        linestyle='--'))
    ax.text(gx0 + gw / 2, gy0 + gh - 0.42,
            'Bulk dielectrophoretic force\n(Maxwell stress)',
            color='#5a5a5a', fontsize=9.6, fontweight='bold',
            va='center', ha='center', linespacing=1.25)
    ax.text(gx0 + gw / 2, gy0 + 0.36,
            r'$F_{DEP}/F_b \sim 10^{-9}$   $\Rightarrow$   negligible',
            color='#5a5a5a', fontsize=9.8, va='center', ha='center')

    # ---- input arrows ---------------------------------------------------
    _arrow(ax, (1.30, cyc), (sx, cyc), color='#555555', lw=1.6, label='')
    _arrow(ax, (1.30, tyc), (sx, tyc), color='#555555', lw=1.6, label='')
    _arrow(ax, (1.30, eyc), (sx, eyc), color='#555555', lw=1.6, label='')

    # ---- coupling arrows (annotated by carried variable) ----------------
    # concentration polarisation (top) -> mixture thermodynamics (middle):
    # surface molality carried straight DOWN into the box below.
    mcx = sx + sw / 2
    _arrow(ax, (mcx, cy), (mcx, ty + th), color=C_POL, lw=2.0)
    ax.text(mcx + 0.16, (cy + ty + th) / 2, r'$m_s$', color=C_POL,
            fontsize=9.6, va='center', ha='left', fontweight='bold')

    # mixture thermodynamics (middle) -> shape (surface tension), UPPER entry
    _arrow(ax, (sx + sw, tyc), (dx0, dy0 + dh * 0.78),
           color=C_THERMO, lw=2.4, conn='arc3,rad=-0.10',
           label=r'$\sigma_{lv}$', dx=0.0, dy=0.28)

    # electrowetting (bottom) -> shape (contact angle), LOWER entry;
    # stays below the sigma_lv arrow so the two never cross.
    _arrow(ax, (sx + sw, eyc), (dx0, dy0 + dh * 0.30),
           color=C_FIELD, lw=2.4, conn='arc3,rad=0.06',
           label=r'$\theta$', dx=0.0, dy=-0.30)

    # shape -> Faradaic growth (detachment volume/diameter)
    _arrow(ax, (dx0 + dw, dy0 + dh * 0.55), (fx0, fy0 + fh * 0.5),
           color=C_SHAPE, lw=2.4, label=r'$V_d,\ D_d$', dx=0.0, dy=0.28)

    # Faradaic growth -> output
    _arrow(ax, (fx0 + fw * 0.5, fy0), (fx0 + fw * 0.5, fy0 - 0.85),
           color=C_RATE, lw=2.0, label='', dx=0.0, dy=0.0)
    ax.text(fx0 + fw * 0.5, fy0 - 1.15,
            'departure frequency,\ngas production rate',
            fontsize=9.6, color=C_RATE, fontweight='bold',
            va='center', ha='center')

    # bulk DEP -> weighed against buoyancy in the detachment balance
    _arrow(ax, (gx0 + gw * 0.5, gy0 + gh), (dx0 + dw * 0.5, dy0),
           color=C_DEP, lw=1.6, ls='--',
           label=r'$F_{DEP}$', dx=0.30, dy=0.0, ha='left')

    out_pdf = os.path.join(_HERE, 'manuscript', 'tex', 'figures',
                           'pipeline_layers.pdf')
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf[:-4] + '.png', dpi=200, bbox_inches='tight',
                facecolor='white')

    # keep the case-local fig/ copy in sync (backward-compatible reference)
    fig_dir = os.path.join(_HERE, 'fig')
    if os.path.isdir(fig_dir):
        fig.savefig(os.path.join(fig_dir, 'pipeline_layers.png'),
                    dpi=200, bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(fig_dir, 'pipeline_layers.pdf'),
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Wrote {out_pdf}')


if __name__ == '__main__':
    main()
