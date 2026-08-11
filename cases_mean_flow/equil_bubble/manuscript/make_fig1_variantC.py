"""
make_fig1_variantC.py
=====================

Figure 1 (variant C) for the water-electrolysis single-bubble manuscript.

Design goal for variant C: CLEAN and IMMEDIATE.  A reader should grasp the
context at a glance.  We therefore draw ONE simplified, uncluttered cell (no
busy magnified inset) in which only the essentials of our story appear, and a
minimal four-stage model flow below it.  The two panels are joined by a single
explicit bridge that maps the physical bubble onto the model variables it sets.

  (a) TOP    Simplified alkaline water-electrolysis cell for OUR system,
             focused on the hydrogen bubble at the CATHODE.  Shown: the gas
             bubble (inside H2(g)+H2O(g); outside H2O(l), K+, OH-), the
             emphasised vapour-liquid equilibrium H2O(l) <=> H2O(g) that (with
             the salt) sets the surface tension sigma_lv, the contact angle
             theta, one reactant-transport arrow (H2O(l) to the electrode)
             blocked by the bubble, and the external e- / OH- migration.

  (b) BOTTOM Minimal coupled detachment model.  Mixture thermodynamics
             (e-NRTL, Butler) -> sigma_lv; applied potential (Young-Lippmann)
             -> theta; Young-Laplace shape + maximum-closable volume ->
             V_d, D_d; Faradaic growth -> departure frequency f, gas rate; the
             bulk dielectrophoretic (Maxwell) force is the negligible branch.
             Every coupling arrow is labelled with the variable it carries
             (sigma_lv, theta, V_d, D_d, f).

Bridge: a single labelled connector descends from the bubble interface in (a)
to the interfacial sub-models in (b), stating that the interface composition
plus VLE fix sigma_lv while the applied potential fixes theta.  Colours are
matched (sigma_lv in the thermodynamics blue, theta in the electrowetting
purple) in BOTH panels.

Run (from cases_mean_flow/equil_bubble):
    /home/endres/anaconda3/envs/ddg/bin/python manuscript/make_fig1_variantC.py

Outputs (vector PDF + PNG):
    manuscript/tex/figures/proposals/fig1_variantC.pdf
    manuscript/tex/figures/proposals/fig1_variantC.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Arc,
)

# fig_style lives in the case root (parent of manuscript/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_CASE_ROOT = os.path.dirname(_HERE)
if _CASE_ROOT not in sys.path:
    sys.path.insert(0, _CASE_ROOT)
from fig_style import apply_style  # noqa: E402

# ---- palette (shared with make_layers_diagram / variant B for consistency) ---
C_THERMO = "#3b6ea5"   # mixture thermodynamics -> sigma_lv   (blue)
C_FIELD = "#8a5a9e"     # electrowetting -> theta              (purple)
C_SHAPE = "#5a9e5a"     # Young-Laplace shape -> V_d, D_d      (green)
C_RATE = "#c46d2e"      # Faradaic growth -> f, rate           (orange)
C_DEP = "#9a9a9a"       # negligible bulk branch               (grey)
INK = "#1a1a1a"
HEAD = "white"

C_ELYTE = "#c4dbee"     # electrolyte fill (light steel blue, slide-matched)
C_ELEC = "#c7c7c7"      # electrode grey
C_ION = "#123a5c"       # bulk-ion / migration navy
C_FLUX = "#2a6f4f"      # reactant-transport green
C_BLOCK = "#b5484b"     # muted red for the blocked-transport cut mark
C_BRIDGE = "#444444"    # bridge connector

# canvas (coordinates are in a W x H box, aspect equal)
W, H = 12.5, 14.0


# --------------------------------------------------------------------------- #
#  small drawing helpers
# --------------------------------------------------------------------------- #
def _arrow(ax, p0, p1, color=INK, lw=2.0, ls="-", conn="arc3,rad=0.0",
           mut=15, style="-|>", z=6):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=mut, linewidth=lw,
        color=color, linestyle=ls, shrinkA=0, shrinkB=0,
        connectionstyle=conn, zorder=z))


def _coil(ax, x0, x1, y, n=4, color=INK, lw=1.6, z=5):
    """Inductor-style coil: n humps along a horizontal segment."""
    w = (x1 - x0) / n
    for k in range(n):
        ax.add_patch(Arc((x0 + w / 2 + k * w, y), w, 1.15 * w,
                         theta1=0, theta2=180, lw=lw, color=color, zorder=z))


def _flow_box(ax, x, y, w, h, color, header, body, out,
              fs_head=9.0, fs_body=8.2, fs_out=8.8):
    """Rounded sub-model box: header strip, description, output variable."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=1.4, edgecolor=color, facecolor=color + "20", zorder=3))
    hh = 0.26 * h
    ax.add_patch(FancyBboxPatch(
        (x, y + h - hh), w, hh, boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=0, facecolor=color, zorder=4))
    ax.text(x + w / 2, y + h - hh / 2, header, color=HEAD,
            fontsize=fs_head, fontweight="bold", va="center", ha="center",
            zorder=5)
    ax.text(x + w / 2, y + 0.45 * h, body, color="#2b2b2b",
            fontsize=fs_body, va="center", ha="center", zorder=5,
            linespacing=1.25)
    ax.text(x + w / 2, y + 0.12 * h, out, color=color,
            fontsize=fs_out, fontweight="bold", va="center", ha="center",
            zorder=5)


# --------------------------------------------------------------------------- #
#  panel (a) : simplified reactor cell
# --------------------------------------------------------------------------- #
def _draw_cell(ax):
    # ---- cell body -------------------------------------------------------
    cx0, cx1 = 1.20, 9.60          # cell outer x-range
    cy0, cy1 = 8.30, 12.20         # cell outer y-range
    ca0, ca1 = 1.20, 1.65          # cathode bar (LEFT, our focus)
    an0, an1 = 9.15, 9.60          # anode bar  (right)
    midy = 0.5 * (cy0 + cy1)       # 10.25

    elyte = Rectangle((ca1, cy0), an0 - ca1, cy1 - cy0,
                      facecolor=C_ELYTE, edgecolor="none", zorder=1)
    ax.add_patch(elyte)

    # ---- hydrogen bubble on the cathode (clipped to the electrolyte) ------
    bc = (2.05, midy)
    br = 1.45
    bub = Circle(bc, br, facecolor="white", edgecolor=INK, lw=1.6, zorder=4)
    ax.add_patch(bub)
    bub.set_clip_path(elyte)

    # cell outline + electrode bars (drawn over the clipped bubble foot) -----
    ax.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0,
                           facecolor="none", edgecolor=INK, lw=1.2, zorder=6))
    ax.add_patch(Rectangle((ca0, cy0), ca1 - ca0, cy1 - cy0,
                           facecolor=C_ELEC, edgecolor=INK, lw=1.0, zorder=5))
    ax.add_patch(Rectangle((an0, cy0), an1 - an0, cy1 - cy0,
                           facecolor=C_ELEC, edgecolor=INK, lw=1.0, zorder=5))

    # inside-bubble composition
    ax.text(2.45, midy + 0.18, r"H$_2(g)$", fontsize=9.6, ha="center",
            va="center", color=INK, fontweight="bold", zorder=6)
    ax.text(2.45, midy - 0.28, r"$+\,$H$_2$O$(g)$", fontsize=8.8, ha="center",
            va="center", color=INK, zorder=6)

    dyf = np.sqrt(br ** 2 - (ca1 - bc[0]) ** 2)     # half contact chord
    foot_lo = (ca1, bc[1] - dyf)

    # ---- (1) surface tension sigma_lv : tangent double arrow (blue) -------
    a_sig = np.radians(58.0)
    p_sig = np.array([bc[0] + br * np.cos(a_sig), bc[1] + br * np.sin(a_sig)])
    tvec = np.array([-np.sin(a_sig), np.cos(a_sig)])
    _arrow(ax, p_sig - 0.42 * tvec, p_sig + 0.42 * tvec, color=C_THERMO,
           lw=2.0, mut=12, style="<|-|>", z=8)
    ax.text(p_sig[0] + 0.46, p_sig[1] + 0.16, r"$\sigma_{lv}$",
            fontsize=11.5, color=C_THERMO, ha="left", va="center",
            fontweight="bold", zorder=8)

    # ---- (2) VLE : H2O(l) <=> H2O(g) across the interface (emphasised) ----
    p_vle = np.array([bc[0] + br, bc[1]])          # bubble tip
    _arrow(ax, p_vle + np.array([-0.34, 0.0]), p_vle + np.array([0.42, 0.0]),
           color=C_THERMO, lw=2.2, mut=14, style="<|-|>", z=8)
    ax.text(bc[0] + br + 0.62, midy + 0.04,
            r"H$_2$O$(l)\ \rightleftharpoons\ $H$_2$O$(g)$",
            fontsize=10.5, color=C_THERMO, ha="left", va="center", zorder=8)
    ax.text(bc[0] + br + 0.62, midy - 0.40,
            "vapour-liquid equilibrium", fontsize=8.0, color="#4a6a86",
            ha="left", va="center", fontstyle="italic", zorder=8)

    # ---- (3) contact angle theta at the lower foot (purple) --------------
    ax.add_patch(Arc(foot_lo, 0.94, 0.94, angle=0, theta1=32, theta2=95,
                     color=C_FIELD, lw=1.9, zorder=8))
    ax.text(foot_lo[0] + 0.62, foot_lo[1] + 0.30, r"$\theta$",
            fontsize=12, color=C_FIELD, ha="center", va="center",
            fontweight="bold", zorder=9)

    # ---- (4) reactant transport blocked by the bubble --------------------
    yb = midy - 0.90
    xi = bc[0] + np.sqrt(max(br ** 2 - (yb - bc[1]) ** 2, 0.0))   # interface x
    _arrow(ax, (7.35, yb), (xi + 0.18, yb), color=C_FLUX, lw=1.8, mut=13)
    ax.plot([xi + 0.06, xi + 0.06], [yb - 0.18, yb + 0.18], color=C_BLOCK,
            lw=2.6, zorder=9)
    ax.plot([xi - 0.10, xi + 0.16], [yb - 0.15, yb + 0.15], color=C_BLOCK,
            lw=2.2, zorder=9)
    ax.plot([xi - 0.10, xi + 0.16], [yb + 0.15, yb - 0.15], color=C_BLOCK,
            lw=2.2, zorder=9)
    ax.text(5.85, yb - 0.52, "H$_2$O$(l)$ transport to electrode\n"
            "blocked by the bubble", fontsize=8.4, color=C_FLUX,
            ha="center", va="center", linespacing=1.25, zorder=8)

    # ---- bulk composition + ion migration --------------------------------
    ax.text(6.25, cy1 - 0.42, r"Bulk liquid:  H$_2$O$(l)$,  K$^+$,  OH$^-$",
            fontsize=9.6, ha="center", va="center", color=C_ION, zorder=6)
    _arrow(ax, (6.60, 11.05), (8.70, 11.05), color=C_ION, lw=1.7, mut=13)
    ax.text(7.65, 11.28, r"OH$^-$ migration", fontsize=8.8, ha="center",
            va="bottom", color=C_ION, zorder=6)

    # small O2 bubble at the anode (secondary, muted)
    ax.add_patch(Circle((an0, 9.05), 0.26, facecolor="white",
                        edgecolor="#666666", lw=1.0, zorder=4))
    ax.add_patch(Rectangle((an0, 8.77), an1 - an0, 0.56,
                           facecolor=C_ELEC, edgecolor="none", zorder=5))
    ax.text(an0 - 0.20, 9.55, r"O$_2(g)$", fontsize=8.4, ha="right",
            va="center", color="#666666", zorder=6)

    # ---- external circuit -------------------------------------------------
    xc = (ca0 + ca1) / 2
    xa = (an0 + an1) / 2
    ytop = 12.72
    for xw in (xa, xc):
        ax.plot([xw, xw], [cy1, ytop], color=INK, lw=1.4, zorder=4)
    ax.plot([xc, 4.80], [ytop, ytop], color=INK, lw=1.4, zorder=4)
    ax.plot([6.00, xa], [ytop, ytop], color=INK, lw=1.4, zorder=4)
    _coil(ax, 4.80, 6.00, ytop, n=4, color=INK, lw=1.5)
    # electrons: leave anode (up), enter cathode (down)
    _arrow(ax, (xa, cy1 + 0.14), (xa, cy1 + 0.52), color=INK, lw=1.6, mut=12)
    _arrow(ax, (xc, ytop - 0.14), (xc, ytop - 0.52), color=INK, lw=1.6, mut=12)
    ax.text(xa + 0.18, cy1 + 0.33, r"$e^-$", fontsize=9.0, ha="left",
            va="center", color=INK)
    ax.text(xc - 0.18, ytop - 0.33, r"$e^-$", fontsize=9.0, ha="right",
            va="center", color=INK)

    # ---- electrode captions + cathode half-reaction ----------------------
    ax.text(xc, cy0 - 0.24, "Cathode", fontsize=9.6, ha="center", va="top",
            color=INK, fontweight="bold")
    ax.text(xa, cy0 - 0.24, "Anode", fontsize=9.6, ha="center", va="top",
            color="#555555")
    ax.text(6.00, cy0 - 0.62,
            r"Cathode reaction:  2 H$_2$O $+$ 2 e$^-\ \rightarrow\ $"
            r"H$_2$ $+$ 2 OH$^-$", fontsize=8.6, ha="center", va="top",
            color="#555555")

    return (cx0, cy0)


# --------------------------------------------------------------------------- #
#  panel (b) : minimal coupled model flow
# --------------------------------------------------------------------------- #
def _draw_model(ax):
    def tag(x, y, txt):
        ax.text(x, y, txt, fontsize=8.9, fontweight="bold", ha="center",
                va="center", color=INK, linespacing=1.2)

    # backing panel for the two interfacial sub-models
    ax.add_patch(FancyBboxPatch(
        (1.15, 1.90), 4.40, 3.55, boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=1.0, edgecolor="#c9c9c9", facecolor="#00000006", zorder=2))
    ax.text(4.15, 5.22, "Interfacial sub-models", fontsize=8.6,
            ha="center", va="center", color="#6a6a6a", fontstyle="italic",
            zorder=3)

    # sub-model boxes ------------------------------------------------------
    sx, sw = 1.50, 3.75
    ty0, th = 3.75, 1.25           # mixture thermodynamics
    ey0, eh = 2.10, 1.25           # electrowetting
    tyc, eyc = ty0 + th / 2, ey0 + eh / 2

    _flow_box(ax, sx, ty0, sw, th, C_THERMO, "Mixture thermodynamics",
              "e-NRTL activities,\nButler surface tension",
              r"$\rightarrow\ \sigma_{lv}(m)$  (mN m$^{-1}$)")
    _flow_box(ax, sx, ey0, sw, eh, C_FIELD, "Electrowetting",
              "Young-Lippmann relation\n(saturating)",
              r"$\rightarrow\ \theta$  (contact angle)")

    # detachment box -------------------------------------------------------
    dx0, dw = 6.35, 3.20
    dy0, dh = 2.35, 2.65
    _flow_box(ax, dx0, dy0, dw, dh, C_SHAPE, "Young-Laplace shape",
              "pinned axisymmetric profile;\n"
              "detachment $=$ maximum\nclosable pinned volume\n"
              r"(buoyancy $F_b$ vs pinning $F_{pin}$)",
              r"$\rightarrow\ V_d,\ D_d$  (m$^3$, m)", fs_body=8.0)

    # Faradaic growth box --------------------------------------------------
    fx0, fw = 10.30, 2.05
    fy0, fh = 3.10, 1.25
    _flow_box(ax, fx0, fy0, fw, fh, C_RATE, "Faradaic growth",
              r"gas flux $\propto j$;" + "\n" + r"$f = Q / V_d$",
              r"$\rightarrow\ f$  (Hz)", fs_head=8.0, fs_body=8.0)

    # negligible dielectrophoretic branch ----------------------------------
    gx0, gw = 6.35, 3.20
    gy0, gh = 0.65, 1.05
    ax.add_patch(FancyBboxPatch(
        (gx0, gy0), gw, gh, boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=1.2, edgecolor=C_DEP, facecolor=C_DEP + "16",
        linestyle="--", zorder=3))
    ax.text(gx0 + gw / 2, gy0 + gh - 0.28,
            "Bulk dielectrophoresis (Maxwell stress)",
            color="#5a5a5a", fontsize=8.0, fontweight="bold",
            ha="center", va="center", zorder=5)
    ax.text(gx0 + gw / 2, gy0 + 0.32,
            r"$F_{DEP}/F_b \sim 10^{-9}\ \Rightarrow$ negligible",
            color="#5a5a5a", fontsize=8.2, ha="center", va="center", zorder=5)

    # input tags + arrows --------------------------------------------------
    tag(0.78, tyc, "salt\nmolality $m$")
    tag(0.78, eyc, "applied\npotential $E$")
    _arrow(ax, (1.28, tyc), (sx, tyc), color="#555555", lw=1.5, mut=12)
    _arrow(ax, (1.28, eyc), (sx, eyc), color="#555555", lw=1.5, mut=12)
    tag(fx0 + fw / 2, fy0 + fh + 0.48, "current\ndensity $j$")
    _arrow(ax, (fx0 + fw / 2, fy0 + fh + 0.16), (fx0 + fw / 2, fy0 + fh),
           color="#555555", lw=1.5, mut=12)

    # coupling arrows (each labelled with the variable it carries) ----------
    _arrow(ax, (sx + sw, tyc), (dx0, dy0 + dh * 0.80), color=C_THERMO,
           lw=2.2, mut=14, conn="arc3,rad=-0.10")
    ax.text((sx + sw + dx0) / 2, dy0 + dh * 0.80 + 0.30, r"$\sigma_{lv}$",
            color=C_THERMO, fontsize=10.0, va="center", ha="center",
            fontweight="bold")

    _arrow(ax, (sx + sw, eyc), (dx0, dy0 + dh * 0.22), color=C_FIELD,
           lw=2.2, mut=14, conn="arc3,rad=0.10")
    ax.text((sx + sw + dx0) / 2, dy0 + dh * 0.22 - 0.32, r"$\theta$",
            color=C_FIELD, fontsize=10.6, va="center", ha="center",
            fontweight="bold")

    _arrow(ax, (dx0 + dw, dy0 + dh * 0.5), (fx0, fy0 + fh * 0.5),
           color=C_SHAPE, lw=2.2, mut=14)
    ax.text((dx0 + dw + fx0) / 2, dy0 + dh * 0.5 - 0.34, r"$V_d,\ D_d$",
            color=C_SHAPE, fontsize=8.0, va="center", ha="center",
            fontweight="bold")

    _arrow(ax, (fx0 + fw * 0.5, fy0), (fx0 + fw * 0.5, fy0 - 0.68),
           color=C_RATE, lw=1.9, mut=13)
    ax.text(fx0 + fw * 0.5 + 0.18, fy0 - 0.36, r"$f$", color=C_RATE,
            fontsize=9.6, va="center", ha="left", fontweight="bold")
    ax.text(fx0 + fw * 0.5, fy0 - 1.08,
            "departure frequency $f$,\ngas production rate",
            fontsize=8.2, color=C_RATE, fontweight="bold",
            va="center", ha="center", linespacing=1.2)

    _arrow(ax, (gx0 + gw * 0.5, gy0 + gh), (dx0 + dw * 0.5, dy0),
           color=C_DEP, lw=1.5, ls="--", mut=12)
    ax.text(dx0 + dw * 0.5 + 0.30, (gy0 + gh + dy0) / 2, r"$F_{DEP}$",
            color=C_DEP, fontsize=8.4, va="center", ha="left")

    return (sx + sw / 2, ty0 + th)      # centre-x, top-y of the sub-model stack


# --------------------------------------------------------------------------- #
#  assembly
# --------------------------------------------------------------------------- #
def main() -> tuple[str, str]:
    apply_style()

    fig, ax = plt.subplots(figsize=(7.8, 7.8 * H / W), dpi=150)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    xmid = W / 2

    # ---- panel (a) -------------------------------------------------------
    ax.text(0.15, 13.66, "(a)", fontsize=12.5, fontweight="bold", ha="left",
            va="center", color=INK)
    ax.text(xmid, 13.66,
            "Alkaline water electrolysis: H$_2$ bubble at the cathode",
            fontsize=10.5, fontweight="bold", ha="center", va="center",
            color=INK)
    ax.text(xmid, 13.24,
            "Primary electrolyte KOH(aq); acidic alternative H$_2$SO$_4$",
            fontsize=8.6, ha="center", va="center", color="#666666",
            fontstyle="italic")

    _draw_cell(ax)

    # ---- divider ---------------------------------------------------------
    ax.plot([0.35, W - 0.35], [7.42, 7.42], color="#d3d3d3", lw=1.0, zorder=1)

    # ---- panel (b) -------------------------------------------------------
    ax.text(0.15, 6.02, "(b)", fontsize=12.5, fontweight="bold", ha="left",
            va="center", color=INK)
    ax.text(xmid, 6.02,
            "Coupled model: interfacial variables set bubble departure",
            fontsize=10.5, fontweight="bold", ha="center", va="center",
            color=INK)

    stack_cx, stack_top = _draw_model(ax)

    # ---- bridge : bubble interface -> interfacial sub-models --------------
    # leg 1: down from the cell, under the cathode bubble
    _arrow(ax, (2.50, 8.30), (2.50, 7.32), color=C_BRIDGE, lw=2.4, mut=15)
    # bridge label band
    bx0, by0, bw, bh = 0.55, 6.42, 5.85, 0.86
    ax.add_patch(FancyBboxPatch(
        (bx0, by0), bw, bh, boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=1.0, edgecolor=C_BRIDGE, facecolor="#f3efe6", zorder=3))
    ax.text(bx0 + bw / 2, by0 + bh / 2,
            r"Bubble interface fixes the model inputs:" + "\n"
            r"composition $+$ VLE $\rightarrow\ \sigma_{lv}$;   "
            r"applied potential $\rightarrow\ \theta$",
            fontsize=8.9, ha="center", va="center", color="#333333",
            linespacing=1.3, zorder=4)
    # leg 2: into the interfacial sub-model stack (kept left of the (b) title)
    _arrow(ax, (2.50, by0), (2.50, stack_top + 0.02),
           color=C_BRIDGE, lw=2.4, mut=15)

    # ---- outputs ---------------------------------------------------------
    out_pdf = os.path.join(_HERE, "tex", "figures", "proposals",
                           "fig1_variantC.pdf")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    out_png = out_pdf[:-4] + ".png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    return out_pdf, out_png


if __name__ == "__main__":
    main()
