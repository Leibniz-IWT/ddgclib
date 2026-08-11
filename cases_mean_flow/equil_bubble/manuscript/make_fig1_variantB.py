"""
make_fig1_variantB.py
=====================

Figure 1 (variant B) for the water-electrolysis single-bubble manuscript.

Two stacked, visually connected panels:

  (a) TOP    Compact alkaline water-electrolysis reactor schematic for OUR
             system, focused on the hydrogen bubble at the CATHODE.  Each
             bubble is drawn as a spherical CAP clipped on the electrode
             wall (H2 on the cathode, smaller O2 on the anode).  A magnified
             circular inset zooms the cathode bubble and resolves the
             interfacial physics: the H2 bubble as a spherical CAP sitting
             ON the cathode surface, the curved liquid-vapour interface with
             H2O(l) <=> H2O(g) equilibrium, the surface tension sigma_lv, the
             true three-phase contact angle theta where the cap meets the
             electrode, and liquid-water flux arrows cut off at the bubble
             foot.  Two note boxes give the electrode half-reactions and the
             gas-phase (vapour) transport limit.

  (b) BOTTOM Coupled detachment model flow diagram: concentration
             polarisation -> m_s; mixture thermodynamics (e-NRTL, Butler)
             -> sigma_lv; applied potential (Young-Lippmann) -> theta;
             Young-Laplace shape + maximum closable volume -> V_d, D_d;
             Faradaic growth -> departure frequency f, gas rate; the bulk
             dielectrophoretic (Maxwell) force is the negligible branch.

The zoom is tied to panel (b) by a dashed connector from the inset interface
to the Young-Laplace shape box, and by matched colours: sigma_lv is drawn in
the mixture-thermodynamics blue and theta in the electrowetting purple in
BOTH the inset and the model boxes.

Run (from cases_mean_flow/equil_bubble):
    /home/endres/anaconda3/envs/ddg/bin/python manuscript/make_fig1_variantB.py

Outputs (vector PDF + PNG):
    manuscript/tex/figures/proposals/fig1_variantB.pdf
    manuscript/tex/figures/proposals/fig1_variantB.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, PathPatch, Circle, Rectangle, Arc, Wedge,
)

# fig_style lives in the case root (parent of manuscript/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_CASE_ROOT = os.path.dirname(_HERE)
if _CASE_ROOT not in sys.path:
    sys.path.insert(0, _CASE_ROOT)
from fig_style import apply_style  # noqa: E402

# ---- palette (shared with make_layers_diagram for cross-figure consistency) --
C_THERMO = "#3b6ea5"   # mixture thermodynamics -> sigma_lv   (blue)
C_FIELD = "#8a5a9e"     # electrowetting -> theta              (purple)
C_POL = "#2e8b8b"       # concentration polarisation -> m_s    (teal)
C_SHAPE = "#5a9e5a"     # Young-Laplace shape -> V_d, D_d      (green)
C_RATE = "#c46d2e"      # Faradaic growth -> f, rate           (orange)
C_DEP = "#9a9a9a"       # negligible bulk branch               (grey)
INK = "#1a1a1a"
HEAD = "white"

C_ELYTE = "#bcd7ea"     # electrolyte fill (light steel blue, slide-matched)
C_ELEC = "#c7c7c7"      # electrode grey
C_LIQ = "#123a5c"       # deep blue for liquid-phase text
C_FLUX = "#2a6f4f"      # green for reactant flux arrows
C_BLOCK = "#b5484b"     # muted red for the blocked-diffusion cut mark

# ---- chemical-formula strings (mathtext, reused from variant A) -------------
S_HER = r"Cathode (HER):  $\mathrm{2\,H_2O + 2\,e^- \rightarrow H_2 + 2\,OH^-}$"
S_OER = r"Anode (OER):  $\mathrm{4\,OH^- \rightarrow O_2 + 2\,H_2O + 4\,e^-}$"


# --------------------------------------------------------------------------- #
#  small drawing helpers
# --------------------------------------------------------------------------- #
def _arrow(ax, p0, p1, color=INK, lw=2.0, ls="-", conn="arc3,rad=0.0",
           mut=15, style="-|>", z=6):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=mut, linewidth=lw,
        color=color, linestyle=ls, shrinkA=0, shrinkB=0,
        connectionstyle=conn, zorder=z))


def _round_manhattan(ax, pts, r=0.20, color=INK, lw=1.8, ls="-", z=2):
    """Dashed orthogonal (Manhattan) polyline with slightly rounded
    right-angle corners, drawn as a single Path (a quadratic Bezier softens
    each corner).  Used for the crisp cross-panel bridge connector."""
    P = [np.asarray(p, float) for p in pts]
    verts = [P[0]]
    codes = [Path.MOVETO]
    for i in range(1, len(P) - 1):
        v_in = P[i] - P[i - 1]
        v_out = P[i + 1] - P[i]
        li = np.linalg.norm(v_in)
        lo = np.linalg.norm(v_out)
        rr = min(r, 0.5 * li, 0.5 * lo)
        a = P[i] - v_in / li * rr        # enter the corner
        b = P[i] + v_out / lo * rr       # leave the corner
        verts += [a, P[i], b]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    verts.append(P[-1])
    codes.append(Path.LINETO)
    ax.add_patch(PathPatch(Path(verts, codes), facecolor="none",
                           edgecolor=color, lw=lw, linestyle=ls, zorder=z,
                           capstyle="round", joinstyle="round"))


def _coil(ax, x0, x1, y, n=4, color=INK, lw=1.6, z=5):
    """Inductor-style coil: n humps along a horizontal segment."""
    w = (x1 - x0) / n
    for k in range(n):
        ax.add_patch(Arc((x0 + w / 2 + k * w, y), w, 1.15 * w,
                         theta1=0, theta2=180, lw=lw, color=color, zorder=z))


def _flow_box(ax, x, y, w, h, color, header, body, out,
              fs_head=8.6, fs_body=8.0, fs_out=8.8):
    """Rounded sub-model box: header strip, description, output variable."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=1.4, edgecolor=color, facecolor=color + "20", zorder=3))
    hh = 0.29 * h
    ax.add_patch(FancyBboxPatch(
        (x, y + h - hh), w, hh, boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=0, facecolor=color, zorder=4))
    ax.text(x + w / 2, y + h - hh / 2, header, color=HEAD,
            fontsize=fs_head, fontweight="bold", va="center", ha="center",
            zorder=5)
    ax.text(x + w / 2, y + 0.44 * h, body, color="#2b2b2b",
            fontsize=fs_body, va="center", ha="center", zorder=5,
            linespacing=1.2)
    ax.text(x + w / 2, y + 0.13 * h, out, color=color,
            fontsize=fs_out, fontweight="bold", va="center", ha="center",
            zorder=5)


# --------------------------------------------------------------------------- #
#  panel (a) : reactor cell schematic
# --------------------------------------------------------------------------- #
def _draw_cell(ax):
    cy0, cy1 = 8.45, 11.75         # cell outer y-range
    an0, an1 = 1.15, 1.42          # anode bar (left)
    ca0, ca1 = 4.28, 4.55          # cathode bar (right)

    # electrolyte fill between the electrodes
    elyte = Rectangle((an1, cy0), ca0 - an1, cy1 - cy0,
                      facecolor=C_ELYTE, edgecolor="none", zorder=1)
    ax.add_patch(elyte)
    # electrode bars
    ax.add_patch(Rectangle((an0, cy0), an1 - an0, cy1 - cy0,
                           facecolor=C_ELEC, edgecolor=INK, lw=1.0, zorder=2))
    ax.add_patch(Rectangle((ca0, cy0), ca1 - ca0, cy1 - cy0,
                           facecolor=C_ELEC, edgecolor=INK, lw=1.0, zorder=2))
    # cell outline
    ax.add_patch(Rectangle((an1, cy0), ca0 - an1, cy1 - cy0,
                           facecolor="none", edgecolor=INK, lw=1.2, zorder=3))

    xa = (an0 + an1) / 2
    xc = (ca0 + ca1) / 2

    # electrode captions
    ax.text(xa, cy0 - 0.24, "Anode", fontsize=9.5, ha="center", va="top",
            color=INK)
    ax.text((an1 + ca0) / 2, cy0 - 0.24, "Electrolyte", fontsize=9.5,
            ha="center", va="top", color=INK)
    ax.text(xc, cy0 - 0.24, "Cathode", fontsize=9.8, ha="center", va="top",
            color=INK, fontweight="bold")

    # external circuit -----------------------------------------------------
    ytop = 12.28
    ax.plot([xa, xa], [cy1, ytop], color=INK, lw=1.4, zorder=4)
    ax.plot([xc, xc], [cy1, ytop], color=INK, lw=1.4, zorder=4)
    ax.plot([xa, 2.30], [ytop, ytop], color=INK, lw=1.4, zorder=4)
    ax.plot([3.30, xc], [ytop, ytop], color=INK, lw=1.4, zorder=4)
    _coil(ax, 2.30, 3.30, ytop, n=4, color=INK, lw=1.5)
    ax.text((xa + xc) / 2, ytop + 0.34, "Electric current",
            fontsize=9.8, ha="center", va="center", color=INK)
    _arrow(ax, (1.62, ytop), (2.02, ytop), color=INK, lw=1.6, mut=12)
    _arrow(ax, (xc, ytop - 0.30), (xc, ytop - 0.86), color=INK, lw=1.6, mut=12)
    ax.text(xa - 0.16, cy1 + 0.58, r"$e^-$", fontsize=9.2, ha="right",
            va="center", color=INK)
    ax.text(xc + 0.16, ytop - 0.58, r"$e^-$", fontsize=9.2, ha="left",
            va="center", color=INK)

    # bulk liquid labels ---------------------------------------------------
    ax.text((an1 + ca0) / 2, 11.42, r"H$_2$O$(l)$,  aqueous KOH",
            fontsize=8.8, ha="center", va="center", color=C_LIQ)
    ax.text((an1 + ca0) / 2, 11.08, r"(K$^+$,  OH$^-$)", fontsize=8.6,
            ha="center", va="center", color=C_LIQ)

    # O2 bubble on the ANODE wall: spherical cap (right half-disc) ----------
    ax.add_patch(Wedge((an1, 10.28), 0.30, -90, 90, facecolor="white",
                       edgecolor=INK, lw=1.1, zorder=4))
    ax.text(an1 + 0.42, 10.28, r"O$_2(g)$", fontsize=8.8, ha="left",
            va="center", color=INK)

    # H2 bubble on the CATHODE wall: spherical cap (left half-disc) ---------
    hbc, hbr = (ca0, 10.55), 0.55
    ax.add_patch(Wedge(hbc, hbr, 90, 270, facecolor="white",
                       edgecolor=INK, lw=1.4, zorder=4))
    ax.text(3.28, 10.55, r"H$_2(g)$", fontsize=9.2, ha="right",
            va="center", color=INK, fontweight="bold")

    # ion migration: OH- toward the anode (alkaline); H+ if acidic ---------
    _arrow(ax, (3.95, 9.10), (1.95, 9.10), color=C_LIQ, lw=1.7, mut=13)
    ax.text((an1 + ca0) / 2, 9.42, r"OH$^-$ migration (anions $\rightarrow$ anode)",
            fontsize=7.7, ha="center", va="center", color=C_LIQ)
    ax.text((an1 + ca0) / 2, 8.78,
            r"(acidic H$_2$SO$_4$: H$^+$ migrates $\rightarrow$ cathode)",
            fontsize=6.9, ha="center", va="center", color="#5a5a5a",
            fontstyle="italic")

    # magnifier: small dashed circle around the cathode bubble -------------
    mc = (ca0 - 0.20, 10.55)
    mr = 0.62
    ax.add_patch(Circle(mc, mr, facecolor="none", edgecolor="#444444",
                        lw=1.1, ls=(0, (4, 3)), zorder=6))
    return mc, mr


# --------------------------------------------------------------------------- #
#  panel (a) : magnified inset on the cathode bubble interface
# --------------------------------------------------------------------------- #
def _draw_zoom(ax, mc, mr):
    cx, cy, R = 7.25, 9.95, 2.00   # inset circle
    # boundary circle: added up front so it carries a valid transform for the
    # clip paths below; its high zorder keeps the outline drawn on top.
    boundary = Circle((cx, cy), R, facecolor="none", edgecolor=INK,
                      lw=1.8, zorder=8)
    ax.add_patch(boundary)

    # magnifier cone from the small cell circle to the inset ---------------
    ax.plot([mc[0] + mr * 0.72, cx - R * 0.80],
            [mc[1] + mr * 0.66, cy + R * 0.62],
            color="#444444", lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.plot([mc[0] + mr * 0.72, cx - R * 0.80],
            [mc[1] - mr * 0.66, cy - R * 0.62],
            color="#444444", lw=1.0, ls=(0, (4, 3)), zorder=2)

    # liquid fill (whole disc), electrode slab, bubble gas -----------------
    liq = Circle((cx, cy), R, facecolor=C_ELYTE, edgecolor="none", zorder=2)
    ax.add_patch(liq)
    liq.set_clip_path(boundary)

    wx = cx + 0.72                 # electrode surface (vertical wall)
    bx, by, rb = wx + 0.35, cy, 1.25   # bubble circle (spherical cap on wall)

    bub = Circle((bx, by), rb, facecolor="#fcfdff", edgecolor="none", zorder=3)
    ax.add_patch(bub)
    bub.set_clip_path(boundary)

    elec = Rectangle((wx, cy - R), (cx + R) - wx, 2 * R,
                     facecolor=C_ELEC, edgecolor="none", zorder=4)
    ax.add_patch(elec)
    elec.set_clip_path(boundary)

    # bubble feet on the wall
    dy = np.sqrt(rb ** 2 - (wx - bx) ** 2)
    foot_lo = (wx, by - dy)
    foot_hi = (wx, by + dy)
    ang_foot = np.degrees(np.arccos((wx - bx) / rb))

    # electrode surface: straight where the bubble sits, rough above/below --
    for (ya, yb) in [(foot_hi[1], cy + R), (cy - R, foot_lo[1])]:
        ys = np.linspace(ya, yb, 90)
        xs = wx + 0.05 * np.sin(15.0 * (ys - cy))
        ln, = ax.plot(xs, ys, color="#7f7f7f", lw=1.0, zorder=5)
        ln.set_clip_path(boundary)
    wall, = ax.plot([wx, wx], [foot_lo[1], foot_hi[1]], color=INK, lw=1.6,
                    zorder=5)
    wall.set_clip_path(boundary)

    # liquid-vapour interface (left arc of the bubble) --------------------
    tt = np.radians(np.linspace(ang_foot, 360 - ang_foot, 200))
    itf, = ax.plot(bx + rb * np.cos(tt), by + rb * np.sin(tt), color=INK,
                   lw=2.6, zorder=6, solid_capstyle="round")
    itf.set_clip_path(boundary)

    # inside / outside phase labels ---------------------------------------
    ax.text(7.52, cy + 0.24, r"H$_2(g)$", fontsize=9.4, ha="center",
            va="center", color=INK, fontweight="bold", zorder=7)
    ax.text(7.52, cy - 0.24, r"$+\,$H$_2$O$(g)$", fontsize=8.2, ha="center",
            va="center", color=INK, zorder=7)
    ax.text(8.62, cy, "Cathode", fontsize=8.6, ha="center",
            va="center", color="#4f4f4f", rotation=90, zorder=7)

    # (1) liquid-water flux: parallel arrows cut off at the bubble foot -----
    x0 = 5.72
    yq_hi, yq_lo = cy + 0.30, cy - 0.30
    for yq in (yq_hi, yq_lo):
        xi = bx - np.sqrt(rb ** 2 - (yq - by) ** 2)      # interface x at yq
        _arrow(ax, (x0, yq), (xi - 0.02, yq), color=C_FLUX, lw=1.7, mut=12)
        # red cut mark at the blocked tip
        ax.plot([xi - 0.10, xi + 0.14], [yq - 0.15, yq + 0.15],
                color=C_BLOCK, lw=2.2, zorder=9)
        ax.plot([xi - 0.10, xi + 0.14], [yq + 0.15, yq - 0.15],
                color=C_BLOCK, lw=2.2, zorder=9)
    ax.text(6.20, cy + 0.66, r"H$_2$O$(l)$ flux",
            fontsize=8.2, color=C_FLUX, ha="center", va="center", zorder=9)

    # (2) surface tension sigma_lv : tangent double arrow, THERMO colour ----
    a = np.radians(135.0)
    p = np.array([bx + rb * np.cos(a), by + rb * np.sin(a)])
    tvec = np.array([-np.sin(a), np.cos(a)])
    _arrow(ax, p - 0.38 * tvec, p + 0.38 * tvec, color=C_THERMO, lw=2.2,
           mut=13, style="<|-|>", z=8)
    ax.text(p[0] - 0.32, p[1] + 0.34, r"$\sigma_{lv}$", fontsize=11.5,
            color=C_THERMO, ha="center", va="center", fontweight="bold",
            zorder=9)

    # (3) VLE : H2O(l) <=> H2O(g) across the interface ---------------------
    a = np.radians(216.0)
    p = np.array([bx + rb * np.cos(a), by + rb * np.sin(a)])
    nvec = np.array([np.cos(a), np.sin(a)])            # outward, into liquid
    _arrow(ax, p - 0.30 * nvec, p + 0.30 * nvec, color="#2f2f2f", lw=1.8,
           mut=12, style="<|-|>", z=8)
    ax.text(6.28, 8.98, r"H$_2$O$(l)\rightleftharpoons$H$_2$O$(g)$",
            fontsize=7.8, color="#1f1f1f", ha="center", va="center", zorder=9)
    ax.text(6.28, 8.70, "(interfacial VLE)", fontsize=7.0, color="#555555",
            ha="center", va="center", zorder=9, fontstyle="italic")

    # (4) contact angle theta at the lower foot, FIELD colour --------------
    foot_ang = np.degrees(np.arctan2(foot_lo[1] - by, wx - bx)) % 360.0
    tang_ang = (foot_ang - 90.0) % 360.0               # interface tangent
    ax.add_patch(Arc(foot_lo, 0.95, 0.95, angle=0.0, theta1=tang_ang,
                     theta2=270.0, color=C_FIELD, lw=1.9, zorder=8))
    tm = np.radians(0.5 * (tang_ang + 270.0))
    ax.text(foot_lo[0] + 0.66 * np.cos(tm), foot_lo[1] + 0.66 * np.sin(tm),
            r"$\theta$", fontsize=11.5, color=C_FIELD, ha="center",
            va="center", fontweight="bold", zorder=9)

    # start point on the boundary for the cross-panel bridge (lower-right)
    br = np.radians(-35.0)
    bridge_start = (cx + R * np.cos(br), cy + R * np.sin(br))
    return bridge_start


# --------------------------------------------------------------------------- #
#  panel (a) : bottom note boxes (half-reactions + transport limit)
# --------------------------------------------------------------------------- #
def _draw_notes(ax):
    # electrode half-reactions (bottom-left) ------------------------------
    rx, ry, rw, rh = 0.55, 6.20, 4.05, 1.55
    ax.add_patch(FancyBboxPatch(
        (rx, ry), rw, rh, boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=1.1, edgecolor="#9fb8cf", facecolor="#eef4fa", zorder=2))
    ax.text(rx + rw / 2, ry + rh - 0.30, "Electrode half-reactions",
            fontsize=8.8, fontweight="bold", ha="center", va="center",
            color=INK, zorder=3)
    ax.text(rx + 0.22, ry + 0.92, S_HER, fontsize=7.9, ha="left",
            va="center", color=INK, zorder=3)
    ax.text(rx + 0.22, ry + 0.44, S_OER, fontsize=7.9, ha="left",
            va="center", color=INK, zorder=3)

    # gas-phase transport limit (bottom-centre) ---------------------------
    tx, ty, tw, th = 4.72, 6.20, 2.85, 1.55
    ax.add_patch(FancyBboxPatch(
        (tx, ty), tw, th, boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=1.1, edgecolor="#b7d0bf", facecolor="#eef6f0", zorder=2))
    ax.text(tx + tw / 2, ty + th - 0.30, "Gas-phase transport limit",
            fontsize=8.6, fontweight="bold", ha="center", va="center",
            color="#1f5c3d", zorder=3)
    ax.text(tx + tw / 2, ty + 0.52,
            "Bubble covers the site: H$_2$O\n"
            "reaches the electrode only as\n"
            "vapour (gas-phase diffusion),\n"
            "so growth is transport-limited.",
            fontsize=7.3, ha="center", va="center", color="#2b2b2b",
            zorder=3, linespacing=1.32)


# --------------------------------------------------------------------------- #
#  panel (b) : coupled model flow diagram
# --------------------------------------------------------------------------- #
def _draw_model(ax):
    def tag(x, y, txt):
        ax.text(x, y, txt, fontsize=8.8, fontweight="bold", ha="center",
                va="center", color=INK, linespacing=1.15)

    # sub-model column (whole panel shifted up to close the title gap)
    sx, sw = 1.30, 2.85
    cy0, ch = 4.40, 0.92           # concentration polarisation
    ty0, th = 3.05, 1.07           # mixture thermodynamics
    ey0, eh = 1.73, 0.98           # electrowetting
    cyc, tyc, eyc = cy0 + ch / 2, ty0 + th / 2, ey0 + eh / 2

    tag(0.68, cyc, "Current\ndensity $j$")
    tag(0.68, tyc, "Salt molality\n$m$ (KOH)")
    tag(0.68, eyc, "Applied\npotential $E$")

    _flow_box(ax, sx, cy0, sw, ch, C_POL, "Concentration polarisation",
              "1D Nernst diffusion layer", r"$\rightarrow\ m_s$")
    _flow_box(ax, sx, ty0, sw, th, C_THERMO, "Mixture thermodynamics",
              "Electrolyte-NRTL activities,\nButler surface tension",
              r"$\rightarrow\ \sigma_{lv}(m)$  (mN m$^{-1}$)")
    _flow_box(ax, sx, ey0, sw, eh, C_FIELD, "Electrowetting",
              "Young-Lippmann relation",
              r"$\rightarrow\ \theta$  (contact angle)")

    # detachment box (sits under the inset) -------------------------------
    # top edge aligned with the concentration-polarisation box top (cy0 + ch);
    # box is SHORTENED so the dielectrophoretic branch below can rise to sit
    # level with the electrowetting box while keeping the F_DEP link.
    dx0, dw = 5.05, 2.25
    dy0, dh = 3.17, 2.15
    _flow_box(ax, dx0, dy0, dw, dh, C_SHAPE, "Young-Laplace shape",
              "Pinned axisymmetric\nprofile; detachment =\nmaximum closable\n"
              r"pinned volume ($F_b$ vs" + "\npinning force)",
              r"$\rightarrow\ V_d,\ D_d$  (m$^3$, m)", fs_body=8.0)

    # Faradaic growth box -------------------------------------------------
    fx0, fw = 8.00, 1.75
    fy0, fh = 3.00, 1.10
    _flow_box(ax, fx0, fy0, fw, fh, C_RATE, "Faradaic growth",
              r"Gas flux $\propto j$;" + "\n" + r"$f = Q / V_d$",
              r"$\rightarrow\ f$ (Hz), rate", fs_body=8.0)

    # negligible dielectrophoretic branch ---------------------------------
    # raised into the space freed by shortening the Young-Laplace box: its
    # bottom edge now sits level with the electrowetting box bottom (ey0)
    gx0, gw = 5.05, 2.25
    gy0, gh = ey0, 1.08
    ax.add_patch(FancyBboxPatch(
        (gx0, gy0), gw, gh, boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=1.2, edgecolor=C_DEP, facecolor=C_DEP + "18",
        linestyle="--", zorder=3))
    ax.text(gx0 + gw / 2, gy0 + gh - 0.40,
            "Bulk dielectrophoretic\nforce (Maxwell)",
            color="#5a5a5a", fontsize=7.9, fontweight="bold",
            ha="center", va="center", zorder=5, linespacing=1.2)
    ax.text(gx0 + gw / 2, gy0 + 0.30,
            r"$F_{DEP}/F_b \sim 10^{-9}\ \Rightarrow$ negligible",
            color="#5a5a5a", fontsize=8.0, ha="center", va="center", zorder=5)

    # input arrows --------------------------------------------------------
    for yv in (cyc, tyc, eyc):
        _arrow(ax, (0.95, yv), (sx, yv), color="#555555", lw=1.5, mut=12)

    # coupling arrows -----------------------------------------------------
    mcx = sx + sw / 2
    _arrow(ax, (mcx, cy0), (mcx, ty0 + th), color=C_POL, lw=1.9, mut=13)
    ax.text(mcx + 0.14, (cy0 + ty0 + th) / 2, r"$m_s$", color=C_POL,
            fontsize=8.8, va="center", ha="left", fontweight="bold")

    _arrow(ax, (sx + sw, tyc), (dx0, dy0 + dh * 0.58), color=C_THERMO,
           lw=2.2, mut=14, conn="arc3,rad=-0.10")
    ax.text(sx + sw + 0.52, dy0 + dh * 0.64, r"$\sigma_{lv}$", color=C_THERMO,
            fontsize=9.6, va="center", ha="center", fontweight="bold")

    _arrow(ax, (sx + sw, eyc), (dx0, dy0 + dh * 0.16), color=C_FIELD,
           lw=2.2, mut=14, conn="arc3,rad=0.08")
    ax.text(sx + sw + 0.52, dy0 + dh * 0.22, r"$\theta$", color=C_FIELD,
            fontsize=10, va="center", ha="center", fontweight="bold")

    _arrow(ax, (dx0 + dw, dy0 + dh * 0.5), (fx0, fy0 + fh * 0.5),
           color=C_SHAPE, lw=2.2, mut=14)
    ax.text((dx0 + dw + fx0) / 2, dy0 + dh * 0.5 + 0.26, r"$V_d,\ D_d$",
            color=C_SHAPE, fontsize=8.6, va="center", ha="center",
            fontweight="bold")

    _arrow(ax, (fx0 + fw * 0.5, fy0), (fx0 + fw * 0.5, fy0 - 0.62),
           color=C_RATE, lw=1.9, mut=13)
    ax.text(fx0 + fw * 0.5, fy0 - 0.94,
            "departure frequency,\ngas production rate",
            fontsize=8.2, color=C_RATE, fontweight="bold",
            va="center", ha="center", linespacing=1.15)

    _arrow(ax, (gx0 + gw * 0.5, gy0 + gh), (dx0 + dw * 0.5, dy0),
           color=C_DEP, lw=1.5, ls="--", mut=12)
    ax.text(dx0 + dw * 0.5 + 0.28, (gy0 + gh + dy0) / 2, r"$F_{DEP}$",
            color=C_DEP, fontsize=8.2, va="center", ha="left")

    return (dx0, dy0, dw, dh)      # detachment box for the cross-panel link


# --------------------------------------------------------------------------- #
#  assembly
# --------------------------------------------------------------------------- #
def main() -> tuple[str, str]:
    apply_style()

    W, H = 10.0, 13.4
    fig, ax = plt.subplots(figsize=(W, H), dpi=150)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # panel (a) title ------------------------------------------------------
    ax.text(0.15, 13.08, "(a)", fontsize=13, fontweight="bold", ha="left",
            va="center", color=INK)
    ax.text(5.05, 13.08,
            "Alkaline water electrolysis: hydrogen bubble at the cathode",
            fontsize=12.5, fontweight="bold", ha="center", va="center",
            color=INK)

    mc, mr = _draw_cell(ax)
    bridge_start = _draw_zoom(ax, mc, mr)
    _draw_notes(ax)

    # small inset heading
    ax.text(7.25, 12.28, "Magnified cathode interface", fontsize=9.4,
            ha="center", va="center", color="#333333", fontstyle="italic")

    # divider between the system (a) and the model (b)
    ax.plot([0.3, W - 0.3], [6.05, 6.05], color="#cccccc", lw=1.0, zorder=1)

    # panel (b) title ------------------------------------------------------
    ax.text(0.15, 5.78, "(b)", fontsize=13, fontweight="bold", ha="left",
            va="center", color=INK)
    ax.text(5.05, 5.78,
            "Coupled model: interfacial variables set the bubble departure",
            fontsize=12.5, fontweight="bold", ha="center", va="center",
            color=INK)

    dx0, dy0, dw, dh = _draw_model(ax)

    # cross-panel link: inset interface -> Young-Laplace shape box.  Metropolis
    # routing: exit the zoom to the right, drop straight down the right margin,
    # and turn horizontally into the box, with softly rounded right angles.
    land = (dx0 + dw, 5.00)                     # right edge of the box, near top
    x_margin = 9.50
    waypts = [
        (bridge_start[0], bridge_start[1]),     # zoom lower-right boundary
        (x_margin, bridge_start[1]),            # jog right to the margin
        (x_margin, land[1]),                    # drop down the margin
        (land[0] + 0.42, land[1]),              # come back left toward the box
    ]
    _round_manhattan(ax, waypts, r=0.22, color="#666666", lw=1.8,
                     ls=(0, (5, 3)), z=2)
    _arrow(ax, (land[0] + 0.42, land[1]), land, color="#666666", lw=1.8,
           ls=(0, (5, 3)), mut=15, z=2)

    # annotation for the bridge, now in panel (a): between the gas-phase
    # transport-limit box (right edge ~7.57) and the bridge's descent margin.
    ax.text(8.45, 6.98, "Young-Laplace shape\nof this interface",
            fontsize=8.4, color="#555555", ha="center", va="center",
            fontstyle="italic", linespacing=1.2, zorder=3)

    # outputs --------------------------------------------------------------
    out_pdf = os.path.join(_HERE, "tex", "figures", "proposals",
                           "fig1_variantB.pdf")
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
