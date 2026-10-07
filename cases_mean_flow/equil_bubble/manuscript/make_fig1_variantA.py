"""
make_fig1_variantA.py
=====================

Figure 1 (variant A) for the single-bubble water-electrolysis manuscript.

FAITHFUL & COMPLETE two-panel schematic:

  (a) TOP   Alkaline (KOH) water-electrolysis cell, drawn in the style of the
            source slides: anode | electrolyte | cathode, external electric
            current loop (e- arrows + coil), an O2 bubble at the anode and the
            focus H2 bubble at the cathode.  The curved bubble interface carries
            the vapour-liquid equilibrium  H2O(l) <=> H2O(g); the dissolved salt
            (K+, OH-) sets the activities and the surface tension sigma_lv.
            Diffusion of liquid water toward the electrode is blocked/limited by
            the bubble; OH- migrates toward the anode and e- flows in the
            external circuit.  (Acidic H2SO4 route annotated as the alternative.)

  (b) BELOW The coupled sub-model pipeline that predicts detachment:
            mixture thermodynamics (e-NRTL, Butler) -> sigma_lv(m);
            applied potential -> contact angle theta (Young-Lippmann);
            sigma_lv & theta -> Young-Laplace shape + maximum-closable-volume
            detachment (buoyancy F_b vs pinning F_pin) -> V_d, D_d;
            Faradaic growth -> departure frequency f and gas-production rate;
            bulk dielectrophoretic (Maxwell) branch weighed negligible.

The two panels are connected: a bridge arrow runs DOWN from the H2 bubble in (a)
into the detachment model in (b); the loop is closed by a return arrow that runs
UP the right margin from the predicted gas-production rate to the reactor cathode.

Run (from cases_mean_flow/equil_bubble):
    /home/endres/anaconda3/envs/ddg/bin/python manuscript/make_fig1_variantA.py

Outputs (vector PDF + PNG):
    manuscript/tex/figures/proposals/fig1_variantA.pdf
    manuscript/tex/figures/proposals/fig1_variantA.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Wedge, Arc

# ---- shared manuscript style (serif + STIX math) --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CASE_ROOT = os.path.dirname(_HERE)
if _CASE_ROOT not in sys.path:
    sys.path.insert(0, _CASE_ROOT)
from fig_style import apply_style  # noqa: E402

# ---- palette ---------------------------------------------------------------
C_ELYTE = "#bcd8ec"   # electrolyte fill (slide aesthetic)
C_ELEC = "#c9c9c9"    # electrode gray
C_BUB = "#ffffff"     # bubble gas fill
INK = "#1a1a1a"
C_WATER = "#2b6ea5"   # liquid-water diffusion arrows
C_BLOCK = "#c0392b"   # blocked marker (red)

# model sub-models (coupling-variable colours are shared with panel (a))
C_THERMO = "#3b6ea5"  # mixture thermodynamics -> sigma_lv   (blue)
C_FIELD = "#8a5a9e"   # electrowetting -> contact angle      (purple)
C_POL = "#2e8b8b"     # concentration polarisation
C_SHAPE = "#5a9e5a"   # Young-Laplace shape + detachment
C_RATE = "#c46d2e"    # Faradaic growth -> rate / feedback loop
C_DEP = "#9a9a9a"     # negligible bulk branch
C_BRIDGE = "#33475b"  # inter-panel bridge

# ---- chemical-formula strings (mathtext) -----------------------------------
S_H2O_l = r"$\mathrm{H_2O_{(l)}}$"
S_H2O_g = r"$\mathrm{H_2O_{(g)}}$"
S_H2 = r"$\mathrm{H_2\,(g)}$"
S_O2 = r"$\mathrm{O_2\,(g)}$"
S_OH = r"$\mathrm{OH^-}$"
S_K = r"$\mathrm{K^+}$"
S_e = r"$\mathrm{e^-}$"
S_sig = r"$\sigma_{lv}$"
S_VLE = r"$\mathrm{H_2O_{(l)}}\ \rightleftharpoons\ \mathrm{H_2O_{(g)}}$"
S_HER = r"Cathode (HER):  $\mathrm{2\,H_2O + 2\,e^- \rightarrow H_2 + 2\,OH^-}$"
S_OER = r"Anode (OER):  $\mathrm{4\,OH^- \rightarrow O_2 + 2\,H_2O + 4\,e^-}$"


# ======================================================================
#  small helpers
# ======================================================================
def _arrow(ax, p0, p1, color=INK, lw=2.0, ls="-", ms=15, conn="arc3,rad=0.0"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        color=color, linestyle=ls, shrinkA=1, shrinkB=1,
        connectionstyle=conn))


def _coil(ax, x0, x1, y, n=4, h=0.42, color=INK, lw=1.6):
    """Inductor-style coil (n up-bumps) on the top wire."""
    w = (x1 - x0) / n
    for i in range(n):
        cx = x0 + w * (i + 0.5)
        ax.add_patch(Arc((cx, y), width=w, height=h, angle=0,
                         theta1=0, theta2=180, color=color, lw=lw))


def _box(ax, x, y, w, h, color, header, body, out, fs_head=9.0,
         fs_body=8.4, fs_out=9.2, ls="-"):
    """Rounded sub-model box: header strip, description, output variable."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=1.5, edgecolor=color, facecolor=color + "20", linestyle=ls))
    hh = 0.26 * h
    ax.add_patch(FancyBboxPatch(
        (x, y + h - hh), w, hh, boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=0, facecolor=color))
    ax.text(x + w / 2, y + h - hh / 2, header, color="white",
            fontsize=fs_head, fontweight="bold", va="center", ha="center")
    ax.text(x + w / 2, y + 0.44 * h, body, color="#2b2b2b",
            fontsize=fs_body, va="center", ha="center", linespacing=1.25)
    ax.text(x + w / 2, y + 0.13 * h, out, color=color,
            fontsize=fs_out, fontweight="bold", va="center", ha="center")


def _tag(ax, x, y, text, color=INK):
    ax.text(x, y, text, color=color, fontsize=9.2, fontweight="bold",
            va="center", ha="center", linespacing=1.2)


# ======================================================================
#  panel (a): the electrolysis cell
# ======================================================================
def draw_reactor(ax):
    CELL_L, CELL_R = 3.2, 10.8
    CELL_BOT, CELL_TOP = 11.05, 15.55
    AN_R, CA_L = 3.9, 10.1
    WIRE_L, WIRE_R = 3.55, 10.45
    Y_WIRE = 16.3

    # electrodes + electrolyte ----------------------------------------
    ax.add_patch(Rectangle((CELL_L, CELL_BOT), AN_R - CELL_L,
                           CELL_TOP - CELL_BOT, facecolor=C_ELEC,
                           edgecolor=INK, lw=1.3, zorder=2))
    ax.add_patch(Rectangle((CA_L, CELL_BOT), CELL_R - CA_L,
                           CELL_TOP - CELL_BOT, facecolor=C_ELEC,
                           edgecolor=INK, lw=1.3, zorder=2))
    ax.add_patch(Rectangle((AN_R, CELL_BOT), CA_L - AN_R,
                           CELL_TOP - CELL_BOT, facecolor=C_ELYTE,
                           edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((CELL_L, CELL_BOT), CELL_R - CELL_L,
                           CELL_TOP - CELL_BOT, facecolor="none",
                           edgecolor=INK, lw=1.6, zorder=3))

    # external circuit -------------------------------------------------
    ax.plot([WIRE_L, WIRE_L], [CELL_TOP, Y_WIRE], color=INK, lw=1.6, zorder=2)
    ax.plot([WIRE_R, WIRE_R], [CELL_TOP, Y_WIRE], color=INK, lw=1.6, zorder=2)
    ax.plot([WIRE_L, 6.35], [Y_WIRE, Y_WIRE], color=INK, lw=1.6, zorder=2)
    ax.plot([7.65, WIRE_R], [Y_WIRE, Y_WIRE], color=INK, lw=1.6, zorder=2)
    _coil(ax, 6.35, 7.65, Y_WIRE, n=4)
    ax.text(7.0, 16.78, "Electric current", fontsize=10.5, ha="center",
            va="center", color=INK)
    _arrow(ax, (4.9, Y_WIRE), (5.6, Y_WIRE), color=INK, lw=1.6, ms=13)
    _arrow(ax, (WIRE_L, 15.62), (WIRE_L, 16.18), color=INK, lw=2.0, ms=14)
    _arrow(ax, (WIRE_R, 16.18), (WIRE_R, 15.62), color=INK, lw=2.0, ms=14)
    ax.text(WIRE_L - 0.26, 15.95, S_e, fontsize=10, ha="right", va="center")
    ax.text(WIRE_R + 0.26, 15.95, S_e, fontsize=10, ha="left", va="center")

    # bubbles ----------------------------------------------------------
    o2c, o2r = (AN_R, 12.6), 0.72
    ax.add_patch(Wedge(o2c, o2r, -90, 90, facecolor=C_BUB,
                      edgecolor=INK, lw=1.4, zorder=4))
    ax.text(4.24, 12.6, S_O2, fontsize=8.0, ha="center", va="center",
            zorder=5)

    h2c, h2r = (CA_L, 13.5), 1.35
    ax.add_patch(Wedge(h2c, h2r, 90, 270, facecolor=C_BUB,
                      edgecolor=INK, lw=1.9, zorder=4))
    ax.text(9.48, 13.96, S_H2, fontsize=11, ha="center", va="center",
            fontweight="bold", zorder=5)
    ax.text(9.48, 13.50, "+ " + S_H2O_g, fontsize=9.4, ha="center",
            va="center", zorder=5)
    ax.text(9.48, 13.14, "(water vapour)", fontsize=7.8, ha="center",
            va="center", style="italic", color="#444444", zorder=5)

    # bulk-electrolyte species label ----------------------------------
    ax.text(6.9, 15.18, S_H2O_l + r"    aqueous KOH   (" + S_K + ", " + S_OH +
            ")", fontsize=9.4, ha="center", va="center", zorder=5)

    # liquid-water diffusion toward the cathode, blocked by the bubble
    for yd, xarc in ((14.35, 9.051), (13.95, 8.827)):
        _arrow(ax, (6.75, yd), (xarc - 0.05, yd), color=C_WATER, lw=1.8, ms=12)
        ax.plot([xarc + 0.03, xarc + 0.03], [yd - 0.16, yd + 0.16],
                color=C_BLOCK, lw=2.8, zorder=6, solid_capstyle="round")
    ax.text(5.35, 14.15, "Liquid-water\ndiffusion limited\nby the bubble",
            fontsize=7.8, ha="center", va="center", color=C_WATER,
            linespacing=1.2, zorder=5)

    # curved VLE interface: sigma_lv on the arc (mid-height, clear of arrows)
    ax.plot([8.52, 8.74], [13.5, 13.5], color=C_THERMO, lw=1.0, zorder=5)
    ax.text(8.48, 13.5, S_sig, fontsize=11, ha="right", va="center",
            color=C_THERMO, fontweight="bold", zorder=6)
    # explanatory VLE callout with a leader to the lower-left arc
    ax.plot([8.10, 8.83], [12.68, 13.00], color="#555555", lw=0.9, zorder=5)
    ax.text(6.15, 12.55,
            "Curved interface (VLE):\n" + S_VLE + "\n"
            r"salt sets activities $\rightarrow\ \sigma_{lv}$",
            fontsize=8.0, ha="center", va="center", color="#333333",
            linespacing=1.3, zorder=5)

    # OH- migration toward the anode (below both bubbles)
    _arrow(ax, (8.30, 11.52), (5.05, 11.52), color=INK, lw=2.0, ms=15)
    ax.text(6.40, 11.82, S_OH + " migration  (anions " + r"$\rightarrow$" +
            " anode)", fontsize=8.8, ha="center", va="center", zorder=5)

    ax.text(6.20, 11.30, "acidic route (" + r"$\mathrm{H_2SO_4}$" + "): " +
            r"$\mathrm{H^+}$" + " migrates " + r"$\rightarrow$" + " cathode",
            fontsize=7.6, ha="center", va="center", style="italic",
            color="#6a6a6a", zorder=5)

    # ground labels + ticks -------------------------------------------
    for xc, lab in ((3.55, "Anode"), (7.0, "Electrolyte"), (10.45, "Cathode")):
        ax.plot([xc, xc], [10.86, CELL_BOT], color=INK, lw=1.0)
        ax.text(xc, 10.66, lab, fontsize=10, ha="center", va="top")

    # half-reactions (bottom-left, clear of the bridge) ---------------
    ax.text(0.55, 10.25, S_HER, fontsize=8.4, ha="left", va="center")
    ax.text(0.55, 9.90, S_OER, fontsize=8.4, ha="left", va="center")

    ax.text(0.35, 15.5, "(a)", fontsize=13, fontweight="bold", ha="left",
            va="center")

    # arc start point for the bridge (lower-left of the H2 bubble)
    ang = np.deg2rad(215.0)
    h2_arc = (CA_L + h2r * np.cos(ang), 13.5 + h2r * np.sin(ang))
    return dict(h2_arc=h2_arc, cathode_feed=(10.82, 12.4))


# ======================================================================
#  panel (b): the coupled model pipeline
# ======================================================================
def draw_model(ax):
    MTx, MTy, MTw, MTh = 1.70, 6.55, 3.75, 1.30    # mixture thermodynamics
    CPx, CPy, CPw, CPh = 1.70, 4.55, 3.75, 1.12    # concentration polarisation
    EWx, EWy, EWw, EWh = 1.70, 2.50, 3.75, 1.12    # electrowetting
    DTx, DTy, DTw, DTh = 6.25, 3.15, 3.50, 3.60    # Young-Laplace detachment
    FGx, FGy, FGw, FGh = 10.75, 4.35, 2.80, 1.40   # Faradaic growth
    DPx, DPy, DPw, DPh = 6.25, 1.05, 3.50, 1.35    # bulk DEP (negligible)

    _box(ax, MTx, MTy, MTw, MTh, C_THERMO, "Mixture thermodynamics",
         "e-NRTL activities,\nButler surface tension",
         r"$\rightarrow\ \sigma_{lv}(m)$  (mN m$^{-1}$)")
    _box(ax, CPx, CPy, CPw, CPh, C_POL, "Concentration polarisation",
         "1D Nernst diffusion layer",
         r"$\rightarrow\ m_s$  (surface molality)")
    _box(ax, EWx, EWy, EWw, EWh, C_FIELD, "Electrowetting",
         "Young-Lippmann relation\n(saturating)",
         r"$\rightarrow\ \theta$  (contact angle)")
    _box(ax, DTx, DTy, DTw, DTh, C_SHAPE, "Young-Laplace shape",
         "Pinned axisymmetric profile;\n"
         "detachment = maximum\nclosable pinned volume\n"
         r"(buoyancy $F_b$ vs pinning $F_{pin}$)",
         r"$\rightarrow\ V_d,\ D_d$  (m$^3$, m)", fs_body=8.2)
    _box(ax, FGx, FGy, FGw, FGh, C_RATE, "Faradaic growth",
         r"Gas flux $\propto j$;   $f = Q/V_d$",
         r"$\rightarrow\ f$  (Hz), rate")

    ax.add_patch(FancyBboxPatch(
        (DPx, DPy), DPw, DPh, boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=1.3, edgecolor=C_DEP, facecolor=C_DEP + "18", linestyle="--"))
    ax.text(DPx + DPw / 2, DPy + DPh - 0.40,
            "Bulk dielectrophoretic force (Maxwell stress)",
            color="#5a5a5a", fontsize=8.6, fontweight="bold",
            va="center", ha="center")
    ax.text(DPx + DPw / 2, DPy + 0.40,
            r"$F_{DEP}/F_b \sim 10^{-9}$   $\Rightarrow$   negligible",
            color="#5a5a5a", fontsize=8.8, va="center", ha="center")

    # input tags (left) ------------------------------------------------
    _tag(ax, 0.90, MTy + MTh / 2, "Salt,\nmolality $m$")
    _tag(ax, 0.90, CPy + CPh / 2, "Current\ndensity $j$")
    _tag(ax, 0.90, EWy + EWh / 2, "Applied\npotential $E$")
    for yc in (MTy + MTh / 2, CPy + CPh / 2, EWy + EWh / 2):
        _arrow(ax, (1.40, yc), (MTx - 0.02, yc), color="#666666", lw=1.5, ms=12)

    # coupling arrows --------------------------------------------------
    cx = CPx + CPw / 2
    _arrow(ax, (cx, CPy + CPh), (cx, MTy - 0.02), color=C_POL, lw=1.9, ms=14)
    ax.text(cx + 0.18, (CPy + CPh + MTy) / 2, r"$m_s$", color=C_POL,
            fontsize=9.2, ha="left", va="center", fontweight="bold")

    _arrow(ax, (MTx + MTw, MTy + MTh / 2), (DTx, DTy + DTh * 0.80),
           color=C_THERMO, lw=2.3, ms=16, conn="arc3,rad=-0.10")
    ax.text(5.78, DTy + DTh * 0.80 + 0.30, S_sig, color=C_THERMO,
            fontsize=10.5, ha="center", va="center", fontweight="bold")

    _arrow(ax, (EWx + EWw, EWy + EWh / 2), (DTx, DTy + DTh * 0.22),
           color=C_FIELD, lw=2.3, ms=16, conn="arc3,rad=0.08")
    ax.text(5.78, DTy + DTh * 0.22 - 0.28, r"$\theta$", color=C_FIELD,
            fontsize=11, ha="center", va="center", fontweight="bold")

    _arrow(ax, (DTx + DTw, DTy + DTh * 0.52), (FGx, FGy + FGh / 2),
           color=C_SHAPE, lw=2.3, ms=16)
    ax.text((DTx + DTw + FGx) / 2, DTy + DTh * 0.52 + 0.34, r"$V_d,\ D_d$",
            color=C_SHAPE, fontsize=9.4, ha="center", va="center",
            fontweight="bold")

    _arrow(ax, (DPx + DPw / 2, DPy + DPh), (DTx + DTw / 2, DTy),
           color=C_DEP, lw=1.5, ms=13, ls="--")
    ax.text(DPx + DPw / 2 + 0.20, (DPy + DPh + DTy) / 2, r"$F_{DEP}$",
            color=C_DEP, fontsize=8.8, ha="left", va="center")

    _arrow(ax, (FGx + FGw / 2, FGy), (FGx + FGw / 2, FGy - 0.72),
           color=C_RATE, lw=2.0, ms=15)
    ax.text(FGx + FGw / 2, FGy - 1.12,
            "departure frequency $f$,\ngas-production rate",
            fontsize=9.0, color=C_RATE, fontweight="bold", ha="center",
            va="center", linespacing=1.25)

    # panel label + title ---------------------------------------------
    ax.text(0.35, 8.92, "(b)", fontsize=13, fontweight="bold", ha="left",
            va="center")
    ax.text(1.05, 8.92, "Coupled bubble-detachment model", fontsize=12.5,
            fontweight="bold", ha="left", va="center", color=INK)
    ax.text(1.05, 8.52,
            r"sub-models linked by the interfacial variables they set "
            r"($\sigma_{lv}$, $\theta$)",
            fontsize=9.0, ha="left", va="center", color="#555555",
            style="italic")

    return dict(model_top=(DTx + DTw / 2, DTy + DTh),
                rate_out=(FGx + FGw, FGy + FGh / 2))


# ======================================================================
#  bridge + closing loop
# ======================================================================
def draw_bridge(ax, a, b):
    # DOWN: the same H2 bubble, modelled below
    _arrow(ax, a["h2_arc"], b["model_top"], color=C_BRIDGE, lw=2.6, ms=18,
           conn="arc3,rad=0.05")
    ax.text(8.95, 9.72, "the same H2 bubble at the\ncathode, modelled in (b)",
            fontsize=8.4, ha="left", va="center", color=C_BRIDGE,
            fontweight="bold", linespacing=1.25)

    # UP: predicted rate feeds bubble growth (clean right-margin loop)
    x0, y0 = b["rate_out"]                    # Faradaic right edge
    xr, yt = 13.95, a["cathode_feed"][1]
    ax.plot([x0, xr, xr, 11.25], [y0, y0, yt, yt], color=C_RATE, lw=2.4,
            solid_joinstyle="round", solid_capstyle="round", zorder=2)
    _arrow(ax, (11.4, yt), (a["cathode_feed"][0], yt), color=C_RATE,
           lw=2.4, ms=18)
    ax.text(12.5, 13.05, "gas-production rate\nfeeds bubble growth",
            fontsize=8.4, ha="center", va="center", color=C_RATE,
            fontweight="bold", linespacing=1.25)


# ======================================================================
def main() -> None:
    apply_style()
    plt.rcParams.update({"axes.grid": False})

    fig, ax = plt.subplots(figsize=(9.0, 11.4))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 17.9)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(7.25, 17.55, "Single-bubble model of alkaline (KOH) "
            "water electrolysis", fontsize=13.5, fontweight="bold",
            ha="center", va="center", color=INK)
    ax.text(7.25, 17.12, "focus on H2 evolution and detachment at the cathode",
            fontsize=10, ha="center", va="center", color="#555555",
            style="italic")

    a = draw_reactor(ax)
    b = draw_model(ax)
    draw_bridge(ax, a, b)

    out_dir = os.path.join(_HERE, "tex", "figures", "proposals")
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, "fig1_variantA.pdf")
    png = os.path.join(out_dir, "fig1_variantA.png")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
