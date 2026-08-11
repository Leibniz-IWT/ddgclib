"""
make_water_figure_sets.py
=========================

Regenerate the full water-electrolysis figure family (fig1..7 from
``bubble_enrtl_electrostatic_toy.demo``) as two side-by-side sets that differ
ONLY in the e-NRTL interaction parameters tau:

  fig/water_representative/  -- hand-tuned "representative" tau (KOH 11.5/-4.5,
                                H2SO4 13.0/-5.2), swapped convention. Trend-
                                correct but NOT fitted; gives unphysical phi<0.
  fig/water_real/            -- DATA-FITTED tau in the standard convention,
                                from enrtl_fit/ (Hamer & Wu 1972 for KOH to
                                m<=10; Que 2011 Fig.10 digitised for H2SO4 to
                                m<=6). See manuscript/CORRECTIONS.md sec 3b and
                                enrtl_fit/DOCUMENTATION.md.

Everything else (sigma0, molar surface area A_s, apparent molar volume V_app,
Sechenov coefficient k_sech, molality grids, contact angle, criterion) is held
identical, so any difference between the two folders is attributable to the
thermodynamic parameters alone.  Note fig3 (salting-out) depends only on
k_sech, so it is identical between the two sets by construction.

Run:  python make_water_figure_sets.py                # both sets
      python make_water_figure_sets.py representative # representative only
      python make_water_figure_sets.py real           # data-fitted only
"""
from __future__ import annotations
import os
import sys
from dataclasses import replace

import bubble_enrtl_electrostatic_toy as toy
from detachment_fitted import fit_salt

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "enrtl_fit", "data")
_CSV = {"KOH": os.path.join(_DATA, "koh_hamer_wu.csv"),
        "H2SO4": os.path.join(_DATA, "h2so4_que2011_fig10.csv")}


def representative_set() -> None:
    """fig/water_representative/ -- the hand-tuned toy.KOH / toy.H2SO4."""
    banner = ("REPRESENTATIVE (hand-tuned) e-NRTL tau -- trend-correct, NOT "
              "data-fitted  [KOH tau_sw=11.5/tau_ws=-4.5;  H2SO4 13.0/-5.2]")
    toy.demo(fig_dir_name="fig/water_representative",
             out_dir_name="out/water_representative", citation_banner=banner)


def fitted_salts():
    """Fitted KOH / H2SO4 toy.Salt objects (canonical names), + fit metadata."""
    salts, info = [], {}
    for name in ("KOH", "H2SO4"):
        salt, p_fit, m_max = fit_salt(name, _CSV[name])
        salts.append(replace(salt, name=name))     # canonical name for demo()
        info[name] = (p_fit, m_max)
    return salts, info


def real_set() -> None:
    """fig/water_real/ -- the data-fitted tau from enrtl_fit/."""
    salts, info = fitted_salts()
    k, h = info["KOH"], info["H2SO4"]
    banner = ("DATA-FITTED e-NRTL tau  "
              f"[KOH tau_sw={k[0].tau_cw:+.2f}/tau_ws={k[0].tau_wc:+.2f} "
              f"(Hamer&Wu 1972, m<={k[1]:.0f});  "
              f"H2SO4 tau_sw={h[0].tau_cw:+.2f}/tau_ws={h[0].tau_wc:+.2f} "
              f"(Que2011 Fig10, m<={h[1]:.0f})]")
    toy.demo(salts=salts, fig_dir_name="fig/water_real",
             out_dir_name="out/water_real", citation_banner=banner)
    print("Data-fitted tau used (standard convention):")
    for name, (p, mmax) in info.items():
        print(f"  {name:6s}: tau_sw={p.tau_cw:+.3f}  tau_ws={p.tau_wc:+.3f}  (fit m<={mmax:.0f})")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "representative"):
        print("== representative set -> fig/water_representative/ ==")
        representative_set()
    if which in ("both", "real"):
        print("== data-fitted set -> fig/water_real/ ==")
        real_set()


if __name__ == "__main__":
    main()
