"""
bubble_enrtl_toy_real.py
========================

Thin driver that runs exactly the same four-layer pipeline as
``bubble_enrtl_toy.py`` but with LITERATURE-VALIDATED e-NRTL parameters
from two published sources:

    H2SO4 - H2O:  Que, Song & Chen.  J. Chem. Eng. Data 56, 963 (2011).
                  Table 9, dominant-species lump (H3O+, HSO4-)  at 298.15 K.
    KCl   - H2O:  Valverde, Ferro & Giroir-Fendler.
                  Fluid Phase Equil. 572, 113832 (2023).  Table 2.
                  KCl is used as a K+-salt proxy for KOH: neither paper
                  tabulates KOH directly; Valverde explicitly notes that
                  K+ salts (KCl, KBr, K2SO4) all cluster near the (-4, +8)
                  defaults, so KCl is a defensible proxy for the cation
                  contribution.  H2O -> OH- interaction is left at the
                  paper's Cl- refined value as a placeholder.

Output is written to ``toy_bubble/fig_real/`` and ``toy_bubble/out_real/``
so the "representative-parameters" run in ``fig/`` + ``out/`` is preserved
side-by-side for the slide deck.

Run:
    python toy_bubble/bubble_enrtl_toy_real.py
"""

from __future__ import annotations
import os
import sys

import numpy as np

# --- make both this folder and ./lit/ importable -----------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.join(_HERE, 'lit')):
    if p not in sys.path:
        sys.path.insert(0, p)

import bubble_enrtl_toy as toy
import enrtl_parameters as lit


# ---------------------------------------------------------------------------
# Literature-backed Salt instances (LITERATURE sign convention:
# tau_sw = tau_{salt, H2O} is NEGATIVE, tau_ws = tau_{H2O, salt} is POSITIVE)
# ---------------------------------------------------------------------------
#
# The toy's G^ex_NRTL expression as implemented in `bubble_enrtl_toy.py` is
#
#     G^ex_NRTL/(n_t RT) = x_w X_s [ tau_sw G_sw/(x_w + X_s G_sw)
#                                  + tau_ws G_ws/(X_s + x_w G_ws) ]
#
# Reading the NRTL convention "first-subscript = neighbour, second = central",
# the term with water central has coefficient tau_{salt -> water} (salt as
# neighbour), which is what the toy calls ``tau_sw``; the term with salt
# central has coefficient tau_{water -> salt} = ``tau_ws``.
#
# Literature values (both papers follow standard NRTL convention):
#
#     tau_{salt, H2O}  ~ -4 to -6   (salt as perturber, water as centre)
#     tau_{H2O, salt}  ~ +8 to +12
#
# so ``tau_sw`` should be NEGATIVE and ``tau_ws`` POSITIVE.  The original
# "representative" toy had these swapped for historical reasons; the
# conclusions were unchanged (Butler still finds a root; G^ex still stable),
# but curves were not quantitatively calibrated.  Here we use the correct
# signs.
# ---------------------------------------------------------------------------

# --- H2SO4: Que2011 dominant-species lump (H3O+, HSO4-) at 298.15 K ---------
_h2so4 = lit.H2SO4_DOMINANT_PAIR_AT_298_15
H2SO4_real = toy.Salt(
    name='H2SO4',
    z_c=+1, z_a=-2, nu_c=2, nu_a=1,
    tau_sw=_h2so4['tau_salt_water'],      # -5.777  (salt -> water)
    tau_ws=_h2so4['tau_water_salt'],      # +11.840 (water -> salt)
    sigma0=toy.SIGMA0_H2SO4_PURE,
    A_s=toy.A_H2SO4,
    V_app=53e-6,
    k_sech=0.099,
)

# --- KCl:  Valverde2023 refined binary at 298.15 K, K+-salt proxy for KOH ---
_kcl = lit.VALVERDE2023_SALT_WATER['KCl']
KCl_real = toy.Salt(
    name='KCl',
    z_c=+1, z_a=-1, nu_c=1, nu_a=1,
    tau_sw=_kcl.tau_salt_water,           # -4.117
    tau_ws=_kcl.tau_water_salt,           # +8.085
    sigma0=toy.SIGMA0_KOH_PURE,           # keep KOH-ish hypothetical pure-salt sigma
    A_s=toy.A_KOH,
    V_app=27e-6,
    k_sech=0.134,                         # Sechenov for KOH (no Cl-specific; toy)
)


# ---------------------------------------------------------------------------
# Molality domains -- respect each paper's fitting range
# ---------------------------------------------------------------------------
# Valverde2023 reports m_max = 4.8 mol/kg for KCl.
# Que2011 covers up to ~15 m H2SO4 but dominant speciation shifts at higher m
# and the lumped (H3O+, HSO4-) approximation is safest below ~6 m.
M_GRIDS = {
    'KCl':   np.linspace(0.01, _kcl.m_max, 40),    # 0.01 -> 4.8 mol/kg
    'H2SO4': np.linspace(0.01, 6.0, 40),           # 0.01 -> 6.0 mol/kg
}

COLORS = {'KCl': 'C0', 'H2SO4': 'C3'}

CITATION_BANNER = ('literature e-NRTL:  Que et al. 2011 (H$_2$SO$_4$)  +  '
                   'Valverde et al. 2023 (KCl, proxy for KOH)')


def main() -> None:
    """Run the four-layer pipeline with literature e-NRTL parameters."""
    print('Running with literature e-NRTL parameters:')
    for s in (H2SO4_real, KCl_real):
        print(f'  {s.name:6s}: tau_sw = {s.tau_sw:+7.3f}   '
              f'tau_ws = {s.tau_ws:+7.3f}   alpha = 0.2')
    print()

    toy.demo(
        salts=[KCl_real, H2SO4_real],
        m_grids=M_GRIDS,
        colors=COLORS,
        fig_dir_name='fig_real',
        out_dir_name='out_real',
        citation_banner=CITATION_BANNER,
    )


if __name__ == '__main__':
    main()
