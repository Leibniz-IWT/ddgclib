"""
enrtl_parameters.py
===================

Literature binary interaction parameters for the (symmetric-reference)
electrolyte-NRTL activity-coefficient model, extracted verbatim from the
two papers in ``toy_bubble/lit/data/``:

    [Que2011]      H. Que, Y. Song, C.-C. Chen.
                   "Thermodynamic Modeling of the Sulfuric Acid-Water-Sulfur
                   Trioxide System with the Symmetric Electrolyte NRTL Model"
                   J. Chem. Eng. Data 56, 963-977 (2011).
                   DOI: 10.1021/je100930y
                   File: 1-s2.0-... no, the local file is
                   ``thermodynamic-modeling-of-the-sulfuric-acid-water-sulfur-trioxide-system-with-the-symmetric-electrolyte-nrtl-model.pdf``

    [Valverde2023] J.L. Valverde, V.R. Ferro, A. Giroir-Fendler.
                   "Prediction of the solid-liquid equilibrium of ternary and
                   quaternary salt-water systems. Influence of the e-NRTL
                   interaction parameters." Fluid Phase Equilibria 572, 113832 (2023).
                   DOI: 10.1016/j.fluid.2023.113832
                   (The refined binary parameters are originally from the
                    group's earlier paper, Valverde et al. FPE 551, 113264, 2022.)
                   File: ``1-s2.0-S0378381223001127-main.pdf``

This module is intentionally a *pure data* module: it does NOT compute
activities or import any of the ``toy_bubble`` simulation code.  It can
be imported later into any future e-NRTL implementation (lumped or
speciated) without creating a dependency cycle.

Conventions used throughout
---------------------------
* ``tau_{i,j}`` always has ``i`` = *source* species (the subscript
  written first in the NRTL exponential ``G_{ij} = exp(-alpha_{ij} tau_{ij})``),
  ``j`` = *target* species.  Both papers use this convention consistently.
* Default values for unregressed binaries (Aspen / [Que2011] convention):

      tau_{molecule, molecule}       =  0
      tau_{electrolyte, electrolyte} =  0
      tau_{molecule, electrolyte}    = +8
      tau_{electrolyte, molecule}    = -4

  i.e. the water-to-salt interaction is positive (~+8) and the
  salt-to-water interaction is negative (~-4).  The regressed values in
  [Valverde2023] sit close to these defaults.

* Non-randomness factors ``alpha_{ij}``:

      alpha_{molecule-molecule}           = 0.3
      alpha_{molecule-electrolyte}        = 0.2
      alpha_{electrolyte-electrolyte}     = 0.2

  (These are the [Que2011] conventions; [Valverde2023] fixes alpha = 0.2
  for all electrolyte-solvent and electrolyte-electrolyte pairs and 0.3
  for solvent-solvent pairs.)

* Pitzer-Debye-Hueckel "closest approach" parameter ``rho`` = 14.9
  (both papers; matches the current toy value ``RHO_PDH``).

* Temperature dependence (Que2011, their Eq. 8):

      tau_{ij}(T) = tau1_{ij} + tau2_{ij}/T
                  + tau3_{ij} * ((T_ref - T)/T + ln(T/T_ref))
      T_ref = 298.15 K.

  [Valverde2023] regressed only at 298.15 K and set tau3 = tau2 = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Dict, List


# ---------------------------------------------------------------------------
# Global constants (common to both papers)
# ---------------------------------------------------------------------------
T_REF: float = 298.15      # K, reference temperature for tau(T) correlation
RHO_PDH: float = 14.9      # PDH closest-approach parameter

# Default e-NRTL tau values for unregressed binaries (Que2011, p. 968)
TAU_DEFAULT_MOL_MOL: float = 0.0
TAU_DEFAULT_ELEC_ELEC: float = 0.0
TAU_DEFAULT_MOL_ELEC: float = +8.0   # tau_{molecule, electrolyte}
TAU_DEFAULT_ELEC_MOL: float = -4.0   # tau_{electrolyte, molecule}

# Default nonrandomness factors
ALPHA_MOL_MOL: float = 0.3
ALPHA_MOL_ELEC: float = 0.2
ALPHA_ELEC_ELEC: float = 0.2


# ---------------------------------------------------------------------------
# Generic parameter container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TauBinary:
    """One directed binary interaction entry ``tau_{i -> j}(T)``.

    Attributes
    ----------
    i, j      : species labels (source, target).  Electrolytes are written
                as ``"(cation,anion)"`` strings, e.g. ``"(H3O+,HSO4-)"``.
    tau1      : constant part of tau (dimensionless).
    tau2      : 1/T coefficient (K).
    tau3      : coefficient of ``(T_ref - T)/T + ln(T/T_ref)`` (dimensionless).
    alpha     : non-randomness factor for this pair (symmetric in i,j).
    source    : citation key ("Que2011" or "Valverde2023").
    """
    i: str
    j: str
    tau1: float
    tau2: float = 0.0
    tau3: float = 0.0
    alpha: float = 0.2
    source: str = ""

    def tau(self, T: float = T_REF) -> float:
        """Evaluate tau_{ij}(T)."""
        if self.tau2 == 0.0 and self.tau3 == 0.0:
            return self.tau1
        return (
            self.tau1
            + self.tau2 / T
            + self.tau3 * ((T_REF - T) / T + log(T / T_REF))
        )

    def G(self, T: float = T_REF) -> float:
        """Evaluate Boltzmann factor G_{ij} = exp(-alpha_{ij} tau_{ij}(T))."""
        from math import exp
        return exp(-self.alpha * self.tau(T))


# ===========================================================================
# [Que2011] -- H2SO4 / H2O / SO3 system, Table 9 (p. 975)
# ===========================================================================
#
# Speciation (Figure 1, p. 964):
#   Molecules  : H2O, H2SO4, SO3, H2S2O7 (disulfuric acid, nonvolatile)
#   Cations    : H3O+, H5O2+   (monohydrate and dihydrate of the proton)
#   Anions     : HSO4-, SO4-2
#
# Reactions (R1..R5):
#   R1: H2SO4 + H2O  <-> HSO4-  + H3O+
#   R2: HSO4- + H2O  <-> SO4-2  + H3O+
#   R3: H3O+  + H2O  <-> H5O2+
#   R4: SO3   + H2O  <-> H2SO4
#   R5: SO3   + H2SO4 <-> H2S2O7
#
# Table 9 columns: i | j | tau1_{ij} | tau2_{ij} | tau3_{ij} | alpha_{ij}
# All entries have source = "Que2011".

QUE2011_TAU: List[TauBinary] = [
    # --- molecule-molecule pairs (alpha = 0.3) ---
    TauBinary("SO3",   "H2SO4",  1.949, -533.53,   0.0, 0.3, "Que2011"),
    TauBinary("H2SO4", "SO3",    4.173, -1813.85,  0.0, 0.3, "Que2011"),

    # --- water <-> (H3O+, HSO4-) ---
    TauBinary("H2O",           "(H3O+,HSO4-)",  6.880,  1478.70, -7.954, 0.2, "Que2011"),
    TauBinary("(H3O+,HSO4-)",  "H2O",          -4.030, -521.03,   3.493, 0.2, "Que2011"),

    # --- water <-> (H5O2+, HSO4-) ---
    TauBinary("H2O",           "(H5O2+,HSO4-)",  6.339,   0.000, -0.118, 0.2, "Que2011"),
    TauBinary("(H5O2+,HSO4-)", "H2O",           -4.391, -13.292, -0.067, 0.2, "Que2011"),

    # --- water <-> (H3O+, SO4-2) ---
    TauBinary("H2O",           "(H3O+,SO4-2)", 12.238, -0.010,  0.229, 0.2, "Que2011"),
    TauBinary("(H3O+,SO4-2)",  "H2O",          -4.081, -0.932,  0.555, 0.2, "Que2011"),

    # --- water <-> (H5O2+, SO4-2) ---
    TauBinary("H2O",           "(H5O2+,SO4-2)",  3.494, -0.403, -0.310, 0.2, "Que2011"),
    TauBinary("(H5O2+,SO4-2)", "H2O",           -2.442,  3.203,  0.673, 0.2, "Que2011"),

    # --- sulfuric acid <-> (H3O+, HSO4-) ---
    TauBinary("H2SO4",        "(H3O+,HSO4-)",  3.978,  1100.60, -0.062, 0.2, "Que2011"),
    TauBinary("(H3O+,HSO4-)", "H2SO4",        -2.541,  -300.97,  0.513, 0.2, "Que2011"),

    # --- electrolyte-electrolyte pairs (alpha = 0.2, no T-dependence) ---
    TauBinary("(H5O2+,HSO4-)",  "(H3O+,HSO4-)",   5.392, 0.0, 0.0, 0.2, "Que2011"),
    TauBinary("(H5O2+,SO4-2)",  "(H5O2+,HSO4-)", -2.465, 0.0, 0.0, 0.2, "Que2011"),
    TauBinary("(H5O2+,HSO4-)",  "(H5O2+,HSO4-)", -0.115, 0.0, 0.0, 0.2, "Que2011"),
    TauBinary("(H3O+,HSO4-)",   "(H5O2+,HSO4-)",  0.130, 0.0, 0.0, 0.2, "Que2011"),
]


# ---------------------------------------------------------------------------
# [Que2011] Table 10 -- Reaction equilibrium constants ln K = A + B/T
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReactionEquilibrium:
    name: str
    reaction: str
    A: float
    B: float
    lnK_298_15: float     # reported value at 298.15 K, for cross-check


QUE2011_REACTIONS: Dict[str, ReactionEquilibrium] = {
    "R1": ReactionEquilibrium("R1", "H2SO4 + H2O <-> HSO4- + H3O+",
                              A=-3.898,  B=3475.0,   lnK_298_15=7.757),
    "R2": ReactionEquilibrium("R2", "HSO4- + H2O <-> SO4-2 + H3O+",
                              A=-5.393,  B=1733.1,   lnK_298_15=0.420),
    "R3": ReactionEquilibrium("R3", "H3O+  + H2O <-> H5O2+",
                              A=-1.741,  B=853.72,   lnK_298_15=1.122),
    "R4": ReactionEquilibrium("R4", "SO3   + H2O <-> H2SO4",
                              A=-12.29,  B=14245.7,  lnK_298_15=34.49),
    "R5": ReactionEquilibrium("R5", "SO3   + H2SO4 <-> H2S2O7",
                              A=-6.307,  B=3122.1,   lnK_298_15=4.165),
}


# ===========================================================================
# [Valverde2023] -- refined binary electrolyte-water parameters, Table 2 (p. 4)
# ===========================================================================
#
# All entries regressed at T = 298.15 K with alpha = 0.2 (fixed).
# The paper reports one "salt -> H2O" tau and one "H2O -> salt" tau per
# lumped salt ca.  Strong electrolytes assumed fully dissociated.
#
# Format of each value below:
#   (tau_{salt, H2O},   tau_{H2O, salt},   m_max (mol/kg))
# where the first value corresponds to the "electrolyte-molecule" position
# (paper default -4) and the second to the "molecule-electrolyte" position
# (paper default +8).

@dataclass(frozen=True)
class SaltWaterPair:
    salt: str                     # chemical formula of the fully-dissociated salt
    tau_salt_water: float         # tau_{ca, H2O}  -- first subscript = salt pair
    tau_water_salt: float         # tau_{H2O, ca}  -- first subscript = water
    m_max: float                  # upper molality bound in the fitting data set (mol/kg)
    alpha: float = 0.2
    source: str = "Valverde2023"

    def tau_sw(self) -> TauBinary:
        """Return the salt-to-water directed pair (T-independent)."""
        return TauBinary(self.salt, "H2O", self.tau_salt_water,
                         alpha=self.alpha, source=self.source)

    def tau_ws(self) -> TauBinary:
        """Return the water-to-salt directed pair (T-independent)."""
        return TauBinary("H2O", self.salt, self.tau_water_salt,
                         alpha=self.alpha, source=self.source)


VALVERDE2023_SALT_WATER: Dict[str, SaltWaterPair] = {
    "NaCl":   SaltWaterPair("NaCl",   -4.591,  9.002, 6.0),
    "KCl":    SaltWaterPair("KCl",    -4.117,  8.085, 4.8),
    "SrCl2":  SaltWaterPair("SrCl2",  -5.079, 10.023, 4.0),
    "KBr":    SaltWaterPair("KBr",    -4.136,  8.072, 5.5),
    "SrBr2":  SaltWaterPair("SrBr2",  -4.806,  9.082, 2.0),
    "MgBr2":  SaltWaterPair("MgBr2",  -5.348, 10.350, 2.5),
    "NaBr":   SaltWaterPair("NaBr",   -4.570,  8.811, 4.0),
    "LiCl":   SaltWaterPair("LiCl",   -5.242, 10.310, 6.0),
    "Na2SO4": SaltWaterPair("Na2SO4", -3.718,  7.581, 4.0),
    "K2SO4":  SaltWaterPair("K2SO4",  -4.310,  8.840, 0.7),
    "Li2SO4": SaltWaterPair("Li2SO4", -4.058,  7.914, 3.0),
}


# ===========================================================================
# Mapping to the current toy model's *lumped-salt* (tau_sw, tau_ws) pair
# ===========================================================================
#
# The toy at ``toy_bubble/bubble_enrtl_toy.py`` treats each salt as a single
# lumped species and carries exactly two NRTL parameters per salt:
#
#     tau_sw : "salt-pair -> solvent"   (first subscript = salt, second = water)
#     tau_ws : "solvent  -> salt-pair"  (first subscript = water, second = salt)
#
# Under the standard NRTL / Que2011 / Valverde2023 sign convention, this means:
#     toy.tau_sw  <->  tau_{salt, H2O}   (negative, ~ -4 to -5)
#     toy.tau_ws  <->  tau_{H2O, salt}   (positive, ~ +8 to +12)
#
# The current "representative" values in the toy are:
#     KOH  : tau_sw = +11.5, tau_ws = -4.5
#     H2SO4: tau_sw = +13.0, tau_ws = -5.2
#
# These have the signs flipped relative to the literature convention above,
# i.e. what the toy calls ``tau_sw`` actually holds a ``tau_{water, salt}``
# number and vice-versa.  Any future substitution of real literature values
# should either (a) swap the labels in the toy, or (b) pass the negated
# values in -- otherwise G^{ex} will have the wrong sign.  This is purely
# a labelling discrepancy; the physics in the toy is consistent internally.
#
# Recommended literature-backed substitutions (for future reference; NOT
# inserted into the toy code):

# -- H2SO4: use Que2011 dominant-species (H3O+, HSO4-) pair at 298.15 K --
#    tau at T_ref: tau = tau1 + tau2/T_ref   (tau3 term vanishes at T = T_ref)
#
#    tau_{H2O, (H3O+,HSO4-)}(298.15) = 6.880 + 1478.70/298.15 =  6.880 + 4.960 = 11.840
#    tau_{(H3O+,HSO4-), H2O}(298.15) = -4.030 - 521.03/298.15 = -4.030 - 1.747 = -5.777
#
# In the *toy's* (swapped) labelling this corresponds to:
#    tau_sw (toy label) ~= +11.84   (literature tau_{H2O, salt}, positive)
#    tau_ws (toy label) ~=  -5.78   (literature tau_{salt, H2O}, negative)
# -- which is indeed close to the toy's current H2SO4 entry (13.0, -5.2).
# The full speciated Que2011 model uses four electrolyte-pairs + five
# reactions, so a single lumped (tau_sw, tau_ws) will be an approximation.

# -- KOH: not in either paper. The closest K+ proxies from Valverde2023 are
#    KCl  (tau_{salt,H2O} = -4.117,  tau_{H2O,salt} = +8.085,  m_max = 4.8)
#    KBr  (tau_{salt,H2O} = -4.136,  tau_{H2O,salt} = +8.072,  m_max = 5.5)
#    K2SO4(tau_{salt,H2O} = -4.310,  tau_{H2O,salt} = +8.840,  m_max = 0.7)
#    All three cluster tightly near the defaults (-4, +8) -- this group's
#    refined fits for K+ salts are consistent with default values to within
#    ~0.1-0.3 units.  Using literature KCl values as a KOH proxy would give
#    (toy-label-swapped):
#        tau_sw (toy label) ~= +8.085
#        tau_ws (toy label) ~= -4.117


# Convenience lookup table for the H2SO4 "dominant species" lumped
# approximation derived above (so future code does not have to re-derive it).
#
# ``tau_{dominant electrolyte pair, water}`` and the reverse at 298.15 K.
H2SO4_DOMINANT_PAIR_AT_298_15: Dict[str, float] = {
    # dominant species at moderate H2SO4 concentration = (H3O+, HSO4-)
    "tau_salt_water":  -4.030 + (-521.03) / T_REF,   # -5.777
    "tau_water_salt":   6.880 + 1478.70  / T_REF,    # 11.840
    "alpha":            0.2,
    "electrolyte_pair_label": "(H3O+,HSO4-)",
}


# ---------------------------------------------------------------------------
# Small convenience accessors
# ---------------------------------------------------------------------------
def que2011_tau(i: str, j: str) -> TauBinary:
    """Return the [Que2011] TauBinary entry for the directed pair (i, j).

    Raises KeyError if not found.
    """
    for entry in QUE2011_TAU:
        if entry.i == i and entry.j == j:
            return entry
    raise KeyError(f"Que2011: no entry for tau_{{{i} -> {j}}}")


def valverde2023_pair(salt: str) -> SaltWaterPair:
    """Return the [Valverde2023] SaltWaterPair for a given salt formula."""
    try:
        return VALVERDE2023_SALT_WATER[salt]
    except KeyError as e:
        available = ", ".join(sorted(VALVERDE2023_SALT_WATER))
        raise KeyError(
            f"Valverde2023: salt {salt!r} not tabulated.  Available: {available}"
        ) from e


__all__ = [
    # constants
    "T_REF", "RHO_PDH",
    "TAU_DEFAULT_MOL_MOL", "TAU_DEFAULT_ELEC_ELEC",
    "TAU_DEFAULT_MOL_ELEC", "TAU_DEFAULT_ELEC_MOL",
    "ALPHA_MOL_MOL", "ALPHA_MOL_ELEC", "ALPHA_ELEC_ELEC",
    # data classes
    "TauBinary", "ReactionEquilibrium", "SaltWaterPair",
    # tables
    "QUE2011_TAU", "QUE2011_REACTIONS", "VALVERDE2023_SALT_WATER",
    "H2SO4_DOMINANT_PAIR_AT_298_15",
    # accessors
    "que2011_tau", "valverde2023_pair",
]
