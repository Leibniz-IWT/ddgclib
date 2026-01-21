#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vcut(z1, z2, coeffs_2x)
- coeffs_2x can be a 10-tuple OR a string like:
  "COEFFS 2x form (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0)"

Axis-aligned quadrics with the 2× convention:
  Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz + J = 0
Assumes D=E=F=G=H=0 (no tilt, no x/y shift). A>0, B>0.

Cross-section at z: Ax^2 + By^2 <= S(z), S(z)=-(C z^2 + 2 I z + J)
Area(z)=π*S(z)/sqrt(A B) if S(z)>0.
V = ∫ Area(z) dz over z in [z1,z2] where S(z)>0 (auto-clipped).
"""

import re
from math import sqrt, pi
import numpy as np

# ----------------------- parsing helpers -----------------------

def parse_coeffs_2x(obj):
    """
    Accepts either:
      - tuple/list of length 10 -> returns tuple
      - string like 'COEFFS 2x form (a,b,c,d,e,f,g,h,i,j)' -> returns tuple
    """
    if isinstance(obj, (tuple, list)) and len(obj) == 10:
        return tuple(float(x) for x in obj)

    if isinstance(obj, str):
        # Extract the 10 comma-separated numbers inside the parentheses
        m = re.search(r"\(([^)]*)\)", obj)
        if not m:
            raise ValueError("Could not find '(...)' with 10 numbers in the string.")
        inside = m.group(1)
        # Split by commas, allow spaces
        parts = [p.strip() for p in inside.split(",")]
        if len(parts) != 10:
            raise ValueError(f"Expected 10 numbers, got {len(parts)}: {parts}")
        try:
            vals = tuple(float(p) for p in parts)
        except ValueError as e:
            raise ValueError(f"Failed to parse numbers from: {parts}") from e
        return vals

    raise TypeError("coeffs_2x must be a 10-tuple/list or a string like 'COEFFS 2x form (...)'.")

# ----------------------- core volume -----------------------

def Vcut(z1: float, z2: float, coeffs_2x, clip: bool = True) -> float:
    """
    Volume between planes z=z1 and z=z2 of the interior of an axis-aligned quadric.
    coeffs_2x uses the 2× convention. Accepts tuple or the 'COEFFS 2x form (...)' string.

    Assumptions: D=E=F=G=H=0, A>0, B>0.
    Auto-clips to S(z)>0 if clip=True.
    """
    (A,B,C,D,E,F,G,H,I,J) = parse_coeffs_2x(coeffs_2x)

    if A == 0 and B == 0: return 0.0  # degenerate case
    
    if A <= 0 and B <= 0:
        # flip ALL coeffs
        A, B, C, D, E, F, G, H, I, J = (
            -A, -B, -C, -D, -E, -F, -G, -H, -I, -J
        )
    if abs(D)+abs(E)+abs(F)+abs(G)+abs(H) > 1e-15:
        raise ValueError("This implementation requires D=E=F=G=H=0 (axis-aligned, no x/y shift/tilt).")
    if A <= 0 or B <= 0:
        #print(A, B, C, D, E, F, G, H, I, J)
        raise ValueError(A, B, C, D, E, F, G, H, I, J,"Requires A>0 and B>0 for elliptical x-y sections.")

    a_z, b_z = (z1, z2) if z1 <= z2 else (z2, z1)

    def S(z):       # positivity condition for a real cross-section
        return -(C*z*z + 2*I*z + J)

    def anti_S(z):  # ∫ S(z) dz
        return -(C*z**3/3.0 + I*z*z + J*z)

    factor = pi / sqrt(A*B)

    if not clip:
        return factor * (anti_S(b_z) - anti_S(a_z))

    # Find intervals within [a_z,b_z] where S(z) > 0
    intervals = []
    if abs(C) < 1e-15:
        # S(z) = -(2 I z + J)
        if abs(I) < 1e-15:
            # S = -J (constant)
            if -J > 0:
                intervals.append((a_z, b_z))
        else:
            z_root = -J/(2.0*I)
            if I > 0:  # S>0 for z < z_root
                L = (a_z, min(b_z, z_root))
                if L[1] > L[0]:
                    intervals.append(L)
            else:      # I < 0 => S>0 for z > z_root
                R = (max(a_z, z_root), b_z)
                if R[1] > R[0]:
                    intervals.append(R)
    else:
        # Quadratic: C z^2 + 2 I z + J = 0
        disc = 4.0*I*I - 4.0*C*J
        if disc < 0:
            # No real roots -> S keeps sign; test midpoint
            mid = 0.5*(a_z+b_z)
            if S(mid) > 0:
                intervals.append((a_z, b_z))
        else:
            rdisc = np.sqrt(disc)
            z1r = (-2.0*I - rdisc) / (2.0*C)
            z2r = (-2.0*I + rdisc) / (2.0*C)
            z_lo, z_hi = (z1r, z2r) if z1r <= z2r else (z2r, z1r)

            # S(z) = -(quadratic). For C>0, quadratic opens up ⇒ S>0 between roots.
            # For C<0, S>0 outside the roots.
            if C > 0:
                cand = (max(a_z, z_lo), min(b_z, z_hi))
                if cand[1] > cand[0]:
                    intervals.append(cand)
            else:
                L = (a_z, min(b_z, z_lo))
                R = (max(a_z, z_hi), b_z)
                if L[1] > L[0]:
                    intervals.append(L)
                if R[1] > R[0]:
                    intervals.append(R)

    V = 0.0
    for L, R in intervals:
        V += anti_S(R) - anti_S(L)
    return factor * V

# ----------------------- quick validations -----------------------

def rel_err(val, ref):
    denom = max(1e-30, abs(ref))
    return abs(val - ref) / denom

if __name__ == "__main__":
    # Your exact input style:
    s = "COEFFS 2x form (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0)"
    z1, z2 = 0.0, 0.2075975
    V = Vcut(z1, z2, s, clip=True)
    V_ref = (pi/2.0) * (z2**2 - z1**2)  # paraboloid z = x^2 + y^2
    print(f"Paraboloid test: Vcut[{z1},{z2}] = {V:.15f} | ref = {V_ref:.15f} | rel.err = {rel_err(V, V_ref):.3e}")

    # Sphere R=1: "COEFFS 2x form (1,1,1,0,0,0,0,0,0,-1)"
    s_sphere = "COEFFS 2x form (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)"
    z1, z2 = -0.3, 0.7
    V = Vcut(z1, z2, s_sphere, clip=True)
    V_ref = pi*(z2 - z2**3/3.0) - pi*(z1 - z1**3/3.0)  # R=1
    print(f"Sphere test:     Vcut[{z1},{z2}] = {V:.15f} | ref = {V_ref:.15f} | rel.err = {rel_err(V, V_ref):.3e}")

    # Hyperboloid of two sheets: z^2/c^2 - x^2/a^2 - y^2/b^2 = 1
    a, b, c = 0.6, 0.6, 0.8
    s_h2 = f"COEFFS 2x form ({1/a**2}, {1/b**2}, {-1/c**2}, 0, 0, 0, 0, 0, 0, 1)"
    z1, z2 = 0.9, 1.2
    V = Vcut(z1, z2, s_h2, clip=True)
    V_ref = np.pi*a*b*((z2**3 - z1**3)/(3.0*c*c) - (z2 - z1))
    print(f"Hyp2 test:       Vcut[{z1},{z2}] = {V:.15f} | ref = {V_ref:.15f} | rel.err = {rel_err(V, V_ref):.3e}")
