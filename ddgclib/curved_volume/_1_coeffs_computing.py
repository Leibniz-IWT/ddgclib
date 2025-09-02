#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rim-safe quadric fitting (NO physical IDs) with smart normalization and CSV columns:
  ABC_A..ABC_J, A_A..A_J, B_A..B_J, C_A..C_J

1× convention:
    f(x,y,z) = A x^2 + B y^2 + C z^2
             + D xy + E xz + F yz
             + G x + H y + I z + J = 0

Smart normalization (scale-aware):
    1) If |J| is not tiny, scale to J = -1.
    2) Else if |H| is not tiny, scale to H = +1.   (ideal for y - x^2 surfaces)
    3) Else leave coefficients unscaled.

Paraboloid special-case (this file adds it):
    - If it looks like a z-axis paraboloid (A≈B, C≈D≈E≈F≈G≈H≈0, I≠0),
      DO NOT normalize J. Instead rescale so A = +1 exactly (B snaps to 1
      if very close). This preserves J and makes I carry the physical scale.

Other features:
- Centroid/PCA/RMS normalization -> whitened SVD (optional IRLS).
- Affine back-transform; snap tiny A..I; optional A/B canonization for hyperbolas.
- Flat-plane guard on ABC patch (skips planar faces).
- Output CSV: <mesh_basename>_COEFFS.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import meshio
from collections import defaultdict

# ---------- Embedded default mesh path ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# You can switch the default by changing the filename below.
EMBEDDED_MESH_PATH = os.path.join(SCRIPT_DIR, "coarse_paraboloid.msh")
EMBEDDED_MESH_PATH = os.path.join(SCRIPT_DIR, "coarse_hyperboloid.msh")
EMBEDDED_MESH_PATH = os.path.join(SCRIPT_DIR, "Ellip_0_sub0_full.msh")
EMBEDDED_MESH_PATH = os.path.join(SCRIPT_DIR, "CylinderSymm_0_tet.msh")
# Examples you may have locally:
# EMBEDDED_MESH_PATH = os.path.join(SCRIPT_DIR, "parabolic_cylinder_y_eq_x2_y1slice.msh")
# EMBEDDED_MESH_PATH = os.path.join(SCRIPT_DIR, "hyperbola_cylinder_x2_minus_y2_z1slice.msh")
# EMBEDDED_MESH_PATH = os.path.join(SCRIPT_DIR, "CylinderSymm_0_tet.msh")

# =====================================================================
#                           QUADRIC HELPERS
# =====================================================================

def rescale_smart(coeff, tol_abs=1e-9, tol_rel=1e-6, prefer="H"):
    """
    Smart normalization for 1× convention coefficients (A..J).
    Strategy:
      - If |J| > thr: scale so J = -1.
      - Else if prefer=='H' and |H| > thr: scale so H = +1.
      - Else: return as-is.
    thr is max(tol_abs, tol_rel * max(|A..I|), 1e-3).
    """
    A,B,C,D,E,F,G,H,I,J = map(float, coeff)
    scale_terms = max(1.0, np.max(np.abs(coeff[:9])))
    thr = max(tol_abs, tol_rel * scale_terms, 1e-3)

    if abs(J) > thr:
        return coeff * (-1.0 / J)

    if prefer.upper() == "H" and abs(H) > thr:
        s = 1.0 / H
        return coeff * s

    return coeff

def coeffs_to_Q(c: np.ndarray) -> np.ndarray:
    """
    (A..J) -> 4x4 symmetric quadric matrix Q for f(x)=0 with the 1× convention.
    Encoding:
      Q = [[A,   D/2, E/2, G/2],
           [D/2, B,   F/2, H/2],
           [E/2, F/2, C,   I/2],
           [G/2, H/2, I/2, J  ]]
    """
    A,B,C,D,E,F,G,H,I,J = [float(v) for v in c]
    return np.array([[A,   D/2, E/2, G/2],
                     [D/2, B,   F/2, H/2],
                     [E/2, F/2, C,   I/2],
                     [G/2, H/2, I/2, J  ]], dtype=float)

def Q_to_coeffs(Q: np.ndarray) -> np.ndarray:
    """
    4x4 symmetric Q -> (A..J) with the 1× convention.
      A=Q00, B=Q11, C=Q22,
      D=2Q01, E=2Q02, F=2Q12,
      G=2Q03, H=2Q13, I=2Q23,
      J=Q33.
    """
    return np.array([
        Q[0,0], Q[1,1], Q[2,2],
        2*Q[0,1], 2*Q[0,2], 2*Q[1,2],
        2*Q[0,3], 2*Q[1,3], 2*Q[2,3],
        Q[3,3]
    ], dtype=float)

def _build_norm_transform(P: np.ndarray):
    """
    Build affine T such that original x = T * y (homog), with y normalized:
      - translate by centroid
      - rotate by PCA (right-handed)
      - isotropic RMS scaling
    Returns: Y (normalized Nx3), T (4x4).
    """
    P = np.asarray(P, float)
    t = P.mean(axis=0) if len(P) else np.zeros(3)
    Q = P - t
    if len(P) >= 3:
        _, _, Vt = np.linalg.svd(Q, full_matrices=False)
        R = Vt.T
        if np.linalg.det(R) < 0:
            R[:, -1] *= -1.0
    else:
        R = np.eye(3)
    s = np.sqrt(np.mean(np.sum(Q**2, axis=1))) if len(P) else 1.0
    if not np.isfinite(s) or s < 1e-12:
        s = 1.0
    Y = (Q @ R) / s
    T = np.eye(4)
    T[:3,:3] = s * R
    T[:3, 3] = t
    return Y, T

def _transform_coeffs_from_norm(c_norm: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Given coeffs in normalized coords (y-space), return coeffs in original x-space (1× convention)."""
    Qy = coeffs_to_Q(c_norm)
    Ti = np.linalg.inv(T)
    Qx = Ti.T @ Qy @ Ti
    return Q_to_coeffs(Qx)

def _design(P: np.ndarray) -> np.ndarray:
    """
    Monomial design matrix for general quadrics (1× convention):
      [x^2, y^2, z^2, xy, xz, yz, x, y, z, 1]
    """
    P = np.asarray(P, float)
    x, y, z = P[:,0], P[:,1], P[:,2]
    return np.column_stack([
        x*x, y*y, z*z,
        x*y, x*z, y*z,
        x, y, z,
        np.ones_like(x)
    ])

def _fit_whitened(X: np.ndarray) -> np.ndarray:
    """Column-whitened homogeneous LS (SVD nullspace)."""
    col = np.linalg.norm(X, axis=0)
    col[col == 0] = 1.0
    Xw = X / col
    _, _, Vt = np.linalg.svd(Xw, full_matrices=False)
    c = Vt[-1] / col
    n = np.linalg.norm(c)
    return c/n if n > 0 and np.isfinite(n) else np.full(10, np.nan)

def _grad_norm(P: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Pointwise ||∇f|| for IRLS weighting (1× convention):
      ∂f/∂x = 2A x + D y + E z + G
      ∂f/∂y = 2B y + D x + F z + H
      ∂f/∂z = 2C z + E x + F y + I
    """
    A,B,C,D,E,F,G,H,I,J = c
    x,y,z = P[:,0], P[:,1], P[:,2]
    gx = 2*A*x + D*y + E*z + G
    gy = 2*B*y + D*x + F*z + H
    gz = 2*C*z + E*x + F*y + I
    return np.sqrt(gx*gx + gy*gy + gz*gz)

def _fit_whitened_irls(P: np.ndarray, iters: int = 0, eps: float = 1e-6) -> np.ndarray:
    """Whitened fit + optional IRLS with gradient-based weights (1× convention)."""
    X = _design(P)
    c = _fit_whitened(X)
    if not np.all(np.isfinite(c)):
        return c
    for _ in range(max(0, iters)):
        g = _grad_norm(P, c)
        w = 1.0 / np.maximum(g, eps)
        med = np.median(w)
        if not np.isfinite(med) or med <= 0:
            med = 1.0
        w = w / med
        Xw = X * w[:, None]
        c = _fit_whitened(Xw)
        if not np.all(np.isfinite(c)):
            break
    return c

def _snap_small_A_to_I(c: np.ndarray, snap_rel: float = 5e-6, snap_abs: float = 1e-9) -> np.ndarray:
    """Snap tiny A..I to zero (never touches J)."""
    c = np.array(c, float)
    scale = max(abs(c[0]), abs(c[1]), abs(c[2]), 1.0)
    thr = max(snap_abs, snap_rel * scale)
    c[:9] = np.where(np.abs(c[:9]) <= thr, 0.0, c[:9])
    return c

def snap_AB_to_pm1(c: np.ndarray, rel_tol: float = 1e-5, abs_tol: float = 1e-8) -> np.ndarray:
    """
    If the fit looks like x^2 - y^2 + J ≈ 0 (C..I ~ 0, A ≈ -B, (A-B)/2 ≈ 1),
    force A=+1, B=-1 exactly. J is left untouched.
    """
    c = np.array(c, float)
    A,B,C,D,E,F,G,H,I,J = c
    others = np.array([C,D,E,F,G,H,I], float)
    scale = max(1.0, np.max(np.abs(c[:9])))
    if np.any(np.abs(others) > np.maximum(abs_tol, rel_tol * scale)):
        return c
    if abs(A + B) > max(abs_tol, rel_tol * max(abs(A), abs(B), 1.0)):
        return c
    s = 0.5 * (A - B)  # target ~ 1
    if abs(s - 1.0) <= max(abs_tol, rel_tol):
        c[0] =  1.0
        c[1] = -1.0
    return c

# ---------- NEW: Paraboloid detection & canonization (A=+1, keep J) ----------

def _is_vertical_paraboloid_like(c: np.ndarray,
                                 rel_tol: float = 2e-3,
                                 abs_tol: float = 1e-7) -> bool:
    """
    z-axis paraboloid pattern (1× convention):
      x^2 and y^2 present and similar; no z^2; no xy,xz,yz; no linear x,y;
      linear z present; J free.
      Typical form:  A x^2 + B y^2 + I z + J = 0  with A≈B, I≠0
    """
    c = np.asarray(c, float)
    A,B,C,D,E,F,G,H,I,J = c
    scale = max(1.0, max(abs(A), abs(B), abs(I)))
    thr   = max(abs_tol, rel_tol * scale)
    if abs(C) > thr: return False
    if any(abs(v) > thr for v in (D,E,F,G,H)): return False
    if abs(A) <= thr or abs(B) <= thr or abs(I) <= thr: return False
    if abs(A - B) > max(abs_tol, rel_tol * max(abs(A), abs(B))):
        return False
    return True

def _canonize_paraboloid_A1(c: np.ndarray,
                            rel_tol: float = 2e-3,
                            abs_tol: float = 1e-7) -> np.ndarray:
    """
    For z-axis paraboloid, rescale so A == +1 exactly (do NOT force J).
    This also makes B ≈ 1 and leaves J untouched; I carries physical scale.
    """
    c = np.array(c, float)
    if not _is_vertical_paraboloid_like(c, rel_tol=rel_tol, abs_tol=abs_tol):
        return c
    A = c[0]
    if A == 0.0:
        return c
    s = 1.0 / abs(A)           # make |A| == 1
    s *= np.sign(A)            # and A positive -> +1
    c *= s
    # Snap B to exactly 1 if extremely close
    if abs(c[1] - 1.0) <= max(abs_tol, rel_tol):
        c[1] = 1.0
    return c

def fit_quadric_stable(P: np.ndarray,
                       irls_iters: int = 0,
                       snap_rel: float = 5e-6,
                       snap_abs: float = 1e-9,
                       canonize_xy: bool = False,
                       debug_j: bool = False,
                       label: str = "") -> np.ndarray:
    """
    Robust, model-agnostic quadric fit (1× convention):
      1) Normalize coords (centroid/PCA/RMS)
      2) Whitened SVD + optional IRLS
      3) Back-transform to original coords
      4) Paraboloid-first rule:
         - If it looks like vertical paraboloid, set A=+1 (do NOT touch J).
         - Else apply smart rescale (J=-1 if safe; else H=1; else none).
      5) Snap tiny A..I; optional A/B canonization to +1/-1 (doesn't touch J)
    """
    P = np.asarray(P, float)
    if P.shape[0] < 10:
        return np.full(10, np.nan)
    Y, T = _build_norm_transform(P)
    c_norm = _fit_whitened_irls(Y, iters=irls_iters)
    if not np.all(np.isfinite(c_norm)):
        return np.full(10, np.nan)
    c = _transform_coeffs_from_norm(c_norm, T)

    # ----- DEBUG: print J BEFORE any normalization -----
    if debug_j and np.all(np.isfinite(c)):
        scale_terms = np.max(np.abs(c[:9])) if np.all(np.isfinite(c[:9])) else np.nan
        print(f"[debugJ] {label} preJ={c[9]:.12g}  A={c[0]:.6g}  H={c[7]:.6g}  scaleRef={scale_terms:.6g}")

    # ---- Paraboloid-first rule (skip J-normalization for paraboloids) ----
    if _is_vertical_paraboloid_like(c):
        c = _canonize_paraboloid_A1(c)
    else:
        c = rescale_smart(c)  # smart normalization for all others

    c = _snap_small_A_to_I(c, snap_rel=snap_rel, snap_abs=snap_abs)
    if canonize_xy:
        c = snap_AB_to_pm1(c)          # optional hyperbola canonization
    return c

# =====================================================================
#                    SURFACE TRIANGLES & COMPONENTS
# =====================================================================

def get_all_surface_triangles(mesh: meshio.Mesh) -> np.ndarray:
    tris = []
    for cb in mesh.cells:
        if cb.type == "triangle":
            tris.append(cb.data.astype(int))
        elif cb.type == "triangle6":
            tris.append(cb.data[:, :3].astype(int))
    if tris:
        all_tris = np.vstack(tris)
        print(f"[triangles] explicit surface faces: {all_tris.shape[0]} (from {len(tris)} block(s))")
        return all_tris

    # Fallback: derive boundary faces from tets
    face_count = defaultdict(int)
    keep_oriented = {}
    for cb in mesh.cells:
        if cb.type == "tetra":
            for (a,b,c,d) in cb.data:
                for f in ((a,b,c),(a,b,d),(a,c,d),(b,c,d)):
                    key = tuple(sorted(f))
                    face_count[key] += 1
                    if key not in keep_oriented:
                        keep_oriented[key] = f
    boundary = [keep_oriented[k] for k,cnt in face_count.items() if cnt == 1]
    if not boundary:
        raise RuntimeError("No surface triangles found.")
    all_tris = np.asarray(boundary, int)
    print(f"[triangles] derived from tets: {all_tris.shape[0]} faces")
    return all_tris

def triangle_adjacency(tris: np.ndarray):
    edge_map = defaultdict(list)  # (i<j) -> [tri ids]
    for t,(a,b,c) in enumerate(tris):
        for u,v in ((a,b),(b,c),(c,a)):
            key = (u,v) if u < v else (v,u)
            edge_map[key].append(t)
    nbrs = [[] for _ in range(len(tris))]
    for ids in edge_map.values():
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                ti,tj = ids[i], ids[j]
                nbrs[ti].append(tj)
                nbrs[tj].append(ti)
    return [sorted(set(v)) for v in nbrs]

def triangle_normals(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    A = points[tris[:,0]]; B = points[tris[:,1]]; C = points[tris[:,2]]
    N = np.cross(B - A, C - A)
    nrm = np.linalg.norm(N, axis=1, keepdims=True)
    return np.divide(N, np.maximum(nrm, 1e-20))

def smooth_components(points: np.ndarray, tris: np.ndarray, theta_max_deg: float = 35.0) -> np.ndarray:
    N = triangle_normals(points, tris)
    nbrs = triangle_adjacency(tris)
    cos_thr = np.cos(np.radians(theta_max_deg))
    T = len(tris)
    comp_id = -np.ones(T, dtype=int)
    cur = 0
    for t in range(T):
        if comp_id[t] != -1:
            continue
        stack = [t]; comp_id[t] = cur
        while stack:
            i = stack.pop()
            ni = N[i]
            for j in nbrs[i]:
                if comp_id[j] != -1:
                    continue
                nj = N[j]
                if abs(float(np.dot(ni, nj))) >= cos_thr:
                    comp_id[j] = cur
                    stack.append(j)
        cur += 1
    print(f"[smooth] components: {cur} (theta_max={theta_max_deg}°)")
    return comp_id

def build_vertex_adjacency_by_component(nV: int, tris: np.ndarray, comp_id: np.ndarray):
    adj_comp = [defaultdict(set) for _ in range(nV)]
    for t,(a,b,c) in enumerate(tris):
        comp = int(comp_id[t])
        adj_comp[a][comp].update([b,c])
        adj_comp[b][comp].update([a,c])
        adj_comp[c][comp].update([a,b])
    return adj_comp

# ---------- NEW: smooth-edge global vertex adjacency (won't cross sharp edges) ----------
def _edge_to_triangles(tris: np.ndarray):
    edge_map = defaultdict(list)
    for tid,(a,b,c) in enumerate(tris):
        for u,v in ((a,b),(b,c),(c,a)):
            key = (u,v) if u < v else (v,u)
            edge_map[key].append(tid)
    return edge_map

def build_vertex_adjacency_smooth(points: np.ndarray, tris: np.ndarray, theta_max_deg: float):
    """
    Build a global vertex graph using only 'smooth' edges.
    Edge (u,v) is smooth if boundary OR any pair of incident triangles has
    |dot(n_i, n_j)| >= cos(theta_max_deg). This prevents crossing sharp edges.
    """
    nV = int(np.max(tris)) + 1 if tris.size else 0
    adj = [set() for _ in range(nV)]
    N = triangle_normals(points, tris)
    cos_thr = float(np.cos(np.radians(theta_max_deg)))
    edge_map = _edge_to_triangles(tris)

    for (u,v), tids in edge_map.items():
        smooth = False
        if len(tids) < 2:
            smooth = True  # boundary edge -> allow
        else:
            # check all pairs (usually 2 tris)
            for i in range(len(tids)):
                for j in range(i+1, len(tids)):
                    ni = N[tids[i]]
                    nj = N[tids[j]]
                    if abs(float(np.dot(ni, nj))) >= cos_thr:
                        smooth = True
                        break
                if smooth:
                    break
        if smooth:
            adj[u].add(v); adj[v].add(u)
    return adj

def one_ring_comp(v: int, comp: int, adj_comp):
    s = set(adj_comp[v].get(comp, set()))
    s.add(v)
    return np.array(sorted(s), dtype=int)

def two_ring_comp(v: int, comp: int, adj_comp):
    r1 = set(adj_comp[v].get(comp, set()))
    r2 = set(r1)
    for u in r1:
        r2.update(adj_comp[u].get(comp, set()))
    r2.add(v)
    return np.array(sorted(r2), dtype=int)

# ---------- UPDATED: grow to 3rd/4th ring inside component; then smooth-global fallback ----------
def adaptive_patch_in_component(v: int, comp: int, adj_comp, min_pts: int = 10,
                                adj_smooth=None, max_rings: int = 4, global_rings: int = 4):
    # Start with 1-ring within the same component
    patch = set(adj_comp[v].get(comp, set()))
    patch.add(v)
    if len(patch) >= min_pts:
        return np.array(sorted(patch), dtype=int)

    # Expand ring-by-ring within the component, up to max_rings (1..4)
    frontier = set(patch)
    ring = 1
    while len(patch) < min_pts and ring < max_rings:
        new_frontier = set()
        for u in frontier:
            new_frontier |= adj_comp[u].get(comp, set())
        new_frontier -= patch
        if not new_frontier:
            break
        patch |= new_frontier
        frontier = new_frontier
        ring += 1

    # If still short, expand using the smooth-global vertex graph (does NOT cross sharp edges)
    if len(patch) < min_pts and adj_smooth is not None:
        frontier = set(patch)
        visited = set(patch)
        rounds = 0
        while len(patch) < min_pts and rounds < global_rings:
            new_frontier = set()
            for u in frontier:
                new_frontier |= adj_smooth[u]
            new_frontier -= visited
            if not new_frontier:
                break
            patch |= new_frontier
            visited |= new_frontier
            frontier = new_frontier
            rounds += 1

    return np.array(sorted(patch), dtype=int)

# =====================================================================
#                          PLANARITY TEST
# =====================================================================

def is_patch_planar(points: np.ndarray,
                    idx: np.ndarray,
                    plane_rel_tol: float = 1e-3,
                    plane_abs_tol: float = 1e-9):
    """
    PCA/SVD-based plane fit on the ABC patch.
    Returns (is_planar, max_abs_distance, scale, normal, centroid)
    """
    P = points[idx].astype(float)
    if len(P) < 3:
        return True, 0.0, 0.0, np.array([0,0,1.0]), P.mean(axis=0) if len(P) else np.zeros(3)
    c = P.mean(axis=0)
    Q = P - c
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    n = Vt[-1]
    dists = np.abs(Q @ n)
    max_d = float(np.max(dists))
    bbox = np.ptp(P, axis=0)
    scale = float(np.linalg.norm(bbox))
    thr = max(plane_abs_tol, plane_rel_tol * max(scale, 1e-12))
    return (max_d <= thr), max_d, scale, n, c

# =====================================================================
#                        PER-TRIANGLE FIT WRAPPER
# =====================================================================

def fit_coeffs_for_triangle(points: np.ndarray, idxA, idxB, idxC, idxABC,
                            irls_iters, snap_rel, snap_abs, canonize_abc,
                            tri_id=None, debug_j=False):
    coA   = fit_quadric_stable(points[idxA],   irls_iters=irls_iters,
                               snap_rel=snap_rel, snap_abs=snap_abs, canonize_xy=False,
                               debug_j=debug_j, label=f"T{tri_id} A")
    coB   = fit_quadric_stable(points[idxB],   irls_iters=irls_iters,
                               snap_rel=snap_rel, snap_abs=snap_abs, canonize_xy=False,
                               debug_j=debug_j, label=f"T{tri_id} B")
    coC   = fit_quadric_stable(points[idxC],   irls_iters=irls_iters,
                               snap_rel=snap_rel, snap_abs=snap_abs, canonize_xy=False,
                               debug_j=debug_j, label=f"T{tri_id} C")
    coABC = fit_quadric_stable(points[idxABC], irls_iters=irls_iters,
                               snap_rel=snap_rel, snap_abs=snap_abs, canonize_xy=canonize_abc,
                               debug_j=debug_j, label=f"T{tri_id} ABC")
    return coABC, coA, coB, coC

# =====================================================================
#                            MAIN PIPELINE
# =====================================================================

def compute_rimsafe_alltri(
    msh_path: str,
    out_csv: str,
    min_pts: int = 10,
    theta_max_deg: float = 35.0,
    plane_rel_tol: float = 1e-3,
    plane_abs_tol: float = 1e-9,
    irls_iters: int = 0,           # start with 0; raise to 1–2 if needed
    snap_rel: float = 5e-6,
    snap_abs: float = 1e-9,
    canonize_abc: bool = True,     # snap A,B to +1,-1 when safe (ABC only)
    debug_j: bool = False,
):
    mesh = meshio.read(msh_path)
    points = mesh.points.astype(float)
    tris = get_all_surface_triangles(mesh)
    print(f"[mesh] points={len(points)}  surface_triangles={len(tris)}")

    comp_id = smooth_components(points, tris, theta_max_deg=theta_max_deg)
    comp_sizes = np.bincount(comp_id, minlength=comp_id.max()+1)
    adj_comp = build_vertex_adjacency_by_component(len(points), tris, comp_id)
    adj_smooth = build_vertex_adjacency_smooth(points, tris, theta_max_deg=theta_max_deg)  # NEW

    rows = []
    cols_coeff = list("ABCDEFGHIJ")
    skipped_flat = 0

    for tri_id in range(len(tris)):
        a,b,c = map(int, tris[tri_id])
        comp = int(comp_id[tri_id])

        # Build ABC patch indices (auto-grow in-component up to 4 rings, then smooth-global fallback)
        idxA   = adaptive_patch_in_component(a, comp, adj_comp, min_pts=min_pts, adj_smooth=adj_smooth)
        idxB   = adaptive_patch_in_component(b, comp, adj_comp, min_pts=min_pts, adj_smooth=adj_smooth)
        idxC   = adaptive_patch_in_component(c, comp, adj_comp, min_pts=min_pts, adj_smooth=adj_smooth)
        idxABC = np.unique(np.concatenate([idxA, idxB, idxC]))

        # Planarity test on ABC patch
        planar, _, _, _, _ = is_patch_planar(
            points, idxABC, plane_rel_tol=plane_rel_tol, plane_abs_tol=plane_abs_tol
        )
        if planar:
            skipped_flat += 1
            continue

        # Fits
        sizes = dict(A=len(idxA), B=len(idxB), C=len(idxC), ABC=len(idxABC))
        coABC, coA, coB, coC = fit_coeffs_for_triangle(
            points, idxA, idxB, idxC, idxABC,
            irls_iters=irls_iters, snap_rel=snap_rel, snap_abs=snap_abs,
            canonize_abc=canonize_abc, tri_id=tri_id, debug_j=debug_j
        )

        row = {
            "triangle_id": tri_id,
            "component_id": int(comp),
            "component_size": int(comp_sizes[comp]),
            "A_id": a, "Ax": points[a,0], "Ay": points[a,1], "Az": points[a,2],
            "B_id": b, "Bx": points[b,0], "By": points[b,1], "Bz": points[b,2],
            "C_id": c, "Cx": points[c,0], "Cy": points[c,1], "Cz": points[c,2],
            "nA": sizes["A"], "nB": sizes["B"], "nC": sizes["C"], "nABC": sizes["ABC"],
        }

        def pack(prefix, coeff):
            if coeff is None or not np.all(np.isfinite(coeff)):
                for k in cols_coeff:
                    row[f"{prefix}_{k}"] = np.nan
            else:
                for k,val in zip(cols_coeff, coeff):
                    row[f"{prefix}_{k}"] = float(val)

        pack("ABC", coABC); pack("A", coA); pack("B", coB); pack("C", coC)
        rows.append(row)

        if (len(rows)) % 1000 == 0:
            print(f"[progress] kept {len(rows)} rows  |  skipped_flat {skipped_flat}")

    df = pd.DataFrame(rows)

    # Ensure required columns are present and ordered
    ordered_cols = [
        "triangle_id","component_id","component_size",
        "A_id","Ax","Ay","Az","B_id","Bx","By","Bz","C_id","Cx","Cy","Cz",
        "nA","nB","nC","nABC",
        *[f"ABC_{k}" for k in cols_coeff],
        *[f"A_{k}"   for k in cols_coeff],
        *[f"B_{k}"   for k in cols_coeff],
        *[f"C_{k}"   for k in cols_coeff],
    ]
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df.reindex(columns=ordered_cols)

    df.to_csv(out_csv, index=False)
    kept = len(df)
    print(f"[write] CSV -> {out_csv}  (rows kept={kept}, skipped_flat={skipped_flat}, total_tris={len(tris)})")

    if kept == 0:
        print("[warn] No rows kept (all ABC patches planar under tolerances).")
    else:
        print("[ok] Non-planar ABC rows written.")
    return df, len(tris), skipped_flat

# =====================================================================
#                                CLI
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Rim-safe quadric fitting on ALL surface triangles (no physical IDs), "
            "paraboloid-first canonization (A=+1, keep J), smart normalization for others, "
            "flat-plane guard, robust fit (1× convention)."
        )
    )
    ap.add_argument("--msh", type=str, default=EMBEDDED_MESH_PATH,
                    help=f"Input .msh path (default: {EMBEDDED_MESH_PATH})")
    ap.add_argument("--out", type=str, default=None,
                    help="Output CSV (default: <mesh_basename>_COEFFS.csv)")
    ap.add_argument("--min-pts", type=int, default=10,
                    help="Min points for patch (1-ring if >=min_pts else 2-ring)")
    ap.add_argument("--theta-max-deg", type=float, default=35.0,
                    help="Max normal angle for smooth adjacency (deg)")
    ap.add_argument("--plane-rel", type=float, default=1e-3,
                    help="Relative planarity tolerance (× patch size)")
    ap.add_argument("--plane-abs", type=float, default=1e-9,
                    help="Absolute planarity tolerance (length units)")
    ap.add_argument("--irls", type=int, default=0,
                    help="IRLS iterations for robust quadric fit (start with 0)")
    ap.add_argument("--snap-rel", type=float, default=5e-6,
                    help="Relative snap threshold for tiny coefficients (A..I)")
    ap.add_argument("--snap-abs", type=float, default=1e-9,
                    help="Absolute snap threshold for tiny coefficients (A..I)")
    ap.add_argument("--no-canon-abc", action="store_true",
                    help="Disable ABC A/B canonization to +1/-1 (default: enabled)")
    ap.add_argument("--debug-j", action="store_true",
                    help="Print J (and A,H) before any normalization for each fitted block.")
    args = ap.parse_args()

    msh_path = args.msh
    if not os.path.isabs(msh_path):
        msh_path = os.path.join(SCRIPT_DIR, msh_path)
    if not os.path.exists(msh_path):
        raise FileNotFoundError(f"Mesh not found: {msh_path}")

    if args.out:
        out_csv = args.out
    else:
        mesh_dir  = os.path.dirname(msh_path)
        mesh_base = os.path.basename(msh_path)
        if mesh_base.lower().endswith(".msh"):
            mesh_base = mesh_base[:-4]
        out_csv = os.path.join(mesh_dir, f"{mesh_base}_COEFFS.csv")

    compute_rimsafe_alltri(
        msh_path, out_csv,
        min_pts=args.min_pts,
        theta_max_deg=args.theta_max_deg,
        plane_rel_tol=args.plane_rel,
        plane_abs_tol=args.plane_abs,
        irls_iters=args.irls,
        snap_rel=args.snap_rel,
        snap_abs=args.snap_abs,
        canonize_abc=(not args.no_canon_abc),
        debug_j=args.debug_j,
    )

if __name__ == "__main__":
    import argparse, os
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Part-1: fit quadric coeffs for a .msh")
    ap.add_argument("--mesh", required=True, help="Path to .msh (quote if it has spaces)")
    ap.add_argument("--out", default=None, help="Output CSV (default: <mesh>_COEFFS.csv next to mesh)")
    ap.add_argument("--min_pts", type=int, default=10)
    ap.add_argument("--theta_max_deg", type=float, default=35.0)
    ap.add_argument("--plane_rel_tol", type=float, default=1e-3)
    ap.add_argument("--plane_abs_tol", type=float, default=1e-9)
    ap.add_argument("--irls_iters", type=int, default=0)
    ap.add_argument("--snap_rel", type=float, default=5e-6)
    ap.add_argument("--snap_abs", type=float, default=1e-9)
    ap.add_argument("--canonize_abc", action="store_true", default=True)
    ap.add_argument("--no_canonize_abc", dest="canonize_abc", action="store_false")
    ap.add_argument("--debug_j", action="store_true", default=False)
    args = ap.parse_args()

    mesh_path = os.path.abspath(args.mesh)
    out_csv = args.out or os.path.join(os.path.dirname(mesh_path),
                                       f"{Path(mesh_path).stem}_COEFFS.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    df, total_tris, skipped_flat = compute_rimsafe_alltri(
        mesh_path, out_csv,
        min_pts=args.min_pts,
        theta_max_deg=args.theta_max_deg,
        plane_rel_tol=args.plane_rel_tol,
        plane_abs_tol=args.plane_abs_tol,
        irls_iters=args.irls_iters,
        snap_rel=args.snap_rel,
        snap_abs=args.snap_abs,
        canonize_abc=args.canonize_abc,
        debug_j=args.debug_j,
    )

    # Force-write to guarantee the file exists even if the function skips writing
    try:
        df.to_csv(out_csv, index=False)
    except Exception as e:
        print(f"[warn] DataFrame write failed: {e}")

    print(f"[ok] triangles: {total_tris} | skipped planar: {skipped_flat}")
    print(f"[write] {os.path.abspath(out_csv)}")

