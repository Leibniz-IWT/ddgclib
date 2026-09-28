#!/usr/bin/env python3
# coding: utf-8
"""
Quadric fitting (1× convention) with rim-safe growth.

This version:
- Planarity classification uses ONLY plane fitting on a topological k-ring
  neighborhood (no normal clustering).
- d_plane is computed against that fitted plane for the seed vertex.
- Writes:
    <mesh>_POINT_TYPES.csv  (per-vertex: type, rim, plane, d_plane, dihedral_angle)
    <mesh>_RINGS.csv        (per-triangle ring1/ring2 vertex IDs + ID_used_coeffs)
    <mesh>_COEFFS.csv       (per-triangle quadric coefficients + Max_Residual_ABC + Residual_Threshold)
    <mesh>_Tri_TYPE.csv     (per-triangle: plane/curved + max_d + thr + plane_rel_tol)

Single-source angle:
- THETA_MAX_DEG below is the ONLY place the default rim angle (degrees) is set.
"""

import os
import argparse
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional

import numpy as np
import pandas as pd
import meshio

# Single-source config

THETA_MAX_DEG = 55.0  # ← change here to adjust the default once

# Residual settings (for display threshold only)
RES_TOL_ABS_FLOOR = 1e-2 #1e-4 # 1e-2 for ellip  # 5e-3
# (RES_TOL_REL_MULT concept kept implicit; not needed to write the CSV columns)

# Helpers: coeffs

def rescale_smart(coeff: np.ndarray, tol_abs: float = 1e-9, tol_rel: float = 1e-6, prefer: str = "H") -> np.ndarray:
    c = np.array(coeff, float)
    A,B,C,D,E,F,G,H,I,J = map(float, c)
    scale_terms = max(1.0, float(np.max(np.abs(c[:9]))))
    thr = max(tol_abs, tol_rel * scale_terms, 1e-3)
    if abs(J) > thr:
        return c * (-1.0 / J)
    if prefer.upper() == "H" and abs(H) > thr:
        s = 1.0 / H
        return c * s
    return c

def _is_vertical_paraboloid_like(c: np.ndarray,
                                 rel_tol: float = 2e-3,
                                 abs_tol: float = 1e-7) -> bool:
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
    c = np.array(c, float)
    if not _is_vertical_paraboloid_like(c, rel_tol=rel_tol, abs_tol=abs_tol):
        return c
    A = c[0]
    if A == 0.0:
        return c
    s = 1.0 / abs(A)
    s *= np.sign(A)
    c *= s
    if abs(c[1] - 1.0) <= max(abs_tol, rel_tol):
        c[1] = 1.0
    return c

def coeffs_to_Q(c: np.ndarray) -> np.ndarray:
    A,B,C,D,E,F,G,H,I,J = [float(v) for v in c]
    return np.array([[A,   D/2, E/2, G/2],
                     [D/2, B,   F/2, H/2],
                     [E/2, F/2, C,   I/2],
                     [G/2, H/2, I/2, J  ]], dtype=float)

def Q_to_coeffs(Q: np.ndarray) -> np.ndarray:
    return np.array([Q[0,0], Q[1,1], Q[2,2],
                     2*Q[0,1], 2*Q[0,2], 2*Q[1,2],
                     2*Q[0,3], 2*Q[1,3], 2*Q[2,3],
                     Q[3,3]], dtype=float)

def _build_norm_transform(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P, float)
    t = P.mean(axis=0) if len(P) else np.zeros(3)
    Q = P - t
    if len(P) >= 3:
        _,_,Vt = np.linalg.svd(Q, full_matrices=False)
        R = Vt.T
        if np.linalg.det(R) < 0: R[:, -1] *= -1.0
    else:
        R = np.eye(3)
    s = np.sqrt(np.mean(np.sum(Q**2, axis=1))) if len(P) else 1.0
    if not np.isfinite(s) or s <= 1e-12: s = 1.0
    Y = (Q @ R) / s
    T = np.eye(4); T[:3,:3] = s*R; T[:3,3] = t
    return Y, T

def _transform_coeffs_from_norm(c_norm: np.ndarray, T: np.ndarray) -> np.ndarray:
    Qy = coeffs_to_Q(c_norm); Ti = np.linalg.inv(T)
    Qx = Ti.T @ Qy @ Ti
    return Q_to_coeffs(Qx)

def _design(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float)
    x,y,z = P[:,0], P[:,1], P[:,2]
    return np.column_stack([x*x, y*y, z*z, x*y, x*z, y*z, x, y, z, np.ones_like(x)])

def _fit_whitened(X: np.ndarray) -> np.ndarray:
    col = np.linalg.norm(X, axis=0); col[col==0] = 1.0
    Xw = X / col
    _,_,Vt = np.linalg.svd(Xw, full_matrices=False)
    c = Vt[-1] / col
    n = np.linalg.norm(c)
    return c/n if (n>0 and np.isfinite(n)) else np.full(10, np.nan)

def _grad_norm(P: np.ndarray, c: np.ndarray) -> np.ndarray:
    A,B,C,D,E,F,G,H,I,_ = c
    x,y,z = P[:,0], P[:,1], P[:,2]
    gx = 2*A*x + D*y + E*z + G
    gy = 2*B*y + D*x + F*z + H
    gz = 2*C*z + E*x + F*y + I
    return np.sqrt(gx*gx + gy*gy + gz*gz)

def _fit_whitened_irls(P: np.ndarray, iters: int = 0, eps: float = 1e-6) -> np.ndarray:
    X = _design(P)
    c = _fit_whitened(X)
    if not np.all(np.isfinite(c)): return c
    for _ in range(max(0,iters)):
        g = _grad_norm(P, c)
        w = 1.0 / np.maximum(g, eps)
        med = np.median(w) or 1.0
        if not np.isfinite(med) or med<=0: med = 1.0
        Xw = X * (w/med)[:,None]
        c = _fit_whitened(Xw)
        if not np.all(np.isfinite(c)): break
    return c

def _snap_small_A_to_I(c: np.ndarray, snap_rel: float = 5e-6, snap_abs: float = 1e-9) -> np.ndarray:
    c = np.array(c, float)
    scale = max(abs(c[0]), abs(c[1]), abs(c[2]), 1.0)
    thr = max(snap_abs, snap_rel*scale)
    c[:9] = np.where(np.abs(c[:9]) <= thr, 0.0, c[:9])
    return c

def fit_quadric_stable(P: np.ndarray, irls_iters: int = 0, snap_rel: float = 5e-6, snap_abs: float = 1e-9) -> np.ndarray:
    P = np.asarray(P, float)
    if P.shape[0] < 10: return np.full(10, np.nan)
    Y, T = _build_norm_transform(P)
    c_norm = _fit_whitened_irls(Y, iters=irls_iters)
    if not np.all(np.isfinite(c_norm)): return np.full(10, np.nan)
    c = _transform_coeffs_from_norm(c_norm, T)
    if _is_vertical_paraboloid_like(c):
        c = _canonize_paraboloid_A1(c)
    else:
        c = rescale_smart(c)
    c = _snap_small_A_to_I(c, snap_rel, snap_abs)
    return c

# Residual helpers

def _quadric_f_value(p: np.ndarray, c: np.ndarray) -> float:
    """Evaluate implicit quadric f(x,y,z)=Ax^2+By^2+Cz^2+Dxy+Exz+Fyz+Gx+Hy+Iz+J."""
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    A,B,C,D,E,F,G,H,I,J = [float(v) for v in c]
    return (A*x*x + B*y*y + C*z*z +
            D*x*y + E*x*z + F*y*z +
            G*x + H*y + I*z + J)

# Mesh / components

def get_all_surface_triangles(mesh: meshio.Mesh) -> np.ndarray:
    tris = []
    for cb in mesh.cells:
        if cb.type == "triangle":
            tris.append(cb.data.astype(int))
        elif cb.type == "triangle6":
            tris.append(cb.data[:, :3].astype(int))
    if tris:
        all_tris = np.vstack(tris)
        print(f"[triangles] faces: {all_tris.shape[0]}")
        return all_tris
    face_count = defaultdict(int); keep = {}
    for cb in mesh.cells:
        if cb.type.startswith("tetra"):
            for (a,b,c,d) in cb.data[:, :4]:
                for f in ((a,b,c),(a,b,d),(a,c,d),(b,c,d)):
                    key = tuple(sorted(f)); face_count[key] += 1
                    keep.setdefault(key, f)
    boundary = [keep[k] for k,cnt in face_count.items() if cnt == 1]
    if not boundary: raise RuntimeError("No surface triangles found.")
    return np.asarray(boundary, int)

def triangle_adjacency(tris: np.ndarray) -> List[List[int]]:
    edge_map = defaultdict(list)
    for t,(a,b,c) in enumerate(tris):
        for u,v in ((a,b),(b,c),(c,a)):
            key = (u,v) if u<v else (v,u)
            edge_map[key].append(t)
    nbrs: List[List[int]] = [[] for _ in range(len(tris))]
    for ids in edge_map.values():
        if len(ids) < 2: continue
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                ti,tj = ids[i], ids[j]
                nbrs[ti].append(tj); nbrs[tj].append(ti)
    return [sorted(set(v)) for v in nbrs]

def triangle_normals(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    A = points[tris[:,0]]; B = points[tris[:,1]]; C = points[tris[:,2]]
    N = np.cross(B - A, C - A)
    nrm = np.linalg.norm(N, axis=1, keepdims=True)
    return np.divide(N, np.maximum(nrm, 1e-20))

def smooth_components(points: np.ndarray, tris: np.ndarray, theta_max_deg: float) -> np.ndarray:
    N = triangle_normals(points, tris)
    nbrs = triangle_adjacency(tris)
    cos_thr = np.cos(np.radians(theta_max_deg))
    T = len(tris); comp_id = -np.ones(T, dtype=int); cur = 0
    for t in range(T):
        if comp_id[t] != -1: continue
        stack = [t]; comp_id[t] = cur
        while stack:
            i = stack.pop(); ni = N[i]
            for j in nbrs[i]:
                if comp_id[j] != -1: continue
                if abs(float(np.dot(ni, N[j]))) >= cos_thr:
                    comp_id[j] = cur; stack.append(j)
        cur += 1
    print(f"[smooth] components: {cur} (theta_max={theta_max_deg}°)")
    return comp_id

def build_vertex_adjacency_by_component(nV: int, tris: np.ndarray, comp_id: np.ndarray):
    adj_comp: List[Dict[int, Set[int]]] = [defaultdict(set) for _ in range(nV)]
    for t,(a,b,c) in enumerate(tris):
        comp = int(comp_id[t])
        adj_comp[a][comp].update([b,c])
        adj_comp[b][comp].update([a,c])
        adj_comp[c][comp].update([a,b])
    return adj_comp

def _edge_to_triangles(tris: np.ndarray):
    edge_map = defaultdict(list)
    for tid,(a,b,c) in enumerate(tris):
        for u,v in ((a,b),(b,c),(c,a)):
            key = (u,v) if u < v else (v,u)
            edge_map[key].append(tid)
    return edge_map

def build_vertex_adjacency_smooth(points: np.ndarray, tris: np.ndarray, theta_max_deg: float):
    nV = int(np.max(tris)) + 1 if tris.size else 0
    adj: List[Set[int]] = [set() for _ in range(nV)]
    N = triangle_normals(points, tris)
    cos_thr = float(np.cos(np.radians(theta_max_deg)))
    edge_map = _edge_to_triangles(tris)
    for (u,v), tids in edge_map.items():
        smooth = False
        if len(tids) < 2:
            smooth = False
        else:
            for i in range(len(tids)):
                for j in range(i+1, len(tids)):
                    if abs(float(np.dot(N[tids[i]], N[tids[j]]))) >= cos_thr:
                        smooth = True; break
                if smooth: break
        if smooth:
            adj[u].add(v); adj[v].add(u)
    return adj

# RIM stuff

def _boundary_vertices(tris: np.ndarray) -> Set[int]:
    from collections import Counter
    ecount = Counter()
    for a,b,c in tris:
        for e in ((a,b),(b,c),(c,a)):
            ecount[tuple(sorted(e))] += 1
    bverts = set()
    for (u,v), cnt in ecount.items():
        if cnt == 1:
            bverts.add(u); bverts.add(v)
    return bverts

def _vertex_graph(nV: int, tris: np.ndarray):
    G: List[Set[int]] = [set() for _ in range(nV)]
    for a,b,c in tris:
        G[a].update([b,c]); G[b].update([a,c]); G[c].update([a,b])
    return G

_RIM_DEPTH: Optional[np.ndarray] = None

def _rim_depth(nV: int, tris: np.ndarray, boundary_verts: Set[int]) -> np.ndarray:
    from collections import deque
    G = _vertex_graph(nV, tris)
    INF = 10**9
    depth = np.full(nV, INF, dtype=int)
    dq = deque()
    for v in boundary_verts:
        depth[v] = 0; dq.append(v)
    while dq:
        u = dq.popleft(); du = depth[u]
        for w in G[u]:
            if depth[w] > du + 1:
                depth[w] = du + 1; dq.append(w)
    return depth

# Rim by dihedral angle (> theta_max_deg)

def vertex_max_dihedral_deg(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    nV = points.shape[0]
    if tris.size == 0:
        return np.zeros(nV, dtype=float)
    N = triangle_normals(points, tris)
    edge_map = _edge_to_triangles(tris)
    out = np.zeros(nV, dtype=float)
    for (u,v), tids in edge_map.items():
        if len(tids) >= 2:
            max_abs_dot = -1.0
            for i in range(len(tids)):
                ni = N[tids[i]]
                for j in range(i+1, len(tids)):
                    nj = N[tids[j]]
                    d = abs(float(np.dot(ni, nj)))
                    if d > max_abs_dot:
                        max_abs_dot = d
            cosv = float(np.clip(max_abs_dot, -1.0, 1.0))
            ang = float(np.degrees(np.arccos(cosv)))
        else:
            ang = 180.0
        if ang > out[u]: out[u] = ang
        if ang > out[v]: out[v] = ang
    return out

def rim_vertices_by_dihedral(points: np.ndarray, tris: np.ndarray, theta_max_deg: float) -> np.ndarray:
    v_max_dih = vertex_max_dihedral_deg(points, tris)
    return (v_max_dih > float(theta_max_deg))

# Planarity

def is_patch_planar(points: np.ndarray, idx: np.ndarray, plane_rel_tol: float = 1e-3, plane_abs_tol: float = 1e-9):
    P = points[idx].astype(float)
    if len(P) < 3:
        return True, 0.0, 0.0, np.array([0,0,1.0]), P.mean(axis=0) if len(P) else np.zeros(3)
    c = P.mean(axis=0); Q = P - c
    _,_,Vt = np.linalg.svd(Q, full_matrices=False)
    n = Vt[-1]; d = np.abs(Q @ n)
    max_d = float(np.max(d))
    bbox = np.ptp(P, axis=0); scale = float(np.linalg.norm(bbox))
    thr = max(plane_abs_tol, plane_rel_tol*max(scale,1e-12))
    return (max_d <= thr), max_d, scale, n, c

# Fitting wrappers

def fit_coeffs_for_triangle(points, idxA, idxB, idxC, idxABC,
                            irls_iters, snap_rel, snap_abs,
                            plane_rel_tol, plane_abs_tol,
                            canonize_abc=True, tri_id=None, debug_j=False,
                            a=None, b=None, c=None):
    def maybe_fit(idx):
        if idx is None or len(idx) < 10:
            return np.full(10, np.nan)
        planar, *_ = is_patch_planar(points, idx, plane_rel_tol=plane_rel_tol, plane_abs_tol=plane_abs_tol)
        if planar:
            return np.full(10, np.nan)
        return fit_quadric_stable(points[idx], irls_iters=irls_iters, snap_rel=snap_rel, snap_abs=snap_abs)

    coA   = maybe_fit(idxA)
    coB   = maybe_fit(idxB)
    coC   = maybe_fit(idxC)

    # ABC fit with higher weight for vertices A, B, C (without changing their coordinates)
    if idxABC is None or len(idxABC) < 10:
        coABC = np.full(10, np.nan)
    else:
        planar_abc, *_ = is_patch_planar(points, idxABC, plane_rel_tol=plane_rel_tol, plane_abs_tol=plane_abs_tol)
        if planar_abc:
            coABC = np.full(10, np.nan)
        else:
            P_ABC = points[idxABC]
            if a is not None and b is not None and c is not None:
                weight_reps = 20
                extra_blocks = []
                for vid in (a, b, c):
                    if np.any(idxABC == vid):
                        extra_blocks.append(
                            np.repeat(points[vid][None, :], weight_reps, axis=0)
                        )
                if extra_blocks:
                    P_ABC = np.vstack([P_ABC, *extra_blocks])
            coABC = fit_quadric_stable(P_ABC, irls_iters=irls_iters, snap_rel=snap_rel, snap_abs=snap_abs)

    return coABC, coA, coB, coC

# Topological graphs & rings

def _vertex_graph_from_tris(nV: int, tris: np.ndarray) -> List[Set[int]]:
    G: List[Set[int]] = [set() for _ in range(nV)]
    for a,b,c in tris:
        G[a].update([b,c]); G[b].update([a,c]); G[c].update([a,b])
    return G

def _ring1_ring2_topo(a: int, b: int, c: int, G: List[Set[int]]):
    seeds = {a, b, c}
    ring1 = (G[a] | G[b] | G[c]) - seeds
    ring2 = set()
    for u in ring1:
        ring2 |= G[u]
    ring2 -= (ring1 | seeds)
    return sorted(ring1), sorted(ring2)

def _ring1_ring2_smooth(a: int, b: int, c: int, comp: int, adj_comp):
    seeds = {a, b, c}
    r1 = set()
    for v in seeds:
        r1 |= set(adj_comp[v].get(comp, set()))
    r1 -= seeds
    r2 = set()
    for u in r1:
        r2 |= set(adj_comp[u].get(comp, set()))
    r2 -= (r1 | seeds)
    return sorted(r1), sorted(r2)

# per-vertex max dihedral angle (degrees); boundary edges => 180°

def vertex_max_dihedral_deg(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    nV = points.shape[0]
    if tris.size == 0:
        return np.zeros(nV, dtype=float)
    N = triangle_normals(points, tris)
    edge_map = _edge_to_triangles(tris)
    out = np.zeros(nV, dtype=float)
    for (u,v), tids in edge_map.items():
        if len(tids) >= 2:
            max_abs_dot = -1.0
            for i in range(len(tids)):
                ni = N[tids[i]]
                for j in range(i+1, len(tids)):
                    nj = N[tids[j]]
                    d = abs(float(np.dot(ni, nj)))
                    if d > max_abs_dot:
                        max_abs_dot = d
            cosv = float(np.clip(max_abs_dot, -1.0, 1.0))
            ang = float(np.degrees(np.arccos(cosv)))
        else:
            ang = 180.0
        if ang > out[u]: out[u] = ang
        if ang > out[v]: out[v] = ang
    return out

# Face roles (boundary / interior / nonmanifold)

def _extract_tets(mesh: meshio.Mesh) -> np.ndarray:
    tets = []
    for cb in mesh.cells:
        if cb.type.startswith("tetra"):
            tets.append(cb.data[:, :4].astype(int))
    return np.vstack(tets) if tets else np.empty((0,4), dtype=int)

def _collect_surface_tris(mesh: meshio.Mesh) -> np.ndarray:
    tris = []
    for cb in mesh.cells:
        if cb.type == "triangle":
            tris.append(cb.data[:, :3].astype(int))
        elif cb.type == "triangle6":
            tris.append(cb.data[:, :3].astype(int))
    return np.vstack(tris) if tris else np.empty((0,3), dtype=int)

def compute_point_face_roles(mesh: meshio.Mesh):
    nV = mesh.points.shape[0]
    on_boundary = np.zeros(nV, dtype=bool)
    on_interior = np.zeros(nV, dtype=bool)
    on_nonman = np.zeros(nV, dtype=bool)

    tets = _extract_tets(mesh)
    if tets.size > 0:
        face_count = Counter()
        for (a,b,c,d) in tets:
            for f in ((a,b,c), (a,b,d), (a,c,d), (b,c,d)):
                face_count[tuple(sorted(f))] += 1
        for (u,v,w), cnt in face_count.items():
            if cnt == 1:
                on_boundary[u] = on_boundary[v] = on_boundary[w] = True
            elif cnt == 2:
                on_interior[u] = on_interior[v] = on_interior[w] = True
            else:
                on_nonman[u] = on_nonman[v] = on_nonman[w] = True
        return on_boundary, on_interior, on_nonman

    tris = _collect_surface_tris(mesh)
    if tris.size:
        u = np.unique(tris)
        on_boundary[u] = True
    return on_boundary, on_interior, on_nonman

# Planarity classifier (PLANE ONLY)

def _k_ring_topological(G: List[Set[int]], seed: int, min_pts: int = 10, max_k: int = 4) -> np.ndarray:
    patch = {seed}
    frontier = {seed}
    k = 0
    while len(patch) < min_pts and k < max_k:
        new_frontier = set()
        for u in frontier:
            new_frontier |= set(G[u])
        new_frontier -= patch
        if not new_frontier:
            break
        patch |= new_frontier
        frontier = new_frontier
        k += 1
    return np.array(sorted(patch), dtype=int)

def classify_points_and_write_csv(points: np.ndarray, tris: np.ndarray, rim_depth: np.ndarray, adj_smooth,
                                  out_csv_path: str,
                                  plane_rel_tol: float = 1e-3,
                                  plane_abs_tol: float = 1e-9,
                                  min_pts: int = 10,
                                  max_k: int = 4,
                                  rim_flags_override: Optional[np.ndarray] = None,
                                  plane_angle_deg: float = 5.0,
                                  face_role_triplets: Optional[Tuple[np.ndarray,np.ndarray,np.ndarray]] = None,
                                  G_topo: Optional[List[Set[int]]] = None):
    nV = points.shape[0]
    used_verts = set(np.unique(tris)) if tris.size else set()

    if face_role_triplets is not None:
        on_b, on_i, on_nm = face_role_triplets
    else:
        on_b = np.zeros(nV, dtype=bool)
        on_i = np.zeros(nV, dtype=bool)
        on_nm = np.zeros(nV, dtype=bool)

    if G_topo is None:
        G_topo = _vertex_graph_from_tris(nV= len(points), tris=tris)

    v_max_dih = vertex_max_dihedral_deg(points, tris)

    rows = []
    rim_count = planar_count = curved_count = 0
    type_arr = np.array([""] * nV, dtype=object)

    for v in range(nV):
        sp = (v in used_verts)
        rim_flag = bool(rim_flags_override[v]) if rim_flags_override is not None else (int(rim_depth[v]) == 0)
        plane_flag = False
        d_plane = float("nan")

        if sp:
            idx_used = _k_ring_topological(G_topo, v, min_pts=min_pts, max_k=max_k)
            if idx_used.size >= 3:
                p_planar, _, _, n, c = is_patch_planar(points, idx_used,
                                                       plane_rel_tol=plane_rel_tol,
                                                       plane_abs_tol=plane_abs_tol)
                plane_flag = bool(p_planar)
                pv = points[v].astype(float)
                d_plane = abs(float(np.dot(pv - c, n)))
            else:
                plane_flag = True

        if sp:
            if rim_flag:
                tval = "rim"; rim_count += 1
            elif plane_flag:
                tval = "planar"; planar_count += 1
            else:
                tval = "curved"; curved_count += 1
        else:
            tval = ""

        type_arr[v] = tval

        rows.append({
            "point_id": int(v),
            "type": tval,
            "rim": bool(rim_flag),
            "plane": bool(plane_flag),
            "surfacepoint": bool(sp),
            "d_plane": float(d_plane),
            "dihedral_angle": float(v_max_dih[v]),
            "on_boundary_face": bool(on_b[v]),
            "on_interior_face": bool(on_i[v]),
            "on_nonmanifold_face": bool(on_nm[v]),
        })

    cols = ["point_id", "type", "rim", "plane", "surfacepoint", "d_plane",
            "dihedral_angle", "on_boundary_face", "on_interior_face", "on_nonmanifold_face"]
    df_types = pd.DataFrame(rows, columns=cols)
    df_types.to_csv(out_csv_path, index=False)
    print(f"[write] point types -> {out_csv_path}  "
          f"(rim={rim_count}, planar={planar_count}, curved={curved_count}, "
          f"surface_points={sum(df_types.surfacepoint)}, total_points={nV})")

    surface_mask = df_types["surfacepoint"].to_numpy(dtype=bool)
    nonplanar_mask = (type_arr != "planar") & surface_mask
    return df_types, nonplanar_mask

# rim-safe, component-aware patch growth

def adaptive_patch_in_component(
    seed: int,
    comp: int,
    adj_comp: List[Dict[int, Set[int]]],
    min_pts: int = 10,
    adj_smooth: Optional[List[Set[int]]] = None,
    max_rings: int = 4,
    global_rings: int = 4,
    G_topo: Optional[List[Set[int]]] = None,
) -> np.ndarray:
    import numpy as _np

    def rim_ok(v: int) -> bool:
        return (int(_RIM_DEPTH[v]) >= 1) if int(_RIM_DEPTH[seed]) == 0 else True

    patch = {seed}
    frontier = {seed}

    rings = 0
    while len(patch) < min_pts and rings < max_rings and frontier:
        new_frontier = set()
        for u in frontier:
            for w in adj_comp[u].get(comp, set()):
                if w not in patch and rim_ok(w):
                    new_frontier.add(w)
        new_frontier -= patch
        if not new_frontier:
            break
        patch |= new_frontier
        frontier = new_frontier
        rings += 1

    if len(patch) < min_pts and global_rings > 0 and G_topo is not None:
        frontier = set(patch)
        rings = 0
        while len(patch) < min_pts and rings < global_rings and frontier:
            new_frontier = set()
            for u in frontier:
                for w in G_topo[u]:
                    if w not in patch and rim_ok(w):
                        new_frontier.add(w)
            new_frontier -= patch
            if not new_frontier:
                break
            patch |= new_frontier
            frontier = new_frontier
            rings += 1

    return _np.array(sorted(patch), dtype=int)

# Ring CSV helpers

def _ring1_ring2_for_csv(points, tris, comp_id, adj_comp):
    G_topo = _vertex_graph_from_tris(len(points), tris)
    rows = []
    for tri_id, (a,b,c) in enumerate(tris):
        comp = int(comp_id[tri_id])
        r1_topo, r2_topo = _ring1_ring2_topo(a, b, c, G_topo)
        r1_smooth, r2_smooth = _ring1_ring2_smooth(a, b, c, comp, adj_comp)
        rows.append({
            "triangle_id": tri_id,
            "A_id": int(a), "B_id": int(b), "C_id": int(c),
            "ring1_topo_ids": " ".join(map(str, r1_topo)),
            "ring2_topo_ids": " ".join(map(str, r2_topo)),
            "ring1_smooth_ids": " ".join(map(str, r1_smooth)),
            "ring2_smooth_ids": " ".join(map(str, r2_smooth)),
        })
    return pd.DataFrame(rows, columns=[
        "triangle_id","A_id","B_id","C_id",
        "ring1_topo_ids","ring2_topo_ids",
        "ring1_smooth_ids","ring2_smooth_ids"
    ])

# Main pipeline

def compute_rimsafe_alltri(
    msh_path: str,
    out_csv: str,
    min_pts: int = 10,
    theta_max_deg: Optional[float] = None,
    plane_rel_tol: float = 1e-3,
    plane_abs_tol: float = 1e-9,
    plane_angle_deg: float = 5.0,
    irls_iters: int = 0,
    snap_rel: float = 5e-6,
    snap_abs: float = 1e-9,
    canonize_abc: bool = True,
    debug_j: bool = False,
):
    if theta_max_deg is None:
        theta_max_deg = THETA_MAX_DEG

    mesh = meshio.read(msh_path)
    points = mesh.points.astype(float)

    tris = get_all_surface_triangles(mesh)
    print(f"[mesh] points={len(points)}  surface_triangles={len(tris)}")

    rim_flags = rim_vertices_by_dihedral(points, tris, theta_max_deg=theta_max_deg)

    vid = 115
    v_max_dih_all = vertex_max_dihedral_deg(points, tris)
    if 0 <= vid < v_max_dih_all.shape[0]:
        print(f"[RIM CHECK] theta_max_deg={theta_max_deg}  "
              f"dihedral(115)={v_max_dih_all[vid]:.6f}  rim_flags[115]={bool(rim_flags[vid])}")
    else:
        print(f"[RIM CHECK] vid {vid} out of range (nV={v_max_dih_all.shape[0]}); skipping per-vertex print.")

    on_boundary_face, on_interior_face, on_nonman_face = compute_point_face_roles(mesh)

    comp_id = smooth_components(points, tris, theta_max_deg=theta_max_deg)
    comp_sizes = np.bincount(comp_id, minlength=comp_id.max()+1)
    adj_comp = build_vertex_adjacency_by_component(len(points), tris, comp_id)
    adj_smooth = build_vertex_adjacency_smooth(points, tris, theta_max_deg=theta_max_deg)

    global _RIM_DEPTH
    bverts = _boundary_vertices(tris)
    _RIM_DEPTH = _rim_depth(len(points), tris, bverts)

    G_topo = _vertex_graph_from_tris(len(points), tris)

    base = os.path.splitext(os.path.abspath(msh_path))[0]
    out_types_csv = base + "_POINT_TYPES.csv"

    _, nonplanar_mask = classify_points_and_write_csv(
        points, tris, _RIM_DEPTH, adj_smooth,
        out_csv_path=out_types_csv,
        plane_rel_tol=plane_rel_tol,
        plane_abs_tol=plane_abs_tol,
        min_pts=min_pts,
        max_k=4,
        rim_flags_override=rim_flags,
        plane_angle_deg=plane_angle_deg,
        face_role_triplets=(on_boundary_face, on_interior_face, on_nonman_face),
        G_topo=G_topo,
    )

    tri_type_rows = []

    rows = []; cols_coeff = list("ABCDEFGHIJ"); skipped_flat = 0
    TARGET_TRI = 208

    tri_to_used_ids: Dict[int, str] = {}

    for tri_id, (a,b,c) in enumerate(tris):
        comp = int(comp_id[tri_id])

        idxA = adaptive_patch_in_component(a, comp, adj_comp,
                                           min_pts=min_pts, adj_smooth=adj_smooth,
                                           max_rings=4, global_rings=4, G_topo=G_topo)
        idxB = adaptive_patch_in_component(b, comp, adj_comp,
                                           min_pts=min_pts, adj_smooth=adj_smooth,
                                           max_rings=4, global_rings=4, G_topo=G_topo)
        idxC = adaptive_patch_in_component(c, comp, adj_comp,
                                           min_pts=min_pts, adj_smooth=adj_smooth,
                                           max_rings=4, global_rings=4, G_topo=G_topo)

        r1_topo, r2_topo = _ring1_ring2_topo(a, b, c, G_topo)
        r1_smooth, r2_smooth = _ring1_ring2_smooth(a, b, c, comp, adj_comp)

        selected: list = []
        for seed in (a, b, c):
            if 0 <= seed < nonplanar_mask.shape[0] and nonplanar_mask[seed]:
                if seed not in selected:
                    selected.append(int(seed))

        def add_filtered(dest: list, src_iter):
            for v in src_iter:
                v = int(v)
                if v < 0 or v >= nonplanar_mask.shape[0]:
                    continue
                if not nonplanar_mask[v]:
                    continue
                if v not in dest:
                    dest.append(v)

        add_filtered(selected, r1_smooth)
        if len(selected) <= 10:
            add_filtered(selected, r2_smooth)
            if len(selected) < 10:
                add_filtered(selected, r1_topo)
                if len(selected) < 10:
                    add_filtered(selected, r2_topo)

        idxABC = np.array(sorted(selected), dtype=int)

        if tri_id == TARGET_TRI:
            rd = _RIM_DEPTH
            tri_depths = (int(rd[a]), int(rd[b]), int(rd[c]))
            print(f"\n[DEBUG T{tri_id}] verts: A={a} B={b} C={c}  rim-depths={tri_depths}")
            print(f"[DEBUG T{tri_id}] |A|={len(idxA)}  |B|={len(idxB)}  |C|={len(idxC)}  |ABC|={len(idxABC)}")
            print(f"[DEBUG T{tri_id}] ABC IDs:", idxABC.tolist())

        planar_abc, max_d, scale, *_ = is_patch_planar(
            points, idxABC,
            plane_rel_tol=plane_rel_tol,
            plane_abs_tol=plane_abs_tol
        )
        thr = max(plane_abs_tol, plane_rel_tol * max(scale, 1e-12))

        tri_type_rows.append({
            "triangle_id": int(tri_id),
            "type": "plane" if planar_abc else "curved",
            "max_d": float(max_d),
            "thr": float(thr),
            "plane_rel_tol": float(plane_rel_tol),
        })

        if (not planar_abc) and (len(idxABC) >= 10):
            tri_to_used_ids[tri_id] = " ".join(map(str, idxABC.tolist()))
        else:
            tri_to_used_ids[tri_id] = ""

        if planar_abc:
            skipped_flat += 1
            continue

        coABC, coA, coB, coC = fit_coeffs_for_triangle(
            points, idxA, idxB, idxC, idxABC,
            irls_iters=irls_iters, snap_rel=snap_rel, snap_abs=snap_abs,
            plane_rel_tol=plane_rel_tol, plane_abs_tol=plane_abs_tol,
            canonize_abc=canonize_abc, tri_id=tri_id, debug_j=debug_j,
            a=a, b=b, c=c
        )

        # residuals at triangle vertices (ABC fit)
        if coABC is not None and np.all(np.isfinite(coABC)):
            fA = abs(_quadric_f_value(points[a], coABC))
            fB = abs(_quadric_f_value(points[b], coABC))
            fC = abs(_quadric_f_value(points[c], coABC))
            max_residual_abc = float(max(fA, fB, fC))
        else:
            max_residual_abc = float("nan")

        residual_threshold = float(1000 * RES_TOL_ABS_FLOOR * scale)  # = 5e-4 * scale

        row = {
            "triangle_id": tri_id,
            "component_id": int(comp),
            "component_size": int(comp_sizes[comp]),
            "A_id": int(a), "Ax": float(points[a,0]), "Ay": float(points[a,1]), "Az": float(points[a,2]),
            "B_id": int(b), "Bx": float(points[b,0]), "By": float(points[b,1]), "Bz": float(points[b,2]),
            "C_id": int(c), "Cx": float(points[c,0]), "Cy": float(points[c,1]), "Cz": float(points[c,2]),
            "nA": int(len(idxA)), "nB": int(len(idxB)), "nC": int(len(idxC)), "nABC": int(len(idxABC)),
            "Max_Residual_ABC": max_residual_abc,
            "Residual_Threshold": residual_threshold,
        }

        def pack(prefix, coeff):
            for k in cols_coeff: row[f"{prefix}_{k}"] = np.nan
            if coeff is not None and np.all(np.isfinite(coeff)):
                for k,val in zip(cols_coeff, coeff):
                    row[f"{prefix}_{k}"] = float(val)

        pack("ABC", coABC); pack("A", coA); pack("B", coB); pack("C", coC)
        rows.append(row)

        if (len(rows) % 1000) == 0:
            print(f"[progress] kept {len(rows)} rows | skipped_flat={skipped_flat}")

    # write COEFFS CSV
    df = pd.DataFrame(rows)
    ordered_cols = [
        "triangle_id","component_id","component_size",
        "A_id","Ax","Ay","Az","B_id","Bx","By","Bz","C_id","Cx","Cy","Cz",
        "nA","nB","nC","nABC",
        "Max_Residual_ABC","Residual_Threshold",
        *[f"ABC_{k}" for k in cols_coeff],
        *[f"A_{k}"   for k in cols_coeff],
        *[f"B_{k}"   for k in cols_coeff],
        *[f"C_{k}"   for k in cols_coeff],
    ]
    for col in ordered_cols:
        if col not in df.columns: df[col] = np.nan
    df = df.reindex(columns=ordered_cols)

    df.to_csv(out_csv, index=False)
    print(f"[write] CSV -> {out_csv}  (rows kept={len(df)}, skipped_flat={skipped_flat}, total_tris={len(tris)})")
    if len(df) == 0:
        print("[warn] No rows kept (all filtered ABC patches planar or too small).")
    else:
        print("[ok] Non-planar ABC rows written.")

    # write triangle TYPE CSV (now with plane_rel_tol)
    out_tri_type_csv = base + "_Tri_TYPE.csv"
    pd.DataFrame(tri_type_rows, columns=["triangle_id","type","max_d","thr","plane_rel_tol"]).to_csv(out_tri_type_csv, index=False)
    print(f"[write] tri types -> {out_tri_type_csv}  (triangles={len(tris)})")

    # write RINGS CSV (with ID_used_coeffs)
    df_rings = _ring1_ring2_for_csv(points, tris, comp_id, adj_comp)
    df_rings["ID_used_coeffs"] = df_rings["triangle_id"].map(lambda t: tri_to_used_ids.get(int(t), ""))
    out_rings_csv = base + "_RINGS.csv"
    df_rings.to_csv(out_rings_csv, index=False)
    print(f"[write] rings -> {out_rings_csv}  (triangles={len(tris)})")

    return df, len(tris), skipped_flat

# CLI

def main():
    ap = argparse.ArgumentParser(description="Quadric fitting with plane-only vertex classifier (topo k-ring) + ring diagnostics.")
    ap.add_argument("--msh", type=str, required=True, help="Input .msh path")
    ap.add_argument("--out", type=str, default=None, help="Output CSV path for triangle coeffs (default: <mesh>_COEFFS.csv)")
    ap.add_argument("--min_pts", type=int, default=10)
    ap.add_argument("--theta_max_deg", type=float, default=None, help="Override rim angle in degrees; default comes from THETA_MAX_DEG in code")
    ap.add_argument("--plane_rel_tol", type=float, default=1e-3)
    ap.add_argument("--plane_abs_tol", type=float, default=1e-9)
    ap.add_argument("--plane_angle_deg", type=float, default=5.0, help="Unused in plane-only mode (kept for compatibility)")
    ap.add_argument("--irls_iters", type=int, default=0)
    ap.add_argument("--snap_rel", type=float, default=5e-6)
    ap.add_argument("--snap_abs", type=float, default=1e-9)
    ap.add_argument("--canonize_abc", action="store_true", default=True)
    ap.add_argument("--no_canonize_abc", dest="canonize_abc", action="store_false")
    ap.add_argument("--debug_j", action="store_true", default=False)
    args = ap.parse_args()

    msh_path = os.path.abspath(args.msh)
    out_csv = args.out if args.out else os.path.splitext(msh_path)[0] + "_COEFFS.csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    compute_rimsafe_alltri(
        msh_path, out_csv,
        min_pts=args.min_pts,
        theta_max_deg=args.theta_max_deg,  # None → uses THETA_MAX_DEG
        plane_rel_tol=args.plane_rel_tol,
        plane_abs_tol=args.plane_abs_tol,
        plane_angle_deg=args.plane_angle_deg,
        irls_iters=args.irls_iters,
        snap_rel=args.snap_rel,
        snap_abs=args.snap_abs,
        canonize_abc=args.canonize_abc,
        debug_j=args.debug_j,
    )

if __name__ == "__main__":
    main()
