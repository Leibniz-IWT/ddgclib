import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def icosahedron():
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ],
        dtype=float,
    )
    verts /= np.linalg.norm(verts, axis=1)[:, None]
    faces = np.array(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=int,
    )
    return orient_faces(verts, faces)


def orient_faces(verts, faces):
    out = faces.copy()
    for i, (a, b, c) in enumerate(out):
        A, B, C = verts[a], verts[b], verts[c]
        if np.dot(np.cross(B - A, C - A), A + B + C) < 0:
            out[i] = (a, c, b)
    return verts, out


def subdivide(verts, faces):
    verts_list = [v.copy() for v in verts]
    midpoint_cache = {}

    def midpoint(i, j):
        key = tuple(sorted((int(i), int(j))))
        if key in midpoint_cache:
            return midpoint_cache[key]
        m = verts[key[0]] + verts[key[1]]
        m = m / np.linalg.norm(m)
        idx = len(verts_list)
        verts_list.append(m)
        midpoint_cache[key] = idx
        return idx

    new_faces = []
    for a, b, c in faces:
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        new_faces.extend(
            [
                (a, ab, ca),
                (b, bc, ab),
                (c, ca, bc),
                (ab, bc, ca),
            ]
        )
    return orient_faces(np.array(verts_list), np.array(new_faces, dtype=int))


def tangent_frame(n):
    n = np.asarray(n, dtype=float)
    n /= np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, n)) > 0.8:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(helper, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2, n


def signed_area2(uv):
    return (
        (uv[1, 0] - uv[0, 0]) * (uv[2, 1] - uv[0, 1])
        - (uv[1, 1] - uv[0, 1]) * (uv[2, 0] - uv[0, 0])
    )


def tri_moment_2(uv):
    """Return signed area, int u^2, int u v, int v^2 over a 2D triangle."""
    area = 0.5 * signed_area2(uv)
    u = uv[:, 0]
    v = uv[:, 1]
    int_u2 = area / 6.0 * (
        np.sum(u * u) + u[0] * u[1] + u[1] * u[2] + u[2] * u[0]
    )
    int_v2 = area / 6.0 * (
        np.sum(v * v) + v[0] * v[1] + v[1] * v[2] + v[2] * v[0]
    )
    # Product integral from barycentric moments:
    # int u v = A/12 * sum_i u_i v_i + A/24 * sum_{i!=j} u_i v_j
    int_uv = area / 12.0 * np.sum(u * v)
    int_uv += area / 24.0 * sum(u[i] * v[j] for i in range(3) for j in range(3) if i != j)
    return area, int_u2, int_uv, int_v2


def graph_patch_volume_from_quadratic(uv, coeffs):
    """Volume contribution from X=p0+u e1+v e2+h n for h=quadratic.

    coeffs are a,b,c,d,e,f for h=a u^2 + b u v + c v^2 + d u + e v + f.
    For p0=n0 on the unit sphere, the surface-integral contribution is
    1/3 * int(1 + h - u h_u - v h_v) dA
    = 1/3 * int(1 + f - a u^2 - b u v - c v^2) dA.
    """
    a, b, c, _d, _e, f = coeffs
    area, int_u2, int_uv, int_v2 = tri_moment_2(uv)
    return (area * (1.0 + f) - a * int_u2 - b * int_uv - c * int_v2) / 3.0


def graph_patch_volume_from_quadratic_gq(uv, coeffs):
    """Degree-2 triangle Gaussian quadrature for a local quadratic graph."""
    a, b, c, _d, _e, f = coeffs
    area = 0.5 * signed_area2(uv)
    bary = np.array(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ]
    )
    qp = bary @ uv
    u = qp[:, 0]
    v = qp[:, 1]
    integrand = (1.0 + f - a * u * u - b * u * v - c * v * v) / 3.0
    return area / 3.0 * float(np.sum(integrand))


def pl_volume(verts, faces):
    total = 0.0
    for a, b, c in faces:
        A, B, C = verts[a], verts[b], verts[c]
        total += np.linalg.det(np.column_stack((A, B, C))) / 6.0
    return total


def build_vertex_face_adjacency(faces, nverts):
    adj = [[] for _ in range(nverts)]
    for fi, face in enumerate(faces):
        for v in face:
            adj[int(v)].append(fi)
    return adj


def one_ring_patch_vertices(face, faces, vertex_faces):
    candidate_faces = set()
    for v in face:
        candidate_faces.update(vertex_faces[int(v)])
    verts = set()
    for fi in candidate_faces:
        verts.update(int(v) for v in faces[fi])
    return sorted(verts)


def paraboloid_volumes(verts, faces):
    vertex_faces = build_vertex_face_adjacency(faces, len(verts))
    tangent_total = 0.0
    fitted_total = 0.0

    for face in faces:
        pts = verts[face]
        n0 = pts.mean(axis=0)
        n0 /= np.linalg.norm(n0)
        p0 = n0.copy()
        e1, e2, n0 = tangent_frame(n0)

        rel = pts - p0
        uv = np.column_stack((rel @ e1, rel @ e2))
        if signed_area2(uv) < 0:
            uv = uv[[0, 2, 1]]

        # Ideal local quadratic model integrated by Gaussian quadrature.
        tangent_total += graph_patch_volume_from_quadratic_gq(
            uv, (-0.5, 0.0, -0.5, 0.0, 0.0, 0.0)
        )

        patch_ids = one_ring_patch_vertices(face, faces, vertex_faces)
        patch_pts = verts[patch_ids]
        patch_rel = patch_pts - p0
        u = patch_rel @ e1
        v = patch_rel @ e2
        h = patch_rel @ n0
        X = np.column_stack((u * u, u * v, v * v, u, v, np.ones_like(u)))
        coeffs, *_ = np.linalg.lstsq(X, h, rcond=None)
        fitted_total += graph_patch_volume_from_quadratic(uv, coeffs)

    return tangent_total, fitted_total


def mesh_size(verts, faces):
    edges = set()
    for a, b, c in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            edges.add(tuple(sorted((int(i), int(j)))))
    lens = np.array([np.linalg.norm(verts[i] - verts[j]) for i, j in edges])
    return float(lens.max()), float(lens.mean())


def fit_order(h, err):
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    mask = err > 1e-15
    p = np.polyfit(np.log(h[mask]), np.log(err[mask]), 1)
    return float(p[0])


def local_orders(h, err):
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    return np.log(err[1:] / err[:-1]) / np.log(h[1:] / h[:-1])


def save_plot(rows):
    import matplotlib.pyplot as plt

    out = Path(__file__).with_name("sphere_paraboloid_order_test_with_orders.png")
    h = np.array([r["hmean"] for r in rows])
    faces = np.array([r["faces"] for r in rows])
    series = [
        ("PL enclosed volume", np.array([r["pl_err"] for r in rows]), "o"),
        ("Tangent paraboloid", np.array([r["tangent_para_err"] for r in rows]), "s"),
        ("Fitted paraboloid", np.array([r["fit_para_err"] for r in rows]), "^"),
    ]

    fig, (ax_err, ax_p) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for name, err, marker in series:
        refined_p = fit_order(h[2:], err[2:])
        ax_err.loglog(
            h,
            100.0 * err,
            marker=marker,
            linewidth=2.2,
            label=f"{name} (p={refined_p:.2f})",
        )

    ref = 100.0 * series[1][1][-1] * (h / h[-1]) ** 2
    ax_err.loglog(h, ref, "k--", linewidth=1.6, label=r"$O(h^2)$ reference")
    ax_err.invert_xaxis()
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.set_xlabel("mean edge length h")
    ax_err.set_ylabel("relative volume error (%)")
    ax_err.set_title("Error decreases like h^2")
    ax_err.legend(fontsize=9)

    intervals = np.arange(1, len(rows))
    labels = [f"{faces[i - 1]}\n->\n{faces[i]}" for i in intervals]
    for name, err, marker in series:
        ax_p.plot(intervals, local_orders(h, err), marker=marker, linewidth=2.2, label=name)

    ax_p.axhline(2.0, color="k", linestyle="--", linewidth=1.5, label="second order")
    ax_p.set_xticks(intervals)
    ax_p.set_xticklabels(labels, fontsize=8)
    ax_p.set_ylim(0.8, 2.4)
    ax_p.grid(True, alpha=0.3)
    ax_p.set_xlabel("face-count refinement")
    ax_p.set_ylabel("measured local order")
    ax_p.set_title("Local order approaches 2")
    ax_p.legend(fontsize=9, loc="lower right")

    fig.suptitle("Unit sphere volume: paraboloid reconstruction used on a sphere", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"\nSaved plot: {out}")


def main():
    true_volume = 4.0 * math.pi / 3.0
    verts, faces = icosahedron()
    rows = []
    for level in range(0, 6):
        if level > 0:
            verts, faces = subdivide(verts, faces)
        hmax, hmean = mesh_size(verts, faces)
        vpl = pl_volume(verts, faces)
        vtan, vfit = paraboloid_volumes(verts, faces)
        rows.append(
            {
                "level": level,
                "faces": len(faces),
                "hmean": hmean,
                "hmax": hmax,
                "pl_err": abs(vpl - true_volume) / true_volume,
                "tangent_para_err": abs(vtan - true_volume) / true_volume,
                "fit_para_err": abs(vfit - true_volume) / true_volume,
            }
        )

    print("Unit sphere volume test: paraboloid reconstruction used on a sphere")
    print(f"V_exact = {true_volume:.16f}")
    print(
        "level faces h_mean      PL_rel_err     tangent_parab_rel_err  fit_parab_rel_err"
    )
    for r in rows:
        print(
            f"{r['level']:>2d} {r['faces']:>6d} {r['hmean']:.6e} "
            f"{r['pl_err']:.6e} {r['tangent_para_err']:.6e} {r['fit_para_err']:.6e}"
        )

    h = [r["hmean"] for r in rows]
    print()
    print("Observed order p from log(error) ~ p log(h_mean) + c:")
    print(f"  PL sphere volume:                 {fit_order(h, [r['pl_err'] for r in rows]):.3f}")
    print(
        f"  tangent paraboloid on sphere:     {fit_order(h, [r['tangent_para_err'] for r in rows]):.3f}"
    )
    print(
        f"  one-ring fitted paraboloid:       {fit_order(h, [r['fit_para_err'] for r in rows]):.3f}"
    )
    print()
    print(
        "Interpretation: this tests an ideal/local paraboloid model and a simple one-ring "
        "fitted paraboloid on sphere meshes. It is not the manuscript's sphere/ellipsoid "
        "closed-form quadric path."
    )
    save_plot(rows)


if __name__ == "__main__":
    main()
