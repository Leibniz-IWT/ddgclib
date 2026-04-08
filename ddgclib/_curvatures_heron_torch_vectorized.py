"""
This file uses Heron's formula to compute the curvature contribution.

Original scalar routines are kept intact.
A new PyTorch-vectorized routine ``int_HNdC_ijk_vectorized`` is added,
together with analytical tests, an exactly equal-edged sphere test using a
regular icosahedron, and a random-batch benchmark.
"""

import time
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _select_vectorized_device_and_dtype():
    """
    Auto-select the best PyTorch backend for the vectorized path.

    Preference order:
    1. CUDA with float64
    2. MPS with float32 (MPS does not support float64)
    3. CPU with float64
    """
    if torch is None:
        raise ImportError('PyTorch is not installed.')

    if torch.cuda.is_available():
        return torch.device('cuda'), torch.float64

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        #return torch.device('mps'), torch.float32 # MPS does not support float64, so we use float32 here.
        return torch.device('cpu'), torch.float64
    

    return torch.device('cpu'), torch.float64


def _synchronize_device(device):
    """Synchronize asynchronous backends so timing is fair."""
    if torch is None:
        return

    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


# -----------------------------------------------------------------------------
# Original routines kept intact
# -----------------------------------------------------------------------------

def HNdC_ijk(e_ij, l_ij, l_jk, l_ik):
    """
    Computes the dual edge and dual area using Heron's formula.

    :param e_ij: vector, edge e_ij
    :param l_ij: float, length of edge ij
    :param l_jk: float, length of edge jk
    :param l_ik: float, length of edge ik
    :return: hnda_ijk: vector, curvature vector
             c_ijk: float, dual areas
    """
    lengths = [l_ij, l_jk, l_ik]
    lengths.sort()
    a = lengths[2]
    b = lengths[1]
    c = lengths[0]
    A = (1 / 4.0) * np.sqrt((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c)))
    w_ij = (1 / 8.0) * (l_jk ** 2 + l_ik ** 2 - l_ij ** 2) / A
    hnda_ijk = w_ij * e_ij
    h_ij = 0.5 * l_ij
    b_ij = abs(w_ij) * l_ij
    c_ijk = 0.5 * b_ij * h_ij
    return hnda_ijk, c_ijk


def A_i(v, n_i=None):
    """
    Compute the discrete normal area of vertex v_i.
    Kept intact from the original file.
    """
    if n_i is not None:
        n_i = v.x

    NdA_i = np.zeros(3)
    C_i = 0.0
    vi = v
    for vj in v.nn:
        e_i_int_e_j = vi.nn.intersection(vj.nn)
        e_ij = vj.x_a - vi.x_a
        e_ij = - e_ij
        vk = list(e_i_int_e_j)[0]
        e_ik = vk.x_a - vi.x_a

        if 1:
            wedge_ij_ik = np.cross(e_ij, e_ik)
            if np.dot(normalized(wedge_ij_ik)[0], n_i) < 0:
                print(f'e_ij_prev = {e_ij}')
                e_ij = vi.x_a - vj.x_a

        if len(e_i_int_e_j) == 1:
            pass
        else:
            vl = list(e_i_int_e_j)[1]
            e_jk = vk.x_a - vj.x_a
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)

            e_il = vl.x_a - vi.x_a
            e_jl = vl.x_a - vj.x_a
            l_il = np.linalg.norm(e_il)
            l_jl = np.linalg.norm(e_jl)
            hnda_ijl, c_ijl = HNdC_ijk(e_ij, l_ij, l_jl, l_il)

            HNdA_i += hnda_ijk
            HNdA_i += hnda_ijl
            C_i += c_ijl
            C_i += c_ijk

    return NdA_i


def hndA_i(v, n_i=None):
    """
    Compute the mean normal curvature of vertex.
    Kept intact from the original file.
    """
    if n_i is not None:
        n_i = v.x

    HNdA_i = np.zeros(3)
    C_i = 0.0
    vi = v
    for vj in v.nn:
        e_i_int_e_j = vi.nn.intersection(vj.nn)
        e_ij = vj.x_a - vi.x_a
        e_ij = - e_ij
        vk = list(e_i_int_e_j)[0]
        e_ik = vk.x_a - vi.x_a

        if 0:
            wedge_ij_ik = np.cross(e_ij, e_ik)
            if np.dot(normalized(wedge_ij_ik)[0], n_i) < 0:
                e_ij = vi.x_a - vj.x_a

        if len(e_i_int_e_j) == 1:
            vk = list(e_i_int_e_j)[0]
            e_ik = vk.x_a - vi.x_a
            e_jk = vk.x_a - vj.x_a
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)

            HNdA_i += hnda_ijk
            C_i += c_ijk

        else:
            vl = list(e_i_int_e_j)[1]
            e_jk = vk.x_a - vj.x_a
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)

            e_il = vl.x_a - vi.x_a
            e_jl = vl.x_a - vj.x_a
            l_il = np.linalg.norm(e_il)
            l_jl = np.linalg.norm(e_jl)
            hnda_ijl, c_ijl = HNdC_ijk(e_ij, l_ij, l_jl, l_il)

            HNdA_i += hnda_ijk
            HNdA_i += hnda_ijl
            C_i += c_ijl
            C_i += c_ijk

    return HNdA_i, C_i


def int_HNdC_ijk(e_ij, l_ij, l_jk, l_ik):
    """
    Computes the dual edge and dual area using Heron's formula.

    :param e_ij: vector, edge e_ij
    :param l_ij: float, length of edge ij
    :param l_jk: float, length of edge jk
    :param l_ik: float, length of edge ik
    :return: hnda_ijk: vector, curvature vector
             c_ijk: float, dual areas
    """
    lengths = [l_ij, l_jk, l_ik]
    lengths.sort()
    a = lengths[2]
    b = lengths[1]
    c = lengths[0]
    A = (1 / 4.0) * np.sqrt((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c)))
    w_ij = (1 / 8.0) * (l_jk ** 2 + l_ik ** 2 - l_ij ** 2) / A
    hnda_ijk = w_ij * e_ij
    h_ij = 0.5 * l_ij
    b_ij = abs(w_ij) * l_ij
    c_ijk = 0.5 * b_ij * h_ij
    return hnda_ijk, c_ijk


def int_hndA_i(v, n_i=None):
    """
    Compute the mean normal curvature of vertex.
    Kept intact from the original file.
    """
    if n_i is not None:
        n_i = v.x

    HNdA_i = np.zeros(3)
    C_i = 0.0
    vi = v
    for vj in v.nn:
        e_i_int_e_j = vi.nn.intersection(vj.nn)
        e_ij = vj.x_a - vi.x_a
        e_ij = - e_ij
        vk = list(e_i_int_e_j)[0]

        if len(e_i_int_e_j) == 1:
            vk = list(e_i_int_e_j)[0]
            e_ik = vk.x_a - vi.x_a
            e_jk = vk.x_a - vj.x_a
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)
            l_jk = np.linalg.norm(e_jk)

            e_hnda_ij = vj.hnda_i - vi.hnda_i
            hnda_ijk, c_ijk = int_HNdC_ijk(e_hnda_ij, l_ij, l_jk, l_ik)

            HNdA_i += hnda_ijk
            C_i += c_ijk

        else:
            vl = list(e_i_int_e_j)[1]
            e_jk = vk.x_a - vj.x_a
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(vk.x_a - vi.x_a)
            l_jk = np.linalg.norm(e_jk)
            e_hnda_ij = vj.hnda_i - vi.hnda_i
            hnda_ijk, c_ijk = int_HNdC_ijk(e_hnda_ij, l_ij, l_jk, l_ik)

            e_il = vl.x_a - vi.x_a
            e_jl = vl.x_a - vj.x_a
            l_il = np.linalg.norm(e_il)
            l_jl = np.linalg.norm(e_jl)
            e_hnda_ij = vj.hnda_i - vi.hnda_i
            hnda_ijl, c_ijl = int_HNdC_ijk(e_hnda_ij, l_ij, l_jk, l_ik)

            HNdA_i += hnda_ijk
            HNdA_i += hnda_ijl
            C_i += c_ijl
            C_i += c_ijk

    return HNdA_i, C_i


# -----------------------------------------------------------------------------
# New vectorized routine and test helpers
# -----------------------------------------------------------------------------

def int_HNdC_ijk_vectorized(e_ij, l_ij, l_jk, l_ik, device=None, dtype=None, return_numpy=False):
    """
    Vectorized PyTorch version of ``int_HNdC_ijk``.

    Parameters
    ----------
    e_ij : array_like, shape (3,) or (N, 3)
        Edge vector(s).
    l_ij, l_jk, l_ik : array_like, shape () or (N,)
        Edge lengths.
    device : str or torch.device, optional
        PyTorch device. Defaults to ``cpu``.
    dtype : torch.dtype, optional
        Defaults to ``torch.float64``.
    return_numpy : bool, optional
        If True, return NumPy arrays.
    """
    if torch is None:
        raise ImportError("PyTorch is required for int_HNdC_ijk_vectorized.")

    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device('cpu')
    else:
        device = torch.device(device)

    e_ij_t = torch.as_tensor(e_ij, dtype=dtype, device=device)
    l_ij_t = torch.as_tensor(l_ij, dtype=dtype, device=device)
    l_jk_t = torch.as_tensor(l_jk, dtype=dtype, device=device)
    l_ik_t = torch.as_tensor(l_ik, dtype=dtype, device=device)

    squeeze_output = False
    if e_ij_t.ndim == 1:
        if e_ij_t.shape[0] != 3:
            raise ValueError("e_ij must have shape (3,) or (N, 3).")
        e_ij_t = e_ij_t.unsqueeze(0)
        l_ij_t = l_ij_t.reshape(1)
        l_jk_t = l_jk_t.reshape(1)
        l_ik_t = l_ik_t.reshape(1)
        squeeze_output = True
    elif e_ij_t.ndim != 2 or e_ij_t.shape[1] != 3:
        raise ValueError("e_ij must have shape (3,) or (N, 3).")

    lengths = torch.stack((l_ij_t, l_jk_t, l_ik_t), dim=-1)
    lengths_sorted, _ = torch.sort(lengths, dim=-1)

    c = lengths_sorted[..., 0]
    b = lengths_sorted[..., 1]
    a = lengths_sorted[..., 2]

    heron_term = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    A = 0.25 * torch.sqrt(heron_term)

    w_ij = 0.125 * (l_jk_t ** 2 + l_ik_t ** 2 - l_ij_t ** 2) / A
    hnda_ijk = w_ij.unsqueeze(-1) * e_ij_t

    h_ij = 0.5 * l_ij_t
    b_ij = torch.abs(w_ij) * l_ij_t
    c_ijk = 0.5 * b_ij * h_ij

    if squeeze_output:
        hnda_ijk = hnda_ijk[0]
        c_ijk = c_ijk[0]

    if return_numpy:
        if isinstance(hnda_ijk, torch.Tensor):
            hnda_ijk = hnda_ijk.detach().cpu().numpy()
        if isinstance(c_ijk, torch.Tensor):
            c_ijk = c_ijk.detach().cpu().numpy()

    return hnda_ijk, c_ijk


def heron_mean_curvature_vectors(points, faces, device=None, dtype=None, return_numpy=True):
    """
    Assemble vertex mean-curvature vectors for a triangle mesh using the
    PyTorch-vectorized Heron kernel ``int_HNdC_ijk_vectorized``.

    Parameters
    ----------
    points : array_like, shape (N, 3)
        Vertex coordinates.
    faces : array_like, shape (M, 3)
        Triangle connectivity.
    device : str or torch.device, optional
        PyTorch device. If omitted, the preferred vectorized backend is chosen.
    dtype : torch.dtype, optional
        PyTorch dtype. If omitted, the preferred vectorized dtype is chosen.
    return_numpy : bool, optional
        If True, return a NumPy array. Otherwise return a torch.Tensor.
    """
    if torch is None:
        raise ImportError('PyTorch is required for heron_mean_curvature_vectors.')

    if device is None or dtype is None:
        auto_device, auto_dtype = _select_vectorized_device_and_dtype()
        if device is None:
            device = auto_device
        if dtype is None:
            dtype = auto_dtype

    device = torch.device(device)
    points_t = torch.as_tensor(points, dtype=dtype, device=device)
    faces_t = torch.as_tensor(faces, dtype=torch.long, device=device)

    if points_t.ndim != 2 or points_t.shape[1] != 3:
        raise ValueError('points must have shape (N, 3).')
    if faces_t.ndim != 2 or faces_t.shape[1] != 3:
        raise ValueError('faces must have shape (M, 3).')

    n = int(points_t.shape[0])
    H_vecs = torch.zeros((n, 3), dtype=dtype, device=device)
    if faces_t.numel() == 0:
        return H_vecs.detach().cpu().numpy() if return_numpy else H_vecs

    i = faces_t[:, 0]
    j = faces_t[:, 1]
    k = faces_t[:, 2]

    vi = points_t[i]
    vj = points_t[j]
    vk = points_t[k]

    e_ij = vj - vi
    e_ik = vk - vi
    e_ji = vi - vj
    e_jk = vk - vj
    e_ki = vi - vk
    e_kj = vj - vk

    l_ij = torch.linalg.norm(e_ij, dim=1)
    l_ik = torch.linalg.norm(e_ik, dim=1)
    l_jk = torch.linalg.norm(e_jk, dim=1)

    h_ij, _ = int_HNdC_ijk_vectorized(e_ij, l_ij, l_jk, l_ik, device=device, dtype=dtype)
    h_ik, _ = int_HNdC_ijk_vectorized(e_ik, l_ik, l_jk, l_ij, device=device, dtype=dtype)
    h_ji, _ = int_HNdC_ijk_vectorized(e_ji, l_ij, l_ik, l_jk, device=device, dtype=dtype)
    h_jk, _ = int_HNdC_ijk_vectorized(e_jk, l_jk, l_ik, l_ij, device=device, dtype=dtype)
    h_ki, _ = int_HNdC_ijk_vectorized(e_ki, l_ik, l_ij, l_jk, device=device, dtype=dtype)
    h_kj, _ = int_HNdC_ijk_vectorized(e_kj, l_jk, l_ij, l_ik, device=device, dtype=dtype)

    H_vecs.index_add_(0, i, h_ij)
    H_vecs.index_add_(0, i, h_ik)
    H_vecs.index_add_(0, j, h_ji)
    H_vecs.index_add_(0, j, h_jk)
    H_vecs.index_add_(0, k, h_ki)
    H_vecs.index_add_(0, k, h_kj)

    if return_numpy:
        return H_vecs.detach().cpu().numpy()
    return H_vecs


def normalized(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        if n == 0.0:
            return x, n
        return x / n, n
    n = np.linalg.norm(x, axis=1, keepdims=True)
    out = np.zeros_like(x)
    mask = n[:, 0] > 0.0
    out[mask] = x[mask] / n[mask]
    return out, n[:, 0]


def _max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _max_vector_norm_diff(a, b):
    return float(np.max(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1)))


def _max_rel_err_pct(a, b, eps=1e-30):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(100.0 * np.max(np.abs(a - b) / np.maximum(np.abs(b), eps)))


def _vector_norm_rel_err_pct(v, v_ref, eps=1e-30):
    v = np.asarray(v)
    v_ref = np.asarray(v_ref)
    num = np.linalg.norm(v - v_ref, axis=1)
    den = np.maximum(np.linalg.norm(v_ref, axis=1), eps)
    return 100.0 * num / den


def _print_case_header(name):
    print()
    print(f"case                 : {name}")


def _analytical_test_cases():
    sqrt3 = np.sqrt(3.0)
    return [
        {
            'name': 'equilateral_side_1',
            'description': '60 degree opposite angle, clean nonzero reference',
            'e_ij': np.array([1.0, 0.0, 0.0]),
            'l_ij': 1.0,
            'l_jk': 1.0,
            'l_ik': 1.0,
            'theory_hnda': np.array([1.0 / (2.0 * sqrt3), 0.0, 0.0]),
            'theory_c': 1.0 / (8.0 * sqrt3),
        },
        {
            'name': 'right_triangle_zero_weight',
            'description': '90 degree opposite angle, exact zero weight',
            'e_ij': np.array([-1.0, 1.0, 0.0]),
            'l_ij': np.sqrt(2.0),
            'l_jk': 1.0,
            'l_ik': 1.0,
            'theory_hnda': np.array([0.0, 0.0, 0.0]),
            'theory_c': 0.0,
        },
        {
            'name': 'obtuse_120deg',
            'description': '120 degree opposite angle, negative weight and positive c_ijk',
            'e_ij': np.array([-1.5, np.sqrt(3.0) / 2.0, 0.0]),
            'l_ij': np.sqrt(3.0),
            'l_jk': 1.0,
            'l_ik': 1.0,
            'theory_hnda': np.array([np.sqrt(3.0) / 4.0, -0.25, 0.0]),
            'theory_c': np.sqrt(3.0) / 8.0,
        },
    ]


def _run_analytical_tests(device, dtype):
    print('Analytical HNdA test cases')
    print()
    for case in _analytical_test_cases():
        e_ij = case['e_ij']
        l_ij = case['l_ij']
        l_jk = case['l_jk']
        l_ik = case['l_ik']
        theory_hnda = case['theory_hnda']

        scalar_hnda, _ = int_HNdC_ijk(e_ij, l_ij, l_jk, l_ik)
        vectorized_hnda, _ = int_HNdC_ijk_vectorized(
            e_ij, l_ij, l_jk, l_ik, device=device, dtype=dtype, return_numpy=True
        )

        scalar_hnda = np.asarray(scalar_hnda, dtype=np.float64)
        vectorized_hnda = np.asarray(vectorized_hnda, dtype=np.float64)

        theory_norm = np.linalg.norm(theory_hnda)
        if theory_norm == 0.0:
            scalar_hnda_rel = np.nan
            vectorized_hnda_rel = np.nan
        else:
            scalar_hnda_rel = 100.0 * np.linalg.norm(scalar_hnda - theory_hnda) / theory_norm
            vectorized_hnda_rel = 100.0 * np.linalg.norm(vectorized_hnda - theory_hnda) / theory_norm
        vectorized_vs_scalar_rel = 100.0 * np.linalg.norm(vectorized_hnda - scalar_hnda) / max(np.linalg.norm(scalar_hnda), 1e-30)

        _print_case_header(case['name'])
        print(f"scalar_hnda_rel_err_pct_vs_theory     : {scalar_hnda_rel:.6e}%")
        print(f"vectorized_hnda_rel_err_pct_vs_theory : {vectorized_hnda_rel:.6e}%")
        print(f"vectorized_hnda_rel_err_pct_vs_scalar : {vectorized_vs_scalar_rel:.6e}%")


def _orient_faces_outward(vertices, faces):
    faces = np.asarray(faces, dtype=np.int64).copy()
    for idx, tri in enumerate(faces):
        a, b, c = vertices[tri]
        normal = np.cross(b - a, c - a)
        centroid = (a + b + c) / 3.0
        if np.dot(normal, centroid) < 0.0:
            faces[idx] = np.array([tri[0], tri[2], tri[1]], dtype=np.int64)
    return faces


def _triangle_angle_stats(vertices, faces, tol=1e-12):
    min_angle_deg = np.inf
    max_angle_deg = 0.0
    num_obtuse_faces = 0
    num_right_faces = 0

    for tri in faces:
        p, q, r = vertices[np.asarray(tri, dtype=np.int64)]
        edge_sq = np.array(
            [
                np.dot(q - r, q - r),
                np.dot(p - r, p - r),
                np.dot(p - q, p - q),
            ],
            dtype=np.float64,
        )
        edge_sq_sorted = np.sort(edge_sq)
        if edge_sq_sorted[2] > edge_sq_sorted[0] + edge_sq_sorted[1] + tol:
            num_obtuse_faces += 1
        elif abs(edge_sq_sorted[2] - (edge_sq_sorted[0] + edge_sq_sorted[1])) <= tol:
            num_right_faces += 1

        corners = ((p, q, r), (q, r, p), (r, p, q))
        for a, b, c in corners:
            u = b - a
            v = c - a
            cos_angle = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-30)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cos_angle)))
            min_angle_deg = min(min_angle_deg, angle_deg)
            max_angle_deg = max(max_angle_deg, angle_deg)

    return {
        'min_angle_deg': float(min_angle_deg),
        'max_angle_deg': float(max_angle_deg),
        'num_obtuse_faces': int(num_obtuse_faces),
        'num_right_faces': int(num_right_faces),
        'all_non_obtuse': bool(num_obtuse_faces == 0),
    }


def _build_regular_icosahedron(radius=1.0):
    """Build an exactly equal-edged regular icosahedron inscribed in a sphere."""
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    vertices = np.array(
        [
            (-1.0,  phi, 0.0),
            ( 1.0,  phi, 0.0),
            (-1.0, -phi, 0.0),
            ( 1.0, -phi, 0.0),
            (0.0, -1.0,  phi),
            (0.0,  1.0,  phi),
            (0.0, -1.0, -phi),
            (0.0,  1.0, -phi),
            ( phi, 0.0, -1.0),
            ( phi, 0.0,  1.0),
            (-phi, 0.0, -1.0),
            (-phi, 0.0,  1.0),
        ],
        dtype=np.float64,
    )
    vertices = radius * vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )

    faces = _orient_faces_outward(vertices, faces)
    return vertices, faces


def _build_irregular_non_obtuse_icosahedral_sphere(radius=1.0, seed=0, tangential_amplitude=0.08):
    """
    Build a non-obtuse sphere mesh by tangentially perturbing an icosahedron
    and re-projecting every vertex back to the sphere.
    """
    vertices, faces = _build_regular_icosahedron(radius=1.0)
    rng = np.random.default_rng(seed)
    delta = rng.standard_normal(vertices.shape)

    radial = np.sum(delta * vertices, axis=1, keepdims=True) * vertices
    delta_tangent = delta - radial
    delta_tangent, tangent_norm = normalized(delta_tangent)
    if np.any(tangent_norm == 0.0):
        raise ValueError('Tangential perturbation produced a zero vector.')

    vertices_perturbed, _ = normalized(vertices + tangential_amplitude * delta_tangent)
    vertices_perturbed = radius * vertices_perturbed
    faces = _orient_faces_outward(vertices_perturbed, faces)

    angle_stats = _triangle_angle_stats(vertices_perturbed, faces)
    if angle_stats['num_obtuse_faces'] != 0:
        raise ValueError('The irregular icosahedral sphere builder created an obtuse face.')
    return vertices_perturbed, faces


def _build_symmetric_non_obtuse_irregular_sphere(radius=1.0):
    """
    Build a symmetric, unequal-edge, non-obtuse sphere mesh by subdividing a
    regular octahedron once and projecting edge midpoints to the sphere.

    Here, "symmetric" refers to the global mesh pattern, not to each triangle.
    The mesh keeps the rotation/reflection symmetry of the regular octahedron
    because every face is treated by the same midpoint-and-projection rule.
    This does NOT mean each triangle is equilateral (3 equal edges) or
    isosceles (2 equal edges). Triangles may still have three different edge
    lengths; the symmetry is in the whole mesh layout.
    """
    vertices = np.array(
        [
            [ 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0,  1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0,  1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
        ],
        dtype=np.int64,
    )

    midpoint_cache = {}
    vertices_list = [v.copy() for v in vertices]

    def midpoint_index(i, j):
        key = tuple(sorted((int(i), int(j))))
        if key not in midpoint_cache:
            midpoint = 0.5 * (vertices[key[0]] + vertices[key[1]])
            midpoint, _ = normalized(midpoint)
            midpoint_cache[key] = len(vertices_list)
            vertices_list.append(midpoint)
        return midpoint_cache[key]

    new_faces = []
    for tri in faces:
        i, j, k = map(int, tri)
        a = midpoint_index(i, j)
        b = midpoint_index(j, k)
        c = midpoint_index(k, i)
        new_faces.extend([[i, a, c], [a, j, b], [c, b, k], [a, b, c]])

    vertices = radius * np.asarray(vertices_list, dtype=np.float64)
    faces = _orient_faces_outward(vertices, np.asarray(new_faces, dtype=np.int64))

    angle_stats = _triangle_angle_stats(vertices, faces)
    if angle_stats['num_obtuse_faces'] != 0:
        raise ValueError('The symmetric non-obtuse sphere builder created an obtuse face.')
    return vertices, faces


def _build_symmetric_obtuse_octa_face_center_sphere(radius=1.0):
    """
    Build a globally symmetric unequal-edge sphere mesh by splitting every
    octahedron face with its projected face center.

    Here, "symmetric" refers to the global mesh pattern, not to each
    triangle. The mesh keeps the octahedral symmetry because every face is
    treated by the same face-center-and-projection rule. This does NOT mean
    each triangle is equilateral (3 equal edges) or isosceles (2 equal edges).
    Triangles may still have three different edge lengths; the symmetry is in
    the whole mesh layout.

    This particular mesh is intentionally obtuse.
    """
    vertices = np.array(
        [
            [ 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0,  1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0,  1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
        ],
        dtype=np.int64,
    )

    vertices_list = [v.copy() for v in vertices]
    new_faces = []
    for tri in faces:
        center = vertices[tri].sum(axis=0)
        center, _ = normalized(center)
        center_id = len(vertices_list)
        vertices_list.append(center)
        i, j, k = map(int, tri)
        new_faces.extend([[i, j, center_id], [j, k, center_id], [k, i, center_id]])

    vertices = radius * np.asarray(vertices_list, dtype=np.float64)
    faces = _orient_faces_outward(vertices, np.asarray(new_faces, dtype=np.int64))

    angle_stats = _triangle_angle_stats(vertices, faces)
    if angle_stats['num_obtuse_faces'] == 0:
        raise ValueError('The symmetric obtuse sphere builder did not create an obtuse face.')
    return vertices, faces


def _edge_length_stats(vertices, faces):
    edges = set()
    for tri in faces:
        i, j, k = map(int, tri)
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((k, i))))
    edge_lengths = np.array([np.linalg.norm(vertices[i] - vertices[j]) for i, j in sorted(edges)], dtype=np.float64)
    mean_len = float(np.mean(edge_lengths))
    rel_spread_pct = 100.0 * (float(np.max(edge_lengths)) - float(np.min(edge_lengths))) / max(mean_len, 1e-30)
    return {
        'num_edges': int(edge_lengths.size),
        'edge_length_min': float(np.min(edge_lengths)),
        'edge_length_max': float(np.max(edge_lengths)),
        'edge_length_mean': mean_len,
        'edge_length_rel_spread_pct': rel_spread_pct,
    }


def _collect_directed_contributions(vertices, faces, flip_e_ij=True):
    rows = []
    for tri in faces:
        i, j, k = map(int, tri)
        directed = [(i, j, k), (j, k, i), (k, i, j), (j, i, k), (k, j, i), (i, k, j)]
        for a, b, c in directed:
            e_ij = vertices[b] - vertices[a]
            if flip_e_ij:
                e_ij = -e_ij
            l_ij = np.linalg.norm(e_ij)
            l_jk = np.linalg.norm(vertices[c] - vertices[b])
            l_ik = np.linalg.norm(vertices[c] - vertices[a])
            rows.append((a, e_ij, l_ij, l_jk, l_ik))

    vertex_ids = np.array([r[0] for r in rows], dtype=np.int64)
    e_ij = np.array([r[1] for r in rows], dtype=np.float64)
    l_ij = np.array([r[2] for r in rows], dtype=np.float64)
    l_jk = np.array([r[3] for r in rows], dtype=np.float64)
    l_ik = np.array([r[4] for r in rows], dtype=np.float64)
    return vertex_ids, e_ij, l_ij, l_jk, l_ik


def _assemble_scalar(vertex_ids, e_ij, l_ij, l_jk, l_ik, num_vertices):
    H = np.zeros((num_vertices, 3), dtype=np.float64)
    C = np.zeros(num_vertices, dtype=np.float64)
    t0 = time.perf_counter()
    for idx in range(vertex_ids.size):
        hnda_ijk, c_ijk = int_HNdC_ijk(e_ij[idx], l_ij[idx], l_jk[idx], l_ik[idx])
        vid = vertex_ids[idx]
        H[vid] += hnda_ijk
        C[vid] += c_ijk
    elapsed = time.perf_counter() - t0
    return H, C, elapsed


def _assemble_vectorized(vertex_ids, e_ij, l_ij, l_jk, l_ik, num_vertices, device, dtype, repeats=5):
    if torch is None:
        raise ImportError('PyTorch is not installed, so vectorized assembly cannot be tested.')

    device = torch.device(device)
    vertex_ids_t = torch.as_tensor(vertex_ids, dtype=torch.int64, device=device)
    e_ij_t = torch.as_tensor(e_ij, dtype=dtype, device=device)
    l_ij_t = torch.as_tensor(l_ij, dtype=dtype, device=device)
    l_jk_t = torch.as_tensor(l_jk, dtype=dtype, device=device)
    l_ik_t = torch.as_tensor(l_ik, dtype=dtype, device=device)

    original_num_threads = torch.get_num_threads()
    vectorized_timings = []
    try:
        if device.type == 'cpu':
            torch.set_num_threads(1)

        hnda_batch, c_batch = int_HNdC_ijk_vectorized(
            e_ij_t, l_ij_t, l_jk_t, l_ik_t, device=device, dtype=dtype
        )
        H = torch.zeros((num_vertices, 3), dtype=dtype, device=device)
        C = torch.zeros(num_vertices, dtype=dtype, device=device)
        H.index_add_(0, vertex_ids_t, hnda_batch)
        C.index_add_(0, vertex_ids_t, c_batch)
        _synchronize_device(device)

        for _ in range(repeats):
            _synchronize_device(device)
            t0 = time.perf_counter()
            hnda_batch, c_batch = int_HNdC_ijk_vectorized(
                e_ij_t, l_ij_t, l_jk_t, l_ik_t, device=device, dtype=dtype
            )
            H = torch.zeros((num_vertices, 3), dtype=dtype, device=device)
            C = torch.zeros(num_vertices, dtype=dtype, device=device)
            H.index_add_(0, vertex_ids_t, hnda_batch)
            C.index_add_(0, vertex_ids_t, c_batch)
            _synchronize_device(device)
            vectorized_timings.append(time.perf_counter() - t0)
    finally:
        if device.type == 'cpu':
            torch.set_num_threads(original_num_threads)

    return H.detach().cpu().numpy(), C.detach().cpu().numpy(), min(vectorized_timings), vectorized_timings


def _sphere_metrics(vertices, H, C, radius):
    rhat, _ = normalized(vertices)
    H_norm = np.linalg.norm(H, axis=1)
    alignment = np.sum(H * rhat, axis=1) / np.maximum(H_norm, 1e-30)
    signed_curvature = np.sum(H * rhat, axis=1) / np.maximum(C, 1e-30)
    abs_curvature = H_norm / np.maximum(C, 1e-30)

    H_theory = (2.0 / radius) * C[:, None] * rhat
    H_theory_rel_pct = _vector_norm_rel_err_pct(H, H_theory)

    # Under constant surface tension sigma, Fs = sigma * HNdA.
    # We report sigma = 1 here, so Fs_i = HNdA_i and the relative error matches.
    Fs = H.copy()
    Fs_theory = H_theory.copy()
    Fs_theory_rel_pct = H_theory_rel_pct.copy()

    return {
        'alignment_mean': float(np.mean(alignment)),
        'alignment_min': float(np.min(alignment)),
        'alignment_max': float(np.max(alignment)),
        'signed_curvature_mean': float(np.mean(signed_curvature)),
        'signed_curvature_std': float(np.std(signed_curvature)),
        'abs_curvature_mean': float(np.mean(abs_curvature)),
        'abs_curvature_std': float(np.std(abs_curvature)),
        'abs_curv_mean_abs_err_vs_2_over_R': float(abs(np.mean(abs_curvature) - 2.0 / radius)),
        'hnda_theory_max_abs_diff': _max_abs_diff(H, H_theory),
        'hnda_theory_max_rel_err_pct': float(np.max(H_theory_rel_pct)),
        'hnda_theory_mean_rel_err_pct': float(np.mean(H_theory_rel_pct)),
        'Fs_theory_max_abs_diff': _max_abs_diff(Fs, Fs_theory),
        'Fs_theory_max_rel_err_pct': float(np.max(Fs_theory_rel_pct)),
        'Fs_theory_mean_rel_err_pct': float(np.mean(Fs_theory_rel_pct)),
    }


def _run_sphere_mesh_test(vertices, faces, mesh_type, device, dtype, radius=1.0, repeats=5):
    edge_stats = _edge_length_stats(vertices, faces)
    angle_stats = _triangle_angle_stats(vertices, faces)
    vertex_ids, e_ij, l_ij, l_jk, l_ik = _collect_directed_contributions(vertices, faces, flip_e_ij=True)

    H_scalar, C_scalar, t_scalar = _assemble_scalar(vertex_ids, e_ij, l_ij, l_jk, l_ik, len(vertices))
    H_vectorized, C_vectorized, t_vectorized, vectorized_timings = _assemble_vectorized(
        vertex_ids, e_ij, l_ij, l_jk, l_ik, len(vertices), device=device, dtype=dtype, repeats=repeats
    )

    scalar_metrics = _sphere_metrics(vertices, H_scalar, C_scalar, radius)
    vectorized_metrics = _sphere_metrics(vertices, H_vectorized, C_vectorized, radius)

    return {
        'mesh_type': mesh_type,
        'radius': radius,
        'vertices': int(vertices.shape[0]),
        'faces': int(faces.shape[0]),
        'directed_contributions': int(vertex_ids.size),
        'theory_abs_mean_curvature': 2.0 / radius,
        'scalar_time_s': t_scalar,
        'vectorized_time_s': t_vectorized,
        'speedup': (t_scalar / t_vectorized) if t_vectorized > 0.0 else np.inf,
        'vectorized_device': str(torch.device(device)),
        'vectorized_dtype': str(dtype).replace('torch.', ''),
        'vectorized_timings_s': vectorized_timings,
        'edge_stats': edge_stats,
        'angle_stats': angle_stats,
        'hnda_max_abs_diff': _max_abs_diff(H_scalar, H_vectorized),
        'hnda_max_vector_norm_diff': _max_vector_norm_diff(H_scalar, H_vectorized),
        'c_max_abs_diff': _max_abs_diff(C_scalar, C_vectorized),
        'hnda_max_rel_err_pct': float(np.max(_vector_norm_rel_err_pct(H_vectorized, H_scalar))),
        'c_max_rel_err_pct': _max_rel_err_pct(C_vectorized, C_scalar),
        'scalar_metrics': scalar_metrics,
        'vectorized_metrics': vectorized_metrics,
        'sample_vertex_index': 0,
        'sample_vertex_position': vertices[0],
        'sample_scalar_hnda': H_scalar[0],
        'sample_vectorized_hnda': H_vectorized[0],
        'sample_scalar_c': float(C_scalar[0]),
        'sample_vectorized_c': float(C_vectorized[0]),
    }


def _run_equal_edged_sphere_test(device, dtype, radius=1.0, repeats=5):
    vertices, faces = _build_regular_icosahedron(radius=radius)
    return _run_sphere_mesh_test(
        vertices, faces, 'regular_icosahedron_exact_equal_edge',
        device=device, dtype=dtype, radius=radius, repeats=repeats,
    )


def _run_irregular_non_obtuse_sphere_test(device, dtype, radius=1.0, repeats=5):
    vertices, faces = _build_irregular_non_obtuse_icosahedral_sphere(radius=radius)
    return _run_sphere_mesh_test(
        vertices, faces, 'irregular_icosahedron_unequal_edge_non_obtuse',
        device=device, dtype=dtype, radius=radius, repeats=repeats,
    )


def _run_symmetric_non_obtuse_irregular_sphere_test(device, dtype, radius=1.0, repeats=5):
    vertices, faces = _build_symmetric_non_obtuse_irregular_sphere(radius=radius)
    return _run_sphere_mesh_test(
        vertices, faces, 'symmetric_irregular_octa_subdiv1_unequal_edge_non_obtuse',
        device=device, dtype=dtype, radius=radius, repeats=repeats,
    )


def _run_symmetric_obtuse_sphere_test(device, dtype, radius=1.0, repeats=5):
    vertices, faces = _build_symmetric_obtuse_octa_face_center_sphere(radius=radius)
    return _run_sphere_mesh_test(
        vertices, faces, 'symmetric_irregular_octa_face_center_unequal_edge_obtuse',
        device=device, dtype=dtype, radius=radius, repeats=repeats,
    )


def _benchmark_int_HNdC_ijk(device, dtype, num_samples=200000, seed=12345, repeats=5):
    rng = np.random.default_rng(seed)

    p_i = rng.standard_normal((num_samples, 3))
    p_j = rng.standard_normal((num_samples, 3))
    p_k = rng.standard_normal((num_samples, 3))

    e_ij = p_j - p_i
    e_jk = p_k - p_j
    e_ik = p_k - p_i

    l_ij = np.linalg.norm(e_ij, axis=1)
    l_jk = np.linalg.norm(e_jk, axis=1)
    l_ik = np.linalg.norm(e_ik, axis=1)

    hnda_ref = np.empty_like(e_ij)
    c_ref = np.empty(num_samples, dtype=np.float64)

    t0 = time.perf_counter()
    for idx in range(num_samples):
        hnda_ref[idx], c_ref[idx] = int_HNdC_ijk(e_ij[idx], l_ij[idx], l_jk[idx], l_ik[idx])
    t_scalar = time.perf_counter() - t0

    device = torch.device(device)
    original_num_threads = torch.get_num_threads()
    try:
        if device.type == 'cpu':
            torch.set_num_threads(1)

        e_ij_t = torch.as_tensor(e_ij, dtype=dtype, device=device)
        l_ij_t = torch.as_tensor(l_ij, dtype=dtype, device=device)
        l_jk_t = torch.as_tensor(l_jk, dtype=dtype, device=device)
        l_ik_t = torch.as_tensor(l_ik, dtype=dtype, device=device)

        int_HNdC_ijk_vectorized(e_ij_t, l_ij_t, l_jk_t, l_ik_t, device=device, dtype=dtype)
        _synchronize_device(device)

        vectorized_timings = []
        for _ in range(repeats):
            _synchronize_device(device)
            t1 = time.perf_counter()
            hnda_vec_t, c_vec_t = int_HNdC_ijk_vectorized(
                e_ij_t, l_ij_t, l_jk_t, l_ik_t, device=device, dtype=dtype
            )
            _synchronize_device(device)
            vectorized_timings.append(time.perf_counter() - t1)
    finally:
        if device.type == 'cpu':
            torch.set_num_threads(original_num_threads)

    t_vectorized = min(vectorized_timings)
    hnda_vec = hnda_vec_t.detach().cpu().numpy()
    c_vec = c_vec_t.detach().cpu().numpy()

    return {
        'num_samples': num_samples,
        'device': str(device),
        'dtype': str(dtype).replace('torch.', ''),
        'scalar_time_s': t_scalar,
        'vectorized_time_s': t_vectorized,
        'speedup': t_scalar / t_vectorized if t_vectorized > 0.0 else np.inf,
        'vectorized_device': str(torch.device(device)),
        'vectorized_dtype': str(dtype).replace('torch.', ''),
        'vectorized_timings_s': vectorized_timings,
        'hnda_max_abs_diff': _max_abs_diff(hnda_ref, hnda_vec),
        'c_max_abs_diff': _max_abs_diff(c_ref, c_vec),
        'hnda_max_rel_err_pct': float(np.max(_vector_norm_rel_err_pct(hnda_vec, hnda_ref))),
        'c_max_rel_err_pct': _max_rel_err_pct(c_vec, c_ref),
        'sample_scalar_hnda': hnda_ref[0],
        'sample_vectorized_hnda': hnda_vec[0],
        'sample_hnda_diff': hnda_ref[0] - hnda_vec[0],
        'sample_scalar_c': float(c_ref[0]),
        'sample_vectorized_c': float(c_vec[0]),
        'sample_c_diff': float(c_ref[0] - c_vec[0]),
    }


def main():
    print('Testing HNdA and Fs with scalar and vectorized PyTorch')

    if torch is None:
        raise ImportError('PyTorch is not installed, so int_HNdC_ijk_vectorized cannot be tested.')

    device, dtype = _select_vectorized_device_and_dtype()
    print(f'vectorized_device                         : {device}')
    print(f'vectorized_dtype                          : {str(dtype).replace("torch.", "")}')

    print()
    _run_analytical_tests(device=device, dtype=dtype)

    sphere_tests = [
        _run_equal_edged_sphere_test(device=device, dtype=dtype, radius=1.0, repeats=5),
        _run_irregular_non_obtuse_sphere_test(device=device, dtype=dtype, radius=1.0, repeats=5),
        _run_symmetric_non_obtuse_irregular_sphere_test(device=device, dtype=dtype, radius=1.0, repeats=5),
        _run_symmetric_obtuse_sphere_test(device=device, dtype=dtype, radius=1.0, repeats=5),
    ]
    for sphere in sphere_tests:
        print()
        print('Sphere mesh assembled test')
        print(f"mesh_type                                : {sphere['mesh_type']}")
        print(f"vertices                                 : {sphere['vertices']}")
        print(f"faces                                    : {sphere['faces']}")
        print(f"edge_length_rel_spread_pct               : {sphere['edge_stats']['edge_length_rel_spread_pct']:.6f}%")
        print(f"triangle_max_angle_deg                   : {sphere['angle_stats']['max_angle_deg']:.6f}")
        print(f"triangle_num_obtuse_faces                : {sphere['angle_stats']['num_obtuse_faces']}")
        print(f"vectorized_device                        : {sphere['vectorized_device']}")
        print(f"vectorized_dtype                         : {sphere['vectorized_dtype']}")
        print(f"scalar_time_s                            : {sphere['scalar_time_s']:.6f}")
        print(f"vectorized_time_s                        : {sphere['vectorized_time_s']:.6f}")
        print(f"speedup                                  : {sphere['speedup']:.2f}x")
        print(f"scalar_hnda_max_rel_err_pct_vs_theory    : {sphere['scalar_metrics']['hnda_theory_max_rel_err_pct']:.6e}%")
        print(f"scalar_hnda_mean_rel_err_pct_vs_theory   : {sphere['scalar_metrics']['hnda_theory_mean_rel_err_pct']:.6e}%")
        print(f"vectorized_hnda_max_rel_err_pct_vs_theory: {sphere['vectorized_metrics']['hnda_theory_max_rel_err_pct']:.6e}%")
        print(f"vectorized_hnda_mean_rel_err_pct_vs_theory: {sphere['vectorized_metrics']['hnda_theory_mean_rel_err_pct']:.6e}%")
        print(f"vectorized_hnda_rel_err_pct_vs_scalar    : {sphere['hnda_max_rel_err_pct']:.6e}%")
        print(f"scalar_Fs_max_rel_err_pct_vs_theory      : {sphere['scalar_metrics']['Fs_theory_max_rel_err_pct']:.6e}%")
        print(f"scalar_Fs_mean_rel_err_pct_vs_theory     : {sphere['scalar_metrics']['Fs_theory_mean_rel_err_pct']:.6e}%")
        print(f"vectorized_Fs_max_rel_err_pct_vs_theory  : {sphere['vectorized_metrics']['Fs_theory_max_rel_err_pct']:.6e}%")
        print(f"vectorized_Fs_mean_rel_err_pct_vs_theory : {sphere['vectorized_metrics']['Fs_theory_mean_rel_err_pct']:.6e}%")
        print(f"vectorized_Fs_rel_err_pct_vs_scalar      : {sphere['hnda_max_rel_err_pct']:.6e}%")

    print()
    print('Sphere _err_pct_vs_theory summary')
    header = (
        f"{'mesh_type':<52} | {'s_hnda_max%':>12} | {'s_hnda_mean%':>13} | "
        f"{'v_hnda_max%':>12} | {'v_hnda_mean%':>13} | {'s_Fs_max%':>10} | "
        f"{'s_Fs_mean%':>11} | {'v_Fs_max%':>10} | {'v_Fs_mean%':>11}"
    )
    print(header)
    print('-' * len(header))
    for sphere in sphere_tests:
        print(
            f"{sphere['mesh_type']:<52} | "
            f"{sphere['scalar_metrics']['hnda_theory_max_rel_err_pct']:12.6e} | "
            f"{sphere['scalar_metrics']['hnda_theory_mean_rel_err_pct']:13.6e} | "
            f"{sphere['vectorized_metrics']['hnda_theory_max_rel_err_pct']:12.6e} | "
            f"{sphere['vectorized_metrics']['hnda_theory_mean_rel_err_pct']:13.6e} | "
            f"{sphere['scalar_metrics']['Fs_theory_max_rel_err_pct']:10.6e} | "
            f"{sphere['scalar_metrics']['Fs_theory_mean_rel_err_pct']:11.6e} | "
            f"{sphere['vectorized_metrics']['Fs_theory_max_rel_err_pct']:10.6e} | "
            f"{sphere['vectorized_metrics']['Fs_theory_mean_rel_err_pct']:11.6e}"
        )

    bench = _benchmark_int_HNdC_ijk(device=device, dtype=dtype)
    print()
    print('Random-triangle HNdA benchmark')
    print(f"vectorized_device                        : {bench['device']}")
    print(f"vectorized_dtype                         : {bench['dtype']}")
    print(f"scalar_time_s                            : {bench['scalar_time_s']:.6f}")
    print(f"vectorized_time_s                        : {bench['vectorized_time_s']:.6f}")
    print(f"speedup                                  : {bench['speedup']:.2f}x")
    print(f"vectorized_hnda_rel_err_pct_vs_scalar    : {bench['hnda_max_rel_err_pct']:.6e}%")


if __name__ == '__main__':
    main()
