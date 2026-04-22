"""
This file uses the Heron's formula to compute the curvature which is a much faster
routine than the experimental code.

Note, however, that this might not be latest version. It is taken from the notebook
"Sphere area study p ix [new Dec 2023].ipynb" which might not be the latest version
which was actually validated (which is in lsm?)

"""
import numpy as np


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
    # Sort the list, python sorts from the smallest to largest element:
    lengths.sort()
    # We must have use a ≥ b ≥ c in floating-point stable Heron's formula:
    a = lengths[2]
    b = lengths[1]
    c = lengths[0]
    A = (1 / 4.0) * np.sqrt((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c)))
    # Dual weights (scalar):
    w_ij = (1 / 8.0) * (l_jk ** 2 + l_ik ** 2 - l_ij ** 2) / A  # w_ij = abs(w_ij)

    # Mean normal curvature (1x3 vector):
    hnda_ijk = w_ij * e_ij  # curvature from this edge jk in triangle ijk with w_jk = 1/2 cot(theta_i^jk)

    # Dual areas
    h_ij = 0.5 * l_ij
    b_ij = abs(w_ij) * l_ij  # = ||0.5 cot(theta_i^jk)|| * 0.5*l_ij
    c_ijk = 0.5 * b_ij * h_ij
    return hnda_ijk, c_ijk


def A_i(v, n_i=None):
    """
    Compute the discrete normal area of vertex v_i

    :param v: vertex object
    :return: HNdA_i: the curvature tensor at input vertex v
             c_i:  the dual area of the vertex
    """
    # NOTE: THIS MUST BE REPLACED WITH THE LEVEL SET PLANE VECTOR:
    if n_i is not None:
        n_i = v.x

    # Initiate
    NdA_i = np.zeros(3)  # np.zeros([len(v.nn), 3])  # Mean normal curvature
    C_i = 0.0  # np.zeros([len(v.nn), 3])  # Dual area around edge in a surface
    vi = v
    for vj in v.nn:
        # Compute the intersection set of vertices i and j:
        e_i_int_e_j = vi.nn.intersection(vj.nn)  # Set of size 1 or 2
        e_ij = vj.x_a - vi.x_a  # Compute edge ij (1x3 vector)
        e_ij = - e_ij  # WHY???
        vk = list(e_i_int_e_j)[0]  # index in triangle ijk  # NOTE: k = vk.index
        e_ik = vk.x_a - vi.x_a  # Compute edge ik (1x3 vector)

        # NOTE: The code below results in the INCORRECT values unless we set
        #      e_ij = - e_ij  # WHY???
        if 1:
            # Discrete vector area:
            # Simplex areas of ijk and normals
            wedge_ij_ik = np.cross(e_ij, e_ik)
            # If the wrong direction was chosen, choose the other:
            #  print(f'np.dot(normalized(wedge_ij_ik)[0], n_i) = {np.dot(normalized(wedge_ij_ik)[0], n_i)}')
            if np.dot(normalized(wedge_ij_ik)[0], n_i) < 0:
                print(f'e_ij_prev = {e_ij}')
                e_ij = vi.x_a - vj.x_a
                # e_ij = vi.x_a - vj.x_a
            #  e_ij = vj.x_a - vi.x_a  # Does not appear to be needed,
            #                          # but more tests need to be done

        if len(e_i_int_e_j) == 1:  # boundary edge
            pass  # ignore for now

        else:  # len(e_i_int_e_j) == 2 mathematically guaranteed:
            vl = list(e_i_int_e_j)[1]  # index in triangle ijl
            # Compute dual for contact angle alpha
            e_jk = vk.x_a - vj.x_a
            # wedge_ij_ik = np.cross(e_ij, e_ik)
            # Find lengths (norm of the edge vectors):
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)

            # Contact angle beta
            e_il = vl.x_a - vi.x_a
            e_jl = vl.x_a - vj.x_a
            l_il = np.linalg.norm(e_il)  # NOTE: l_ki = l_ik
            l_jl = np.linalg.norm(e_jl)
            hnda_ijl, c_ijl = HNdC_ijk(e_ij, l_ij, l_jl, l_il)

            # Save results
            HNdA_i += hnda_ijk
            HNdA_i += hnda_ijl
            C_i += c_ijl
            C_i += c_ijk

    return NdA_i  # , C_i


# TODO: Since sparse arrays are too expensive to recreate and add to,
#      we might want cache edge lengths instead. higher dimensional
#      simplices could be done with a lexigraphic cache.
#      This is simple to parallelise on CPUs, but might be much harder
#      to do on GPUs.

def hndA_i(v, n_i=None):
    """
    Compute the mean normal curvature of vertex

    :param v: vertex object
    :return: HNdA_i: the curvature tensor at input vertex v
             c_i:  the dual area of the vertex
    """
    # NOTE: THIS MUST BE REPLACED WITH THE LEVEL SET PLANE VECTOR:
    if n_i is not None:
        n_i = v.x

    # Initiate
    HNdA_i = np.zeros(3)  # np.zeros([len(v.nn), 3])  # Mean normal curvature
    C_i = 0.0  # np.zeros([len(v.nn), 3])  # Dual area around edge in a surface
    vi = v
    for vj in v.nn:
        # Compute the intersection set of vertices i and j:
        e_i_int_e_j = vi.nn.intersection(vj.nn)  # Set of size 1 or 2
        e_ij = vj.x_a - vi.x_a  # Compute edge ij (1x3 vector)
        e_ij = - e_ij  # WHY???
        vk = list(e_i_int_e_j)[0]  # index in triangle ijk  # NOTE: k = vk.index
        e_ik = vk.x_a - vi.x_a  # Compute edge ik (1x3 vector)

        # NOTE: The code below results in the INCORRECT values unless we set
        #      e_ij = - e_ij  # WHY???
        if 0:
            # Discrete vector area:
            # Simplex areas of ijk and normals
            wedge_ij_ik = np.cross(e_ij, e_ik)
            # If the wrong direction was chosen, choose the other:
            #  print(f'np.dot(normalized(wedge_ij_ik)[0], n_i) = {np.dot(normalized(wedge_ij_ik)[0], n_i)}')
            if np.dot(normalized(wedge_ij_ik)[0], n_i) < 0:
                e_ij = vi.x_a - vj.x_a
                # e_ij = vi.x_a - vj.x_a
            #  e_ij = vj.x_a - vi.x_a  # Does not appear to be needed,
            #                          # but more tests need to be done

        if len(e_i_int_e_j) == 1:  # boundary edge
            vk = list(e_i_int_e_j)[0]  # Boundary edge index
            # Compute edges in triangle ijk
            e_ik = vk.x_a - vi.x_a
            e_jk = vk.x_a - vj.x_a
            # Find lengths (norm of the edge vectors):
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)

            # Save results
            HNdA_i += hnda_ijk
            C_i += c_ijk

        else:  # len(e_i_int_e_j) == 2 mathematically guaranteed:
            vl = list(e_i_int_e_j)[1]  # index in triangle ijl
            # Compute dual for contact angle alpha
            e_jk = vk.x_a - vj.x_a
            # wedge_ij_ik = np.cross(e_ij, e_ik)
            # Find lengths (norm of the edge vectors):
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)

            # Contact angle beta
            e_il = vl.x_a - vi.x_a
            e_jl = vl.x_a - vj.x_a
            l_il = np.linalg.norm(e_il)  # NOTE: l_ki = l_ik
            l_jl = np.linalg.norm(e_jl)
            hnda_ijl, c_ijl = HNdC_ijk(e_ij, l_ij, l_jl, l_il)

            # Save results
            HNdA_i += hnda_ijk
            HNdA_i += hnda_ijl
            C_i += c_ijl
            C_i += c_ijk

    return HNdA_i, C_i


def _pad3(x_a):
    """Pad a coordinate array to 3D (for np.cross compatibility)."""
    if len(x_a) >= 3:
        return x_a[:3]
    out = np.zeros(3)
    out[:len(x_a)] = x_a
    return out


def hndA_i_interface(v, interface_set, n_i=None):
    """Mean curvature normal restricted to interface sub-mesh.

    Same algorithm as :func:`hndA_i` but only considers neighbours that are
    in ``interface_set``.  Used by multiphase surface tension to compute
    the curvature of the phase boundary rather than the full mesh.

    Works in both 2D and 3D by padding 2D vectors to 3D so that
    ``np.cross`` (used inside ``HNdC_ijk``) operates correctly.

    Parameters
    ----------
    v : vertex object
        Must have ``v.x_a`` and ``v.nn`` populated.
    interface_set : set
        Set of vertex objects forming the interface sub-mesh.
    n_i : ignored
        Kept for API compatibility.

    Returns
    -------
    HNdA_i : ndarray, shape (3,)
        Integrated mean curvature normal vector.
    C_i : float
        Dual area of the vertex on the interface.
    """
    HNdA_i = np.zeros(3)
    C_i = 0.0
    vi = v
    for vj in v.nn:
        if vj not in interface_set:
            continue
        e_i_int_e_j = vi.nn.intersection(vj.nn).intersection(interface_set)
        if len(e_i_int_e_j) == 0:
            continue
        e_ij = _pad3(vj.x_a) - _pad3(vi.x_a)
        e_ij = -e_ij  # Sign convention (matches hndA_i)

        if len(e_i_int_e_j) == 1:
            vk = list(e_i_int_e_j)[0]
            e_ik = _pad3(vk.x_a) - _pad3(vi.x_a)
            e_jk = _pad3(vk.x_a) - _pad3(vj.x_a)
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)
            HNdA_i += hnda_ijk
            C_i += c_ijk
        else:
            vk, vl = list(e_i_int_e_j)[:2]
            e_ik = _pad3(vk.x_a) - _pad3(vi.x_a)
            e_jk = _pad3(vk.x_a) - _pad3(vj.x_a)
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)

            e_il = _pad3(vl.x_a) - _pad3(vi.x_a)
            e_jl = _pad3(vl.x_a) - _pad3(vj.x_a)
            l_il = np.linalg.norm(e_il)
            l_jl = np.linalg.norm(e_jl)
            hnda_ijl, c_ijl = HNdC_ijk(e_ij, l_ij, l_jl, l_il)

            HNdA_i += hnda_ijk
            HNdA_i += hnda_ijl
            C_i += c_ijk
            C_i += c_ijl

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
    # Sort the list, python sorts from the smallest to largest element:
    lengths.sort()
    # We must have use a ≥ b ≥ c in floating-point stable Heron's formula:
    a = lengths[2]
    b = lengths[1]
    c = lengths[0]
    A = (1 / 4.0) * np.sqrt((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c)))
    # Dual weights (scalar):
    w_ij = (1 / 8.0) * (l_jk ** 2 + l_ik ** 2 - l_ij ** 2) / A  # w_ij = abs(w_ij)

    # Mean normal curvature (1x3 vector):
    hnda_ijk = w_ij * e_ij  # curvature from this edge jk in triangle ijk with w_jk = 1/2 cot(theta_i^jk)

    # Dual areas
    h_ij = 0.5 * l_ij
    b_ij = abs(w_ij) * l_ij  # = ||0.5 cot(theta_i^jk)|| * 0.5*l_ij
    c_ijk = 0.5 * b_ij * h_ij
    return hnda_ijk, c_ijk


def int_hndA_i(v, n_i=None):
    """
    Compute the mean normal curvature of vertex

    :param v: vertex object
    :return: HNdA_i: the curvature tensor at input vertex v
             c_i:  the dual area of the vertex
    """
    # NOTE: THIS MUST BE REPLACED WITH THE LEVEL SET PLANE VECTOR:
    if n_i is not None:
        n_i = v.x

    # Initiate
    HNdA_i = np.zeros(3)  # np.zeros([len(v.nn), 3])  # Mean normal curvature
    C_i = 0.0  # np.zeros([len(v.nn), 3])  # Dual area around edge in a surface
    vi = v
    for vj in v.nn:
        # Compute the intersection set of vertices i and j:
        e_i_int_e_j = vi.nn.intersection(vj.nn)  # Set of size 1 or 2
        e_ij = vj.x_a - vi.x_a  # Compute edge ij (1x3 vector)
        e_ij = - e_ij  # WHY???
        vk = list(e_i_int_e_j)[0]  # index in triangle ijk  # NOTE: k = vk.index
        e_ik = vk.x_a - vi.x_a  # Compute edge ik (1x3 vector)

        if len(e_i_int_e_j) == 1:  # boundary edge
            vk = list(e_i_int_e_j)[0]  # Boundary edge index
            # Compute edges in triangle ijk
            e_ik = vk.x_a - vi.x_a
            e_jk = vk.x_a - vj.x_a
            # Find lengths (norm of the edge vectors):
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
            l_jk = np.linalg.norm(e_jk)

            e_hnda_ij = vj.hnda_i - vi.hnda_i
            hnda_ijk, c_ijk = int_HNdC_ijk(e_hnda_ij, l_ij, l_jk, l_ik)

            # Save results
            HNdA_i += hnda_ijk
            C_i += c_ijk

        else:  # len(e_i_int_e_j) == 2 mathematically guaranteed:
            vl = list(e_i_int_e_j)[1]  # index in triangle ijl
            # Compute dual for contact angle alpha
            e_jk = vk.x_a - vj.x_a
            # wedge_ij_ik = np.cross(e_ij, e_ik)
            # Find lengths (norm of the edge vectors):
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
            l_jk = np.linalg.norm(e_jk)
            e_hnda_ij = vj.hnda_i - vi.hnda_i
            hnda_ijk, c_ijk = int_HNdC_ijk(e_hnda_ij, l_ij, l_jk, l_ik)

            # Contact angle beta
            e_il = vl.x_a - vi.x_a
            e_jl = vl.x_a - vj.x_a
            l_il = np.linalg.norm(e_il)  # NOTE: l_ki = l_ik
            l_jl = np.linalg.norm(e_jl)
            e_hnda_ij = vj.hnda_i - vi.hnda_i
            hnda_ijl, c_ijl = int_HNdC_ijk(e_hnda_ij, l_ij, l_jk, l_ik)

            # Save results
            HNdA_i += hnda_ijk
            HNdA_i += hnda_ijl
            C_i += c_ijl
            C_i += c_ijk

    return HNdA_i, C_i


try:
    import torch as _torch
except ImportError:
    _torch = None


def HNdC_ijk_batch(e_ij, l_ij, l_jk, l_ik):
    """
    Vectorized NumPy version of :func:`HNdC_ijk`.

    Parameters
    ----------
    e_ij : ndarray, shape (N, 3)
        Edge vectors.
    l_ij, l_jk, l_ik : ndarray, shape (N,)
        Edge lengths.

    Returns
    -------
    hnda_ijk : ndarray, shape (N, 3)
        Curvature vectors.
    c_ijk : ndarray, shape (N,)
        Dual areas.
    """
    lengths = np.stack((l_ij, l_jk, l_ik), axis=-1)
    lengths_sorted = np.sort(lengths, axis=-1)
    c = lengths_sorted[..., 0]
    b = lengths_sorted[..., 1]
    a = lengths_sorted[..., 2]

    heron_term = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    A = 0.25 * np.sqrt(heron_term)

    w_ij = 0.125 * (l_jk ** 2 + l_ik ** 2 - l_ij ** 2) / A
    hnda_ijk = w_ij[:, np.newaxis] * e_ij

    h_ij = 0.5 * l_ij
    b_ij = np.abs(w_ij) * l_ij
    c_ijk = 0.5 * b_ij * h_ij
    return hnda_ijk, c_ijk


def HNdC_ijk_batch_torch(e_ij, l_ij, l_jk, l_ik, device=None):
    """
    Vectorized PyTorch version of :func:`HNdC_ijk`.

    Accepts numpy arrays or torch tensors.  Returns torch tensors on the
    given *device* (defaults to the device selected by the hyperct
    ``TorchBackend``, or CPU).

    Parameters
    ----------
    e_ij : array_like, shape (N, 3)
        Edge vectors.
    l_ij, l_jk, l_ik : array_like, shape (N,)
        Edge lengths.
    device : str or torch.device, optional
        PyTorch device.  Defaults to ``'cpu'``.

    Returns
    -------
    hnda_ijk : Tensor, shape (N, 3)
        Curvature vectors.
    c_ijk : Tensor, shape (N,)
        Dual areas.
    """
    if _torch is None:
        raise ImportError("PyTorch is required for HNdC_ijk_batch_torch.")

    if device is None:
        device = _torch.device('cpu')
    else:
        device = _torch.device(device)

    dtype = _torch.float64

    e_ij_t = _torch.as_tensor(e_ij, dtype=dtype, device=device)
    l_ij_t = _torch.as_tensor(l_ij, dtype=dtype, device=device)
    l_jk_t = _torch.as_tensor(l_jk, dtype=dtype, device=device)
    l_ik_t = _torch.as_tensor(l_ik, dtype=dtype, device=device)

    lengths = _torch.stack((l_ij_t, l_jk_t, l_ik_t), dim=-1)
    lengths_sorted, _ = _torch.sort(lengths, dim=-1)

    c = lengths_sorted[..., 0]
    b = lengths_sorted[..., 1]
    a = lengths_sorted[..., 2]

    heron_term = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    A = 0.25 * _torch.sqrt(heron_term)

    w_ij = 0.125 * (l_jk_t ** 2 + l_ik_t ** 2 - l_ij_t ** 2) / A
    hnda_ijk = w_ij.unsqueeze(-1) * e_ij_t

    h_ij = 0.5 * l_ij_t
    b_ij = _torch.abs(w_ij) * l_ij_t
    c_ijk = 0.5 * b_ij * h_ij
    return hnda_ijk, c_ijk


def heron_mean_curvature_vectors(points, faces, backend='numpy', device=None):
    """
    Assemble per-vertex mean-curvature vectors for a triangle mesh using the
    vectorized Heron kernel.

    Parameters
    ----------
    points : array_like, shape (V, 3)
        Vertex coordinates.
    faces : array_like, shape (F, 3)
        Triangle connectivity (integer indices into *points*).
    backend : ``'numpy'``, ``'torch'``, or a hyperct ``BatchBackend`` object
        Computation backend.  When a ``BatchBackend`` instance is passed its
        ``batch_heron_curvature`` method is used directly.
    device : str or torch.device, optional
        PyTorch device (only used when *backend* is ``'torch'``).

    Returns
    -------
    H_vecs : ndarray, shape (V, 3)
        Per-vertex integrated mean-curvature normal vector (HNdA_i).
    """
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    n_verts = points.shape[0]
    if faces.size == 0:
        return np.zeros((n_verts, 3), dtype=np.float64)

    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    vi, vj, vk = points[i], points[j], points[k]

    # All six directed edge contributions per triangle
    e_ij = vj - vi;  e_ik = vk - vi
    e_ji = vi - vj;  e_jk = vk - vj
    e_ki = vi - vk;  e_kj = vj - vk

    l_ij = np.linalg.norm(e_ij, axis=1)
    l_ik = np.linalg.norm(e_ik, axis=1)
    l_jk = np.linalg.norm(e_jk, axis=1)

    # Resolve backend: string shorthand or BatchBackend object
    if hasattr(backend, 'batch_heron_curvature'):
        kernel = backend.batch_heron_curvature
        kwargs = {}
        _to_np = np.asarray
    elif backend == 'torch':
        kernel = HNdC_ijk_batch_torch
        kwargs = {'device': device}
        def _to_np(t):
            return t.detach().cpu().numpy()
    else:
        kernel = HNdC_ijk_batch
        kwargs = {}
        def _to_np(t):
            return t

    # Six directed half-edge curvature contributions
    h_ij, _ = kernel(e_ij, l_ij, l_jk, l_ik, **kwargs)
    h_ik, _ = kernel(e_ik, l_ik, l_jk, l_ij, **kwargs)
    h_ji, _ = kernel(e_ji, l_ij, l_ik, l_jk, **kwargs)
    h_jk, _ = kernel(e_jk, l_jk, l_ik, l_ij, **kwargs)
    h_ki, _ = kernel(e_ki, l_ik, l_ij, l_jk, **kwargs)
    h_kj, _ = kernel(e_kj, l_jk, l_ij, l_ik, **kwargs)

    H_vecs = np.zeros((n_verts, 3), dtype=np.float64)
    np.add.at(H_vecs, i, _to_np(h_ij))
    np.add.at(H_vecs, i, _to_np(h_ik))
    np.add.at(H_vecs, j, _to_np(h_ji))
    np.add.at(H_vecs, j, _to_np(h_jk))
    np.add.at(H_vecs, k, _to_np(h_ki))
    np.add.at(H_vecs, k, _to_np(h_kj))
    return H_vecs


"""
Example usage:

# Start main loop
HNdA_ijk_l, C_ijk_l = [], []
C = 0
HNdA = np.zeros(3)
for v in HC.V:
    n_i = v.x_a - np.array([0.0, 0.0, 0.0])  # First approximation
    n_i = normalized(n_i)[0]  
    n_test = n_i + (np.random.rand(3) - 0.5)
    HNdA_i, C_i = hndA_i(v, n_i=n_test)
    C += C_i
    HNdA += HNdA_i 
    

"""