import numpy as np

def chi_H(HC, print_out=False):
    """
    Compute the 2D Euler Characterisitc
    """
    ## Computer Euler Characteristic:
    # Compute the number of vertices
    V = len(list(HC.V))
    # Compute the dges
    E = 0
    for v in HC.V:
        E += len(list(v.nn))

    E = E / 2.0  # We have added precisely twice the number of edges through adding each connection
    # Compute the faces
    HC.dim = 2  # We have taken the boundary of the sphere
    HC.vertex_face_mesh()
    Faces = HC.simplices_fm
    F = len(Faces)

    # Compute Euler
    chi = V - E + F
    if print_out:
        print(f'V = {V}')
        print(f'E = {E}')
        print(f'F = {F}')
        print(f'$\chi = V - E + F$ = {chi}')

    return chi


def K_t(HC, bV=set()):
    """
    Compute the integrated Gaussian curvature on the surface
    """
    KdA = 0.0
    for v in HC.V:
        if v in bV:
            continue
        else:
            N_f0 = v.x_a - np.zeros_like(v.x_a)  # First approximation # TODO: CHANGE FOR CAP RISE!
            F, nn = vectorise_vnn(v)
            # Compute discrete curvatures
            c_outd = curvatures(F, nn, n_i=N_f0)
            KdA += c_outd['Omega_i']  # == c_outd['K']

    return KdA


def k_g_t(HC, bV):
    k_g = 0
    for v in bV:
        Theta_i_jk = 0.0
        Simplices = set()
        Dual = set()
        for vn in v.nn:
            if vn in Dual:
                continue
            for vnn in vn.nn:
                if vnn in v.nn:  # Add edges connected to v_i
                    E_ij = vn.x_a - v.x_a
                    E_ik = vnn.x_a - v.x_a

                    # Discrete vector area:
                    # Simplex areas of ijk and normals
                    wedge_ij_ik = np.cross(E_ij, E_ik)

                    # Wedge_ij_ik[j] = wedge_ij_ik
                    theta_i_jk = np.arctan2(np.linalg.norm(wedge_ij_ik), np.dot(E_ij, E_ik))
                    Theta_i_jk += theta_i_jk
                    Dual.add(vnn)

        k_g += np.pi - Theta_i_jk

    return k_g


def Gauss_Bonnet(HC, bV=set(), print_out=False):
    """
    Compute a single iteration of mean curvature flow
    :param HC: Simplicial complex
    :return:
    """
    chi = chi_H(HC)
    KdA = K_t(HC, bV)
    k_g = k_g_t(HC, bV)
    if print_out:
        print(f' KdA = {KdA}')
        print(f' k_g = {k_g}')
        print(f' 2 pi chi$ = {2 * np.pi * chi}')
        print(f' LHS - RHS = {KdA + k_g - 2 * np.pi * chi}')
    return chi, KdA, k_g

