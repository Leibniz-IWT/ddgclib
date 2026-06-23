def HNdC_ijk(e_ij, l_ij, l_jk, l_ik):
    """
    Computes the dual edge and dual area using Heron's formula.

    :param e_ij: vector, edge e_ij
    :param l_ij: float, length of edge ij
    :param l_jk: float, length of edge jk
    :param l_ik: float, length of edge ik
    :return: hnda_ijk: vector, curvature vector
             c_ijk: float, dual areas

    DEV NOTE: This function is validated moreso than curvature_tensors
    #         in the sense that we can use hyperct.Complex triangulations
    #         of the sphere to get exact point-wise results
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

    # Dual areas (NOTE: NOT needed for just curvatures):
    h_ij = 0.5 * l_ij
    b_ij = abs(w_ij) * l_ij  # = ||0.5 cot(theta_i^jk)|| * 0.5*l_ij
    c_ijk = 0.5 * b_ij * h_ij
    return hnda_ijk, c_ijk

def graph_e_ij(nverts):
    """
    Returns the E_ij graph of edges depending on the number of vertices.
    :param nverts: float, total number of vertices in current cell.
    :return: E_ij, list of lists, the local edge incidence matrix
    """

    if nverts == 3:
        E_ij = [[1, 2],  # edges connected to vertex 0
                [0, 2],  # edges connected to vertex 1
                [0, 1],  # edges connected to vertex 2
                ]
    elif nverts == 4:  # 2 simplices
        E_ij = [[1, 2, 3],  # edges connected to vertex 0
                [0, 2],  # edges connected to vertex 1
                [0, 1, 3],  # edges connected to vertex 2
                [0, 2],  # edges connected to vertex 3
                ]
    elif nverts == 5:  # 3 simplices
        E_ij = [[1, 2, 4],  # edges connected to vertex 0
                [0, 2],  # edges connected to vertex 1
                [0, 1, 3, 4],  # edges connected to vertex 2
                [2, 4],  # edges connected to vertex 3
                [0, 2, 3],  # edges connected to vertex 4
                ]
    elif nverts == 6:  # 4 simplices
        E_ij = [[1, 2, 4, 5],  # edges connected to vertex 0
                [0, 2],  # edges connected to vertex 1
                [0, 1, 3, 4],  # edges connected to vertex 2
                [2, 4],  # edges connected to vertex 3
                [0, 2, 3, 5],  # edges connected to vertex 4
                [0, 4],  # edges connected to vertex 5
                ]
    return E_ij  # [0]