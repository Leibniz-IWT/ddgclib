def curvature_tensors(points_glob, E_ij, n_i=None):
    """
    Compute the curvature tensor of all points.
    :param points_glob: nx3 array, array of all intersection points
    :param E_ij: list of lists, the global edge incidence matrix
    :return: HNdA_ijk: `n x n x n x 3` sparse array, the global curvature tensor
             C_ijk: `n x n x n` array, the dual areas
    """
    # nverts is the total number of intersection points:
    nverts = points_glob.shape[0]  # scalar, number of intersection points
    # Initiate a `n x n x n x 3` array, this should ideally be a sparse array,
    # for simplicity I have initiated a numpy array with the correct
    # dimensions:
    HNdA_ijk = np.zeros([nverts, nverts, nverts, 3])
    # For dual areas we need a `n x n x n` array, this should again ideally be a sparse array,
    # for simplicity I have initiated a numpy array with the correct
    # dimensions:
    C_ijk = np.zeros([nverts, nverts, nverts])

    # Initiate vector of shape (number of intersection points, 3):
    HNdA_i = np.zeros(points_glob.shape)

    # Start main `for` loop for each intersection plot
    for i in range(nverts):
        # NOTE: THIS MUST BE REPLACED WITH THE LEVEL SET PLANE VECTOR:
        n_i = points_glob[i] - np.array([0.5, 0.5, 0.5])  # First approximation
        #######################################################################
        # Initiate
        HNdA_ij = np.zeros([len(E_ij), 3])
        for j in E_ij[i]:
            # Compute the intersection set of vertices i and j:
            e_i_int_e_j = E_ij[i].intersection(E_ij[j])  # Set of size 1 or 2
            e_ij = points_glob[j] - points_glob[i]  # Compute edge ij (1x3 vector)
            e_ik = points_glob[k] - points_glob[i]  # Compute edge ik (1x3 vector)

            # Discrete vector area:
            # Simplex areas of ijk and normals
            wedge_ij_ik = np.cross(e_ij, e_ik)
            if np.dot(normalized(wedge_ij_ik)[0], n_i) < 0:
                e_ij = points_glob[i] - points_glob[j]

            if len(e_i_int_e_j) == 1:  # boundary edge
                k = list(e_i_int_e_j)[0]  # Boundary edge index
                # Compute edges in triangle ijk
                e_ik = points_glob[k] - points_glob[i]
                e_jk = points_glob[k] - points_glob[j]
                # Find lengths (norm of the edge vectors):
                l_ij = np.linalg.norm(e_ij)
                l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
                l_jk = np.linalg.norm(e_jk)
                hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)
                # Save results
                HNdA_ijk[i][j][k] = hnda_ijk
                C_ijk[i][j][k] = c_ijk

            else:  # len(e_i_int_e_j) == 2 mathematically guaranteed:
                k = list(e_i_int_e_j)[0]  # index in triangle ijk
                l = list(e_i_int_e_j)[1]  # index in triangle ijl

                # Compute dual for contact angle alpha
                e_jk = points_glob[k] - points_glob[j]
                # wedge_ij_ik = np.cross(e_ij, e_ik)
                # Find lengths (norm of the edge vectors):
                l_ij = np.linalg.norm(e_ij)
                l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
                l_jk = np.linalg.norm(e_jk)
                hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)

                # Contact angle beta
                e_il = points_glob[l] - points_glob[i]
                e_jl = points_glob[l] - points_glob[j]
                l_il = np.linalg.norm(e_il)  # NOTE: l_ki = l_ik
                l_jl = np.linalg.norm(e_jl)
                hnda_ijl, c_ijl = HNdC_ijk(e_ij, l_ij, l_jl, l_il)

                # Save results
                HNdA_ijk[i][j][k] = hnda_ijk
                C_ijk[i][j][k] = c_ijk
                HNdA_ijk[i][j][l] = hnda_ijl
                C_ijk[i][j][l] = c_ijl

    return HNdA_ijk, C_ijk