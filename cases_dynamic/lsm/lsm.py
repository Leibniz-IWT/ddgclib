# Std. library for Python arrays:
import numpy as np

# Plotting library:
import matplotlib.pyplot as plt

# Local python scripts to extract test cases data
from data_levelset_geometric_shapes.extract_and_process_interface_points import read_data, \
    extract_and_save_unique_intersection_points, plot_intersection_points, plot_unique_intersections, \
    plot_and_save_levelset_points

# Black box computation of curvature in a cell withing a
# `cell_corners_and_intersection` object supplied from the LSM cells.
def curvature_cell_i(cell_i, cell_corners_and_intersections):
    """
    Compute the integrated mean normal curvature in a cell with index cell_i.

    :param cell_i: int, index of current cell for which to compute the curvature
    :param cell_corners_and_intersections: list of dict,
                list of dictionaries containing `intersections` and `corners` of cells
                currently under consideration.
    :return: hndA_i, 1x3 vector, the total integrated mean normal curvature in a cell
    """
    # Compute the global hash table and the connectivity of all intersection
    # points in the current "cell_corners_and_intersections" list
    points_hash, points_glob, E_ij = triangulate_cells(cell_corners_and_intersections)

    # Compute the curvature tensors and dual areas of all intersection points:
    #HNdA_ijk, C_ijk = curvature_tensors(points_glob, E_ij)
    HNdA_ijk, C_ijk = curvature_tensors_new(points_glob, E_ij)

    # Compute the curvatures at each vertex:
    HNdA_i = np.sum(HNdA_ijk, axis=(1, 2))
    ######
    # NOTE: the above line is equivalent to (possibly easier in C++):
    # HNdA_i = np.zeros([nverts, 3])
    # for i in range(nverts):
    #    for HNdA_jk in HNdA_ijk[i]:
    #        for HNdA_k in HNdA_jk:
    #            HNdA_i[i] += HNdA_k
    ######

    C_i = np.sum(C_ijk, axis=(1, 2))
    ######
    # NOTE: the above line is equivalent to (possibly easier in C++):
    # C_i = np.zeros([nverts])
    # for i in range(nverts):
    #    for C_jk in C_ijk[i]:
    #        for C_k in C_jk:
    #            C_i[i] += C_k
    ######

    # Finally we compute the curvature of the local cell i:
    hndA_i = hndA_i_cell(cell_i, cell_corners_and_intersections, E_ij, C_i, HNdA_i, points_hash, C_ijk)
    return hndA_i


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
    w_ij = (1 / 8.0) * (l_jk ** 2 + l_ik ** 2 - l_ij ** 2) / A   # w_ij = abs(w_ij)

    # Mean normal curvature (1x3 vector):
    hnda_ijk = w_ij * e_ij  # curvature from this edge jk in triangle ijk with w_jk = 1/2 cot(theta_i^jk)

    # Dual areas
    h_ij = 0.5 * l_ij
    b_ij = abs(w_ij) * l_ij  # = ||0.5 cot(theta_i^jk)|| * 0.5*l_ij
    c_ijk = 0.5 * b_ij * h_ij
    return hnda_ijk, c_ijk


def hndA_i_cell(i_cell, cell_corners_and_intersections, E_ij, C_i, HNdA_i, points_hash, C_ijk):
    """
    :param i_cell: int, index of current cell for which to compute the curvature
    :param cell_corners_and_intersections: list of dict,
                list of dictionaries containing `intersections` and `corners` of cells
                currently under consideration.
    :param E_ij: list of lists, the global edge incidence matrix.
    :param C_i: `n` array, the dual areas around each vertex.
    :param HNdA_i: `n x 3` array, the mean normal curvature around each vertex.
    :param C_ijk: `n x n x n` array, the dual areas.

    :return: hndA_i, 1x3 vector, the total integrated mean normal curvature in a cell
    """
    # Find the contributions that need to be added here to each cell in the current vertex
    # use the edge
    corners = cell_corners_and_intersections[i_cell]["Corners"]
    intersections = cell_corners_and_intersections[i_cell]["Intersections"]
    nverts = intersections.shape[0]
    local_verts = set()
    for p in intersections:
        # print(points_hash[tuple(p)])
        local_verts.add(points_hash[tuple(p)])  # = i

    # Find the fraction of c_i in all c_ijk in cell:
    # c_i_cell = np.zeros([nverts])  # Local dual areas, array of size number of intersection points
    hndA_i = np.zeros([3])  # Curvature of current cell, a `1 x n` vector
    for i in list(local_verts):  # For each vertex
        c_i_cell = 0  # Local dual area for current i, scalar
        # Find the fraction of c_i in all c_ijk in cell:
        for j in E_ij[i]:
            e_i_int_e_j = E_ij[i].intersection(E_ij[j])  # Set of size 1 or 2
            ind_k = e_i_int_e_j.intersection(local_verts)
            if len(ind_k) == 1:
                k = list(ind_k)[0]
                c_i_cell += C_ijk[i, j, k]
            elif len(ind_k) == 2:
                k = list(ind_k)[0]
                c_i_cell += C_ijk[i, j, k]
                l = list(ind_k)[1]
                c_i_cell += C_ijk[i, j, l]

        # Compute the fraction dual area:
        frac_c_i = C_i[i] / c_i_cell

        # Compute the fraction curvature contributed by the current vertex i
        hndA_i += frac_c_i * HNdA_i[i]

    return hndA_i

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
                e_ik = points_glob[k] - points_glob[i]  # Compute edge ik (1x3 vector)

                # Discrete vector area:
                # Simplex areas of ijk and normals
                wedge_ij_ik = np.cross(e_ij, e_ik)
                # If the wrong direction was chosen, choos the other:
                if np.dot(normalized(wedge_ij_ik)[0], n_i) < 0:
                    k_t = k
                    l_t = l
                    k = l_t
                    l = k_t
                    e_ij = points_glob[j] - points_glob[i]
                    e_ik = points_glob[k] - points_glob[i]

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


def curvature_tensors_new(points_glob, E_ij, n_i=None):
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
            k = list(e_i_int_e_j)[0]  # Boundary edge index or index in triangle ijk
            e_ij = points_glob[j] - points_glob[i]  # Compute edge ij (1x3 vector)
            e_ik = points_glob[k] - points_glob[i]  # Compute edge ik (1x3 vector)

            # Discrete vector area:
            # Simplex areas of ijk and normals
            wedge_ij_ik = np.cross(e_ij, e_ik)
            if np.dot(normalized(wedge_ij_ik)[0], n_i) < 0:
                e_ij = points_glob[i] - points_glob[j]  # TODO: Is this correct???
                #e_ij = points_glob[j] - points_glob[i]  # Gives old results

            if len(e_i_int_e_j) == 1:  # boundary edge
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

def assign_incides(intersections, corners):
    """
    Assign local indices for triangulation.

    :param intersections: `n x 3` array of all intersection points in cells.
    :param corners: `8 x 3` corners of current cell.
    :return: pind_order, list of int, order of vertices.
    """
    # Return ordered points and connectivity matrix E_ij
    nverts = intersections.shape[0]  # number of intersection in current cell
    pf_indices = []
    pind = 0
    pind_order = []  # or int dtype array of size intersections.shape[0]
    # Compute the faces
    for p in intersections:
        pi = find_cell_faces(p, corners)
        pf_indices.append(pi)

    # Find the correct order of points of the intersections
    pind_order.append(0)  # Arbitarily select the first point
    pind = 0  # Previous index
    cf = pf_indices[0][0]  # current face
    while len(pind_order) < nverts:
        for i in range(len(pf_indices)):
            if i == pind:
                continue
            ci = pf_indices[i]
            if cf == ci[0]:
                pind_order.append(i)
                cf = ci[1]  # Move on to new face
                pind = i  # Make i the previous index for the next loop
                break  # Break out of current for loop, continue to next vertex
            elif cf == ci[1]:
                pind_order.append(i)
                cf = ci[0]  # Move on to new face
                pind = i  # Make i the previous index for the next loop
                break  # Break out of current for loop, continue to next vertex

    return pind_order

# Triangulation:
def triangulate_cells(cell_corners_and_intersections):
    """
    Triangulate all cells.

    :param cell_corners_and_intersections: list of dict,
                list of dictionaries containing `intersections` and `corners` of cells
                currently under consideration.
    :return: points_hash: dict, hash table of points (tuple of 1x3 vector --> hashed int index
             points_glob: nx3 array, array of all intersection points
             E_ij: list of lists, the global edge incidence matrix

    """
    E_ij = []
    points_hash, points_glob = glob_hash(cell_corners_and_intersections)
    for i in range(points_glob.shape[0]):  # for the number of points currently in global pool
        E_ij.append(set())

    # ncells = len(cell_corners_and_intersections)  # Number of cells currently under consideration
    # for i in range(ncells):
    # Compute the hash table of all points currently in cell_corners_and_intersections:
    points_hash, points_glob = glob_hash(cell_corners_and_intersections)

    # For each cell,
    E_ij = E_ij_cells(cell_corners_and_intersections, points_hash, E_ij)

    return points_hash, points_glob, E_ij


def glob_hash(cell_corners_and_intersections):
    """
    Build a global hash table and array of all points under consideration.

    :param cell_corners_and_intersections: list of dict,
                list of dictionaries containing `intersections` and `corners` of cells
                currently under consideration.
    :return: points_hash: dict, hash table of points (tuple of 1x3 vector --> hashed int index
             points_glob: nx3 array, array of all intersection points
    """
    # Hash table of intersection indices
    # This is a global hash table of all intersections points.
    # If computing only one local cell is desired, then at minimum
    # the 9 surrounding cells need to be included for curvature to
    # have physical meaning.
    points_hash = {}
    points_glob = []  # or a n x 3 array, where n is the total number of points
    i = 0
    for index, c in enumerate(cell_corners_and_intersections):
        corners = c["Corners"]
        intersections = c["Intersections"]
        for p in intersections:
            try:
                points_hash[tuple(p)]
            except KeyError:
                points_hash[tuple(p)] = i
                i = i + 1
                points_glob.append(p)  # or set row i vector equal to p

    points_glob = np.array(points_glob)  # convert list of points to array
    return points_hash, points_glob


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


def E_ij_cells(cell_corners_and_intersections, points_hash, E_ij):
    """
    Compute global E_ij for all cells under consideration in
    `cell_corners_and_intersections`
    :param cell_corners_and_intersections: list of dict,
                list of dictionaries containing `intersections` and `corners` of cells
                currently under consideration.
    :return: E_ij, list of lists, locally connected edges
    """
    for index, c in enumerate(cell_corners_and_intersections):
        corners = c["Corners"]
        intersections = c["Intersections"]
        p, E_ij_local = assign_incides_graph(intersections, corners)
        points = intersections[p]  # Local points in correct order
        # get global incides
        for pi in p:  # Loop for the index of each local point
            i = points_hash[tuple(points[pi])]  # Get the global index
            for pj in E_ij_local[pi]:  # loop local connects
                if pi == pj:
                    continue
                j = points_hash[tuple(points[pj])]  # Find global index of connection j
                E_ij[i].add(j)
    return E_ij

def assign_incides_graph(intersections, corners):
    """
    Assign the indices of a
    :param intersections: `n x 3` array of all intersection points in cells
    :param corners: `8 x 3` corners of current cell
    :return: pind_order,  list, order of vertices
             E_ij, list of lists, locally connected edges
    """
    # Return ordered points and connectivity matrix E_ij
    nverts = intersections.shape[0]  # number of intersection in current cell
    pf_indices = []
    pind = 0
    pind_order = []  # or int dtype array of size intersections.shape[0]
    # Compute the faces
    for p in intersections:
        pi = find_cell_faces(p, corners)
        pf_indices.append(pi)

    # Find the correct order of points of the intersections
    pind_order.append(0)  # Arbitarily select the first point
    pind = 0  # Previous index
    cf = pf_indices[0][0]  # current face
    while len(pind_order) < nverts:
        for i in range(len(pf_indices)):
            if i == pind:
                continue
            ci = pf_indices[i]
            if cf == ci[0]:
                pind_order.append(i)
                cf = ci[1]  # move on to new face
                pind = i  # Make i the previous index for the next loop
                break
            elif cf == ci[1]:
                pind_order.append(i)
                cf = ci[0]  # move on to new face
                pind = i  # Make i the previous index for the next loop
                break

    E_ij = graph_e_ij(nverts)  # Edges present in current cell

    return pind_order, E_ij

# Misc
def normalized(a, axis=-1, order=2):
    """
    Normalize the input vector a.
    :param a: vector
    :return: an, vector a normalized
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

# Data imports and visualisation (Alexander Bussman's code):
def plot_intersections(axes, intersections, corners, plot_thick):
    number_of_inter = len(intersections)
    for idx in range(0, number_of_inter):
        this_inter = intersections[idx]
        this_inter_faces = find_cell_faces(this_inter, corners)
        for nxt_idx in range(idx + 1, number_of_inter):
            next_inter = intersections[nxt_idx]
            next_inter_faces = find_cell_faces(next_inter, corners)

            if len(this_inter_faces.intersection(next_inter_faces)) > 0:
                if plot_thick:
                    axes.plot([this_inter[0], next_inter[0]], [this_inter[1], next_inter[1]],
                              [this_inter[2], next_inter[2]], marker="x", markersize=1.5, linewidth=0.5, color="red")
                else:
                    axes.plot([this_inter[0], next_inter[0]], [this_inter[1], next_inter[1]],
                              [this_inter[2], next_inter[2]], marker="x", markersize=0.25, linewidth=0.25, color="red")


def find_cell_faces(intersection_point, cell_corners):
    all_sides = [
        0 if intersection_point[0] == cell_corners[0][0] else None,
        1 if intersection_point[0] == cell_corners[6][0] else None,
        2 if intersection_point[1] == cell_corners[0][1] else None,
        3 if intersection_point[1] == cell_corners[6][1] else None,
        4 if intersection_point[2] == cell_corners[0][2] else None,
        5 if intersection_point[2] == cell_corners[6][2] else None
    ]
    # return set([side for side in all_sides if side is not None])
    return [side for side in all_sides if side is not None]
