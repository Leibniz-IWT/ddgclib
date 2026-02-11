import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import numpy as np
import scipy as sp
from scipy import spatial as sp_spatial

from pathlib import Path
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from ddgclib._complex import Complex
from ddgclib._curvatures import *  # plot_surface#, curvature
from ddgclib._curvatures import b_curvatures_hn_ij_c_ij
from data_levelset_geometric_shapes.extract_and_process_interface_points import \
    read_data, \
    extract_and_save_unique_intersection_points, plot_intersection_points, \
    plot_unique_intersections, \
    plot_and_save_levelset_points


def HNdC_ijk(e_ij, l_ij, l_jk, l_ik):
    lengths = [l_ij, l_jk, l_ik]
    # Sort the list, python sorts from the smallest to largest element:
    lengths.sort()
    # We must have use a ≥ b ≥ c in floating-point stable Heron's formula:
    a = lengths[2]
    b = lengths[1]
    c = lengths[0]
    A = (1 / 4.0) * np.sqrt(
        (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c)))
    # Dual weights (scalar):
    w_ij = (1 / 8.0) * (l_jk ** 2 + l_ik ** 2 - l_ij ** 2) / A

    # Mean normal curvature (1x3 vector):
    hnda_ijk = w_ij * e_ij  # curvature from this edge jk in tringle ijk with w_jk = 1/2 cot(theta_i^jk)

    # Dual areas
    h_ij = 0.5 * l_ij
    b_ij = abs(w_ij) * l_ij  # = ||0.5 cot(theta_i^jk)|| * 0.5*l_ij
    c_ijk = 0.5 * b_ij * h_ij
    return hnda_ijk, c_ijk


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
                    axes.plot([this_inter[0], next_inter[0]],
                              [this_inter[1], next_inter[1]],
                              [this_inter[2], next_inter[2]], marker="x",
                              markersize=1.5, linewidth=0.5, color="red")
                else:
                    axes.plot([this_inter[0], next_inter[0]],
                              [this_inter[1], next_inter[1]],
                              [this_inter[2], next_inter[2]], marker="x",
                              markersize=0.25, linewidth=0.25, color="red")


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


def f_ijk(nverts):
    # Returns the F_ijk matrix of faces depending on the number of vertices
    F_ijk = np.zeros([nverts - 2, 3], dtype=int)
    if nverts == 3:
        F_ijk[:] = [0, 1, 2]
    elif nverts == 4:  # 2 simplices
        F_ijk[0, :] = [0, 1, 2]
        F_ijk[1, :] = [0, 2, 3]
    elif nverts == 5:  # 3 simplices
        F_ijk[0, :] = [0, 1, 2]
        F_ijk[1, :] = [0, 2, 4]
        F_ijk[2, :] = [2, 3, 4]
    elif nverts == 6:  # 4 simplices
        F_ijk[0, :] = [0, 1, 2]
        F_ijk[1, :] = [0, 2, 4]
        F_ijk[2, :] = [2, 3, 4]
        F_ijk[3, :] = [0, 3, 4]

    return F_ijk


def assign_incides(intersections, corners):
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

    F_ijk = f_ijk(nverts)  # Triangles present in current cell
    return pind_order, f_ijk(nverts)


def glob_hash(cell_corners_and_intersections):
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
    # Returns the E_ij graph of edges depending on the number of vertices
    # use std::set for the final version
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
    for index, c in enumerate(cell_corners_and_intersections):
        corners = c["Corners"]
        intersections = c["Intersections"]
        p, F_ijk_local, E_ij_local = assign_incides_graph(intersections,
                                                          corners)
        points = intersections[p]  # Local points in correct order
        # get global incides
        for pi in p:  # Loop for the index of each local point
            i = points_hash[tuple(points[pi])]  # Get the global index
            for pj in E_ij_local[pi]:  # loop local connects
                if pi == pj:
                    continue
                j = points_hash[
                    tuple(points[pj])]  # Find global index of connection j
                E_ij[i].add(j)
    return E_ij


def assign_incides_graph(intersections, corners):
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

    F_ijk = f_ijk(nverts)  # Triangles present in current cell
    E_ij = graph_e_ij(nverts)  # Edges present in current cell

    return pind_order, F_ijk, E_ij


def triangulate_cells(cell_corners_and_intersections):
    E_ij = []
    points_hash, points_glob = glob_hash(cell_corners_and_intersections)
    for i in range(points_glob.shape[
                       0]):  # for the number of points currently in global pool
        E_ij.append(set())

    # ncells = len(cell_corners_and_intersections)  # Number of cells currently under consideration
    # for i in range(ncells):
    # Compute the hash table of all points currently in cell_corners_and_intersections:
    points_hash, points_glob = glob_hash(cell_corners_and_intersections)

    # For each cell,
    E_ij = E_ij_cells(cell_corners_and_intersections, points_hash, E_ij)

    return points_hash, points_glob, E_ij


filename = Path("../../ddgclib/data_levelset_geometric_shapes/sphere_coarse/extraction_data_0.000000.txt")
result_folder = Path("../../ddgclib/data_levelset_geometric_shapes/X_intersections_sphere_coarse")
plot_single_cells = False

# Create the result folder
if not os.path.exists(result_folder):
    os.mkdir(result_folder)

# Read and plot the data
corners_and_intersections = read_data(filename)
#unique_intersections = extract_and_save_unique_intersection_points(corners_and_intersections, result_folder)
#plot_and_save_levelset_points(corners_and_intersections, result_folder)
#plot_unique_intersections(unique_intersections, result_folder)
#plot_intersection_points(corners_and_intersections, result_folder, plot_single_cells)

cell_corners_and_intersections = corners_and_intersections
def intersection_is_found(intersection, ref_intersections):
    return any([all([np.abs(coord - coord_ref) <= 1e-14 for coord, coord_ref in zip(intersection, ref_inter)]) for ref_inter in ref_intersections])
    # Get unique intersections
all_intersections    = [inter for data in cell_corners_and_intersections for inter in data["Intersections"]]
unique_intersections = np.unique(np.array(all_intersections), axis=0)

lpoints = []
for p in cell_corners_and_intersections:
    lpoints.append(p['Levelset'][0])

lpoints = np.array(lpoints)

import scipy.spatial
tri = scipy.spatial.Delaunay(lpoints)
hull = scipy.spatial.ConvexHull(lpoints)


lpoints = []
for p in cell_corners_and_intersections:
    lpoints.append(p['Levelset'][0])

lpoints = np.array(lpoints)


#
indices = tri.simplices
faces = tri.points[indices]
#

print('area: ', hull.area)
print('volume: ', hull.volume)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.dist = 30
#ax.azim = -140
ax.set_xlim([0.3, 0.7])
ax.set_ylim([0.3, 0.7])
ax.set_zlim([0.3, 0.7])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

for f in faces:
    face = a3.art3d.Poly3DCollection([f])
    face.set_color(mpl.colors.rgb2hex(sp.rand(3)))
    face.set_edgecolor('k')
    face.set_alpha(0.5)
    ax.add_collection3d(face)

#
HC = Complex(3)
#HC.vf_to_vv(tri.points, tri.simplices)
#dV = HC.boundary_d(HC.V)
fig = plt.figure()
HC.vf_to_vv(tri.points, tri.simplices)
dV = HC.boundary_d(HC.V)
HC.plot_complex()
#HC.V.print_out()
plt.draw()
#plt.show()
plt.draw()
##

plt.show()