#!/usr/bin/env python
# coding: utf-8

import copy
import sys
import numpy as np
import polyscope as ps

# Local library imports
from ddgclib import *
from ddgclib._curvatures import HC_curvatures_sessile
from hyperct import *
from ddgclib._sphere import *
from ddgclib._sessile import *
from ddgclib._capillary_rise_flow import *  # plot_surface, curvature
from ddgclib._eos import *
from ddgclib._plotting import plot_polyscope


def sector_volume(HC):
    """
    Compute the volume of the complex by splitting it into sectors centered on the origin.

    This method computes the volume contribution for each vertex (sector)
    using its dual area and the projection of the vertex position along
    an approximated normal.

    Parameters
    ----------
    HC : object
        The complex object containing vertices and connectivity information.
        It is expected that HC.V contains the vertices with attributes 'x' and 'x_a'.

    Returns
    -------
    float
        The total bubble volume computed by summing contributions from all sectors.
    """
    (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
     HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
     Theta_i_cache) = HC_curvatures_sessile(HC, bV, RadFoot, theta_p, printout=0)
    total_bubble_volume = 0.0
    totalArea = 0.0
    for v in HC.V:
        dualArea = sum(C_ij_cache[v.x])
        totalArea += dualArea
        H = HNdA_i_cache[v.x]
        N_approx = -normalized(H)[0]
        total_bubble_volume += dualArea * sum(v.x_a[:] * N_approx[:]) / 3.0
    return total_bubble_volume


def prism_volume(HC):
    """
    Compute the volume of the complex by splitting it into prisms over the surface.

    Note
    ----
    Boundaries are not included.

    Parameters
    ----------
    HC : object
        The complex object containing vertices and connectivity information.

    Returns
    -------
    float
        The total bubble volume computed by summing contributions from prisms.
    """
    (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
     HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
     Theta_i_cache) = HC_curvatures_sessile(HC, bV, RadFoot, theta_p, printout=0)
    total_bubble_volume = 0.0
    totalArea = 0.0
    for v in HC.V:
        dualArea = sum(C_ij_cache[v.x])
        totalArea += dualArea
        H = HNdA_i_cache[v.x]
        N_approx = outward_normal(v, H)
        total_bubble_volume += dualArea * N_approx[2] * v.x_a[2]
    return total_bubble_volume


def triangle_prism_volume(HC):
    """
    Compute the volume of the complex by splitting it into triangular prisms over the surface.

    Parameters
    ----------
    HC : object
        The complex object containing vertices and connectivity information.

    Returns
    -------
    tuple of (float, float, float)
        total_bubble_volume : Total bubble volume.
        total_bubble_area   : Total bubble area.
        bubble_centroid     : Bubble centroid (weighted z-coordinate average).
    """
    total_bubble_volume = 0.0
    total_bubble_area = 0.0
    bubble_centroid = 0.0
    for v in HC.V:
        for vn1 in v.nn:
            for vn2 in v.nn:
                if vn2 in vn1.nn:
                    triArea = 0.5 * cross_prod(vn1.x_a - v.x_a, vn2.x_a - v.x_a)
                    # Set the area vector pointing away from the z-axis.
                    if sum(triArea[:3] * v.x_a[:3]) < 0:
                        triArea = -triArea
                    triangle_centroid = (v.x_a[2] + vn1.x_a[2] + vn2.x_a[2]) / 3
                    prism_volume = triArea[2] * triangle_centroid / 2 / 3
                    total_bubble_volume += prism_volume
                    bubble_centroid += triangle_centroid * prism_volume / 2
                    total_bubble_area += np.linalg.norm(triArea) / 2 / 3
                    # Divide by 2 because vn1 and vn2 are equivalent.
                    # Divide by 3 because each triangle is counted at the three vertices.
    bubble_centroid = bubble_centroid / total_bubble_volume
    return total_bubble_volume, total_bubble_area, bubble_centroid


def save_neighbours(fname):
    """
    Save a text file listing the vertices and their connected neighbors.

    The first column contains the vertex id (starting from 0), followed by the
    ids of the connected vertices.

    Parameters
    ----------
    fname : str
        The filename where the neighbor information will be saved.
    """
    vNum = 0
    posDic = {}
    for v in HC.V:
        posDic[v.x] = vNum
        vNum += 1
    with open(fname, "w") as nei_txt:
        print('saving', fname)
        vNum = 0
        for v in HC.V:
            print(vNum, ' ', end='', file=nei_txt)
            for nei in v.nn:
                try:
                    print(posDic[nei.x], ' ', end='', file=nei_txt)
                except KeyError:
                    print('KeyError caught, not in dictionary', nei.x)
            print('', file=nei_txt)
            vNum += 1
    return


def save_vert_positions(t):
    """
    Save the xyz positions of all vertices in a text file and also save neighbor data.

    The positions are saved to 'data/pos{t}.txt' and the neighbor list is saved
    to 'data/nei{t}.txt'.

    Parameters
    ----------
    t : int or str
        The time step or identifier used in the filename.
    """
    fname = 'data/pos' + str(t) + '.txt'
    with open(fname, "w") as pos_txt:
        print('saving', fname)
        for v in HC.V:
            print(*v.x, file=pos_txt)
    save_neighbours('data/nei' + str(t) + '.txt')


def load_complex(t):
    """
    Load a complex from position and neighbor files and construct the complex.

    Parameters
    ----------
    t : int or str
        The time step or identifier used to determine the filenames.

    Returns
    -------
    tuple
        HC : The constructed complex.
        bV : set of vertices that are considered boundary vertices.
    """
    fname = 'data/pos' + str(t) + '.txt'
    with open(fname) as f:
        print('loading', fname)
        pos = [[float(x) for x in line.split()] for line in f]
    fname = 'data/nei' + str(t) + '.txt'
    with open(fname) as f:
        print('loading', fname)
        nn = [[int(x) for x in line.split()[1:]] for line in f]
    HC = construct_HC(pos, nn)
    bV = set()
    for v in HC.V:
        if abs(v.x[2]) < 1e-9:
            bV.add(v)
    return HC, bV


def refine_edges(HC, dist):
    """
    Refine the complex by splitting edges longer than a specified distance.

    An edge is split by inserting a vertex at its midpoint if the edge is
    in a planar region. Neighboring edges are not refined.

    Parameters
    ----------
    HC : object
        The complex object containing vertices and their connectivity.
    dist : float
        The threshold distance above which edges will be refined.

    Returns
    -------
    int
        The number of edges that were refined.
    """
    to_split_1D = []
    to_split = []
    for i, v1 in enumerate(HC.V):
        if v1 in to_split_1D:
            continue
        for v2 in v1.nn:
            if v2 in to_split_1D:
                continue
            if np.linalg.norm(v1.x_a - v2.x_a) > dist:
                common_neigh = list(v1.nn.intersection(v2.nn))
                # Only refine if the vertices are in a planar region.
                if len(common_neigh) != 2:
                    continue
                if any(n in to_split_1D for n in common_neigh):
                    continue
                to_split.append((v1, v2, *common_neigh))
                to_split_1D.extend((v1, v2, *common_neigh))
    if len(to_split) == 0:
        return 0
    for (v1, v2, *common_neigh) in to_split:
        v1.disconnect(v2)
        # Compute vertex at the center of the edge.
        v_pos = 0.5 * v1.x_a + 0.5 * v2.x_a
        v_new = HC.V[tuple(v_pos)]
        # Connect the new vertex to the original two vertices.
        v_new.connect(v1)
        v_new.connect(v2)
        for n in common_neigh:
            v_new.connect(n)
    return len(to_split)


def refine_boundaries(HC, bV, dist):
    """
    Refine the boundaries of the complex by splitting boundary edges longer than a given distance.

    Parameters
    ----------
    HC : object
        The complex object containing vertices and connectivity.
    bV : set
        Set of boundary vertices.
    dist : float
        The threshold distance above which boundary edges will be refined.

    Returns
    -------
    int
        The number of boundary edges that were refined.
    """
    to_split = []
    to_connect = []
    for i, v1 in enumerate(bV):
        for v2 in v1.nn:
            if v2 not in bV:
                continue
            if (v2, v1) in to_split:
                continue
            if np.linalg.norm(v1.x_a - v2.x_a) > dist:
                common_neigh = list(v1.nn.intersection(v2.nn))
                # Only refine if the vertices are in a planar region.
                if len(common_neigh) != 1:
                    continue
                if any(n in bV for n in common_neigh):
                    continue
                to_split.append((v1, v2))
                to_connect.append(common_neigh[0])
    if len(to_split) == 0:
        return 0
    for i in range(len(to_split)):
        (v1, v2) = to_split[i]
        n = to_connect[i]
        v1.disconnect(v2)
        # Compute vertex at the center of the edge.
        v_pos = 0.5 * v1.x_a + 0.5 * v2.x_a
        v_new = HC.V[tuple(v_pos)]
        # Connect the new vertex to the original vertices and the common neighbor.
        v_new.connect(v1)
        v_new.connect(v2)
        v_new.connect(n)
        bV.add(v_new)
    return len(to_split)


def reconnect_long_diagonals(HC, bV):
    """
    Reconnect long diagonals in the complex by identifying bisected quadrilaterals.

    The function finds quadrilaterals and reconnects them by disconnecting one pair of
    vertices and connecting another pair to improve the mesh quality.

    Parameters
    ----------
    HC : object
        The complex object.
    bV : set
        Set of boundary vertices to exclude from reconnection.

    Returns
    -------
    int
        The number of reconnections performed.
    """
    to_reconnect = []
    for v1 in HC.V:
        for v2 in v1.nn:
            common_neigh = list(v1.nn.intersection(v2.nn))
            # Only reconnect if the vertices are in a planar region.
            if len(common_neigh) != 2:
                continue
            v3 = common_neigh[0]
            v4 = common_neigh[1]
            # Don't reconnect if both vertices are on the boundary.
            if v3 in bV and v4 in bV:
                continue
            common_neigh = list(v3.nn.intersection(v4.nn))
            if common_neigh != [v1, v2] and common_neigh != [v2, v1]:
                continue
            # Ensure quadrilaterals do not overlap.
            if any([len(set(quad).intersection((v1, v2, v3, v4))) > 2 for quad in to_reconnect]):
                continue
            if np.linalg.norm(v1.x_a - v2.x_a) > 1.5 * np.linalg.norm(v3.x_a - v4.x_a):
                to_reconnect.append((v1, v2, v3, v4))
    if len(to_reconnect) == 0:
        return 0
    for (v1, v2, v3, v4) in to_reconnect:
        v1.disconnect(v2)
        v3.connect(v4)
    return len(to_reconnect)


def outward_normal(vert, meanCurv):
    """
    Compute the outward normal vector at a vertex, ensuring it points away from the z-axis.

    If the surface curvature is nearly zero, the direction from the origin is used.

    Parameters
    ----------
    vert : object
        A vertex object with attributes 'x_a' representing its position.
    meanCurv : array_like
        The mean curvature vector at the vertex.

    Returns
    -------
    numpy.ndarray
        The normalized outward normal vector.
    """
    if np.linalg.norm(meanCurv) > 1e-10:
        N_approx = -normalized(meanCurv)[0]
    else:
        N_approx = normalized(vert.x_a)[0]
    if sum(N_approx[:3] * vert.x_a[:3]) < 0:
        N_approx = -N_approx
    return N_approx


def cross_prod(a, b):
    """
    Compute the cross product of two 3D vectors.

    Parameters
    ----------
    a : array_like
        First 3D vector.
    b : array_like
        Second 3D vector.

    Returns
    -------
    numpy.ndarray
        The cross product vector.
    """
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def get_forces(HC, bV):
    """
    Compute the surface tension and pressure forces on the vertices of the complex.

    The function calculates forces due to surface tension (interface), gas pressure, liquid pressure,
    and also accounts for solid contact forces at the boundaries.

    Parameters
    ----------
    HC : object
        The complex object containing vertices and connectivity.
    bV : set
        Set of boundary vertices.

    Returns
    -------
    tuple
        A tuple containing:
            forceDict : dict
                Dictionary mapping vertex positions to force vectors.
            maxForce : float
                The maximum force magnitude encountered.
    """
    (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
     HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
     Theta_i_cache) = HC_curvatures_sessile(HC, bV, RadFoot, theta_p, printout=0)
    total_bubble_volume, total_bubble_area, bubble_centroid = triangle_prism_volume(HC)
    if total_bubble_volume != total_bubble_volume:
        return -1
    gasPressure = P_0 * (initial_volume / total_bubble_volume - 1)
    forceDict = {}
    posDict = {}
    maxForce = 0.0
    height = 0
    net_interf_force = np.array([0.0, 0.0, 0.0])
    net_liq_force = np.array([0.0, 0.0, 0.0])
    net_gas_force = np.array([0.0, 0.0, 0.0])
    net_solid_force = np.array([0.0, 0.0, 0.0])
    for v in HC.V:
        force = 0.0
        height = max(height, v.x_a[2])
        # Compute boundary movements (boundaries are fixed for now)
        if v in bV:
            # Get boundary sector length and normal, perhaps using b_curvatures.
            for vn in v.nn:
                if vn in bV:
                    cLineLen = np.linalg.norm(v.x_a - vn.x_a)
                    common_neigh = list(v.nn.intersection(vn.nn))
                    if len(common_neigh) == 1:
                        midpoint = 0.5 * v.x_a + 0.5 * vn.x_a
                        pullDirection = midpoint - common_neigh[0].x_a
                        ContactForce = gamma * cLineLen * pullDirection / np.linalg.norm(pullDirection) / 2
                        # Divide by 2 as two vertices per edge.
                        net_solid_force += ContactForce
                    else:
                        print('len common_neigh', len(common_neigh))
        else:
            H = HNdA_i_cache[v.x]
            interf_force = gamma * H
            net_interf_force += interf_force
            if False:
                dualNormal = outward_normal(v, H)
                dualArea = sum(C_ij_cache[v.x])
                gas_force = gasPressure * dualNormal * dualArea
                liquidPressure = P_0 - rho * g * v.x_a[2]
                if liquidPressure < 0:
                    print('bubble is too tall, liquidPressure=', liquidPressure)
                liq_force = -liquidPressure * dualNormal * dualArea
            else:
                gas_force = 0
                liq_force = 0
                for vn1 in v.nn:
                    for vn2 in v.nn:
                        if vn2 in vn1.nn:
                            triArea = 0.5 * cross_prod(vn1.x_a - v.x_a, vn2.x_a - v.x_a)
                            # Set the area vector pointing away from the z-axis.
                            centroid = (v.x_a + vn1.x_a + vn2.x_a) / 3
                            if sum(triArea[:3] * centroid[:3]) < 0:
                                triArea = -triArea
                            # Divide by 2 because vn1 and vn2 can be swapped,
                            # and divide by 3 because each triangle contributes to 3 vertices.
                            gas_force += triArea * gasPressure / 2 / 3
                            liquidPressure = -rho * g * centroid[2]
                            liq_force -= triArea * liquidPressure / 2 / 3
            net_gas_force += gas_force
            net_liq_force += liq_force
            force = interf_force + liq_force + gas_force
            maxForce = max(maxForce, np.linalg.norm(force))
            forceDict[v.x] = force
    print(t, total_bubble_volume, gasPressure, maxForce, height, E_0,
          *net_interf_force, *net_gas_force, *net_liq_force, *net_solid_force, file=vol_txt)
    vol_txt.flush()
    return forceDict, maxForce


def get_force_array(posArray):
    """
    Compute and return the force array corresponding to a given flattened position array.

    Parameters
    ----------
    posArray : array_like
        Flattened array containing positions for vertices (3 values per vertex).

    Returns
    -------
    numpy.ndarray
        The flattened force array computed from the force dictionary.
    """
    HC_temp = HC
    bV_temp = set()
    for i, v in enumerate(HC_temp.V):
        HC_temp.V.move(v, tuple(posArray[i * 3:(i + 1) * 3]))
        if posArray[i * 3 + 2] < RadTop * 1e-4:
            bV_temp.add(v)
    forceDict, maxF = get_forces(HC_temp, bV_temp)
    forArray = np.array([fi for f in forceDict.values() for fi in f])
    return forArray


def get_energy(HC):
    """
    Compute and record the energy of the complex, including contributions from gas,
    interface, and gravity.

    Parameters
    ----------
    HC : object
        The complex object.

    Returns
    -------
    float
        The total energy (interface energy plus gravitational energy).
    """
    total_bubble_volume, total_bubble_area, bubble_centroid = triangle_prism_volume(HC)
    idealGasEn = P_0 * initial_volume * np.log(initial_volume / total_bubble_volume)
    interfaceEn = gamma * total_bubble_area
    gravityEn = -rho * g * bubble_centroid * total_bubble_volume
    fname = 'data/energy.txt'
    with open(fname, "a") as en_txt:
        print(t, idealGasEn, interfaceEn, gravityEn, file=en_txt)
        en_txt.flush()
    return interfaceEn + gravityEn  # idealGasEn +


def correct_the_volume(HC, bV):
    """
    Adjust the positions of non-boundary vertices to correct the bubble volume.

    Parameters
    ----------
    HC : object
        The complex object.
    bV : set
        Set of boundary vertices.
    """
    (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
     HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
     Theta_i_cache) = HC_curvatures_sessile(HC, bV, RadFoot, theta_p, printout=0)
    total_bubble_volume, total_bubble_area, bubble_centroid = triangle_prism_volume(HC)
    shift = (initial_volume - total_bubble_volume) / total_bubble_area
    for v in HC.V:
        if v not in bV:
            H = HNdA_i_cache[v.x]
            # dualNormal = outward_normal(v, H)
            dualNormal = v.x_a / (sum(v.x_a[:] ** 2) ** 0.5)
            HC.V.move(v, tuple(v.x_a + dualNormal * shift))
    return


def spherical_cap_init(RadFoot, theta_p, NFoot=4):
    """
    Initialize a complex in the shape of a spherical cap with a given contact angle.

    Parameters
    ----------
    RadFoot : float
        Radius of the bubble foot.
    theta_p : float
        Contact angle in radians.
    NFoot : int, optional
        Number of vertices on the foot. Default is 4.

    Returns
    -------
    tuple
        HC : The constructed complex.
        bV : set of boundary vertices.
    """
    RadSphere = RadFoot / np.sin(theta_p)
    Cone = []
    nn = []
    Cone.append(np.array([0.0, 0.0, 0.0]))
    nn.append([])
    ind = 0
    # Make cone (contact line circle)
    for phi in np.linspace(0.0, 2 * np.pi, NFoot + 1):
        ind += 1
        Cone.append(np.array([np.sin(phi), np.cos(phi), 1]))
        nn.append([])
        if ind > 0:
            nn[0].append(ind)
            nn[ind].append(0)
            nn[ind].append(ind - 1)
            nn[ind].append((ind + 1) % NFoot)
    # Clean Cone (set near-zero values to zero)
    for f in Cone:
        for i, fx in enumerate(f):
            if abs(fx) < 1e-15:
                f[i] = 0.0
    Cone = np.array(Cone)
    nn[1][1] = ind
    # Construct complex from the cone geometry.
    HC = construct_HC(Cone, nn)
    v0 = HC.V[tuple(Cone[0])]
    # Compute boundary vertices.
    V = set()
    for v in HC.V:
        V.add(v)
    bV = V - set([v0])
    for i in range(20):
        v1 = list(v0.nn)[0]
        if sum((v0.x_a[:] - v1.x_a[:]) ** 2) < (maxEdge / RadFoot) ** 2:
            break
        HC.refine_all_star()  # exclude=bV)
        HC.V.merge_all(cdist=0.01 * minEdge)
    # Move vertices to form a spherical cap.
    theta = lambda z: theta_p * min(2 * z, 0.01 * (z - 1) + 1)
    for v in HC.V:
        thet = theta_p * v.x_a[2]
        phi = np.arctan2(v.x_a[1], v.x_a[0])
        x = RadSphere * np.cos(phi) * np.sin(thet)
        y = RadSphere * np.sin(phi) * np.sin(thet)
        z = RadSphere * np.cos(thet) - RadSphere * np.cos(theta_p)
        if abs(z) < RadSphere * 1e-6:
            z = 0.0
        HC.V.move(v, tuple((x, y, z)))
    # Rebuild boundary vertex set after moving vertices.
    bV = set()
    for v in HC.V:
        if abs(v.x[2]) < RadSphere * 1e-4:
            bV.add(v)
    return HC, bV


def cone_init(RadFoot, Volume, NFoot=4):
    """
    Initialize a complex in the shape of a cone.

    Parameters
    ----------
    RadFoot : float
        The radius of the cone's foot.
    Volume : float
        The volume of the cone.
    NFoot : int, optional
        Number of vertices on the foot. Default is 4.

    Returns
    -------
    tuple
        HC : The constructed complex.
        bV : set of boundary vertices.
    """
    height = 3 * Volume / np.pi / RadFoot**2
    print('height', height)
    Cone = []
    nn = []
    Cone.append(np.array([0.0, 0.0, height]))  # Middle vertex
    nn.append([])
    ind = 0
    # Make cone (contact line circle)
    for phi in np.linspace(0.0, 2 * np.pi, NFoot + 1):
        ind += 1
        Cone.append(np.array([RadFoot * np.sin(phi), RadFoot * np.cos(phi), 0]))
        nn.append([])
        if ind > 0:
            nn[0].append(ind)
            nn[ind].append(0)
            nn[ind].append(ind - 1)
            nn[ind].append((ind + 1) % NFoot)
    # Clean Cone.
    for f in Cone:
        for i, fx in enumerate(f):
            if abs(fx) < 1e-15:
                f[i] = 0.0
    Cone = np.array(Cone)
    # Link the beginning and end of the boundary.
    nn[1][1] = ind
    # Construct complex from the cone geometry.
    HC = construct_HC(Cone, nn)
    v0 = HC.V[tuple(Cone[0])]
    # Compute boundary vertices.
    V = set()
    for v in HC.V:
        V.add(v)
    bV = V - set([v0])
    for i in range(20):
        v1 = list(v0.nn)[0]
        if sum((v0.x_a[:] - v1.x_a[:]) ** 2) < maxEdge**2:
            break
        HC.refine_all_star()  # exclude=bV)
        HC.V.merge_all(cdist=0.01 * minEdge)
    # Move refined vertices to form a circular cone.
    for v in HC.V:
        z = v.x_a[2]
        Rad = RadFoot * (height - z) / height
        phi = np.arctan2(v.x_a[1], v.x_a[0])
        x = Rad * np.cos(phi)
        y = Rad * np.sin(phi)
        HC.V.move(v, tuple((x, y, z)))
    # Rebuild boundary vertex set after moving vertices.
    bV = set()
    for v in HC.V:
        if abs(v.x[2]) < height * 1e-4:
            bV.add(v)
        if sum(abs(v.x_a[:])) < height * 1e-4:
            HC.V.remove(v)
    return HC, bV


# ### Parameters
Bo = 0  # Bond number
P_0 = 101.325e3  # Pa, Ambient pressure
gamma = 72.8e-3  # N/m, surface tension
g = 9.81  # m/s^2, gravitational acceleration
RadTop = 1e-3  # m, Radius of curvature of bubble top
rho = Bo * gamma / g / RadTop**2  # kg/m^3, density (e.g., 998.2071 - 1.225 for STP)
print('rho', rho)
theta_p = np.pi / 2  # contact angle

d = 0.0001
psi = 0
r = 0
z = 0
Volume = 0
fname = 'data/adams' + str(Bo) + '.txt'
with open(fname, "w") as adams_txt:
    print('saving', fname)
    for i in range(int(4 / d)):
        r += d * np.cos(psi)
        dz = d * np.sin(psi)
        z += dz
        Volume += np.pi * r**2 * dz
        if i * d * 100 % 1 == 0:
            print(r * RadTop, -z * RadTop, file=adams_txt)
        psi += d * (2 - Bo * z - np.sin(psi) / r)
        if psi > np.pi / 2:
            break
        if psi > np.pi:
            break
        if psi < np.pi / 2 and 2 - Bo * z - np.sin(psi) / r < 0:
            break

# Define parameters tuple used in the solver.
Volume = Volume * RadTop**3
RadFoot = r * RadTop
minEdge = 0.1 * RadFoot
maxEdge = 2 * minEdge
maxMove = 0.5 * minEdge**2 / RadFoot
print(f'RadFoot = {RadFoot}')
tInit = 1500
t = tInit
if tInit == 0:
    HC, bV = cone_init(RadFoot, Volume, NFoot=6)
    # HC, bV = spherical_cap_init(RadFoot, theta_p, NFoot=6)
else:
    HC, bV = load_complex(t)
print('vol area centroid', triangle_prism_volume(HC))
plot_polyscope(HC)
pastE = [np.nan] * 10
prevE = [np.nan] * 10
constMove = False
initial_volume = Volume
fname = 'data/vol.txt'
with open(fname, "a") as vol_txt:
    print('saving', fname)
    # Surface energy minimisation
    while t <= tInit + 300:
        if t % 10 == 0:
            save_vert_positions(t)
        E_0 = get_energy(HC)
        print('t', t, 'nVerts', len(list(HC.V)), 'maxMove', maxMove)
        t += 1
        if constMove:
            HC.V.merge_nn(cdist=minEdge, exclude=bV)
            refine_edges(HC, maxEdge)
            refine_boundaries(HC, bV, maxEdge)
            reconnect_long_diagonals(HC, bV)
            for v in HC.V:
                if len(v.nn) < 2:
                    print('v.nn', v.nn)
                    print('remove', v.x_a)
                    HC.V.remove(v)
            if False:  # sum(prevE)/len(prevE) < sum(pastE)/len(pastE)
                maxMove *= 0.5
                E_0 = np.nan
            pastE.append(E_0)
            prevE.append(pastE.pop(0))
            prevE.pop(0)
        forceDict, maxForce = get_forces(HC, bV)
        if constMove:
            alpha = maxMove / maxForce
        else:
            alpha = 1e-1  # /gamma
            maxMove = alpha * maxForce
        for v in HC.V:
            if v.x in forceDict:
                normFor = alpha * forceDict[v.x]
                HC.V.move(v, tuple(v.x_a + normFor))
        if t % 10 == 0:
            save_vert_positions(str(t) + 'uncor')
    # plot_polyscope(HC) is assumed to open a window via polyscope
plot_polyscope(HC)
plt.show()
