"""
Note that the actual working init seems to be cube_to_tube and the older functions are possibly not needed
"""


import numpy as np
import math
import copy
import os
import sys
import polyscope as ps



# Add base directory (~/ddg/ddgclib/) to sys.path from notebook location
#module_path = os.path.abspath(os.path.join('../..'))
#if module_path not in sys.path:
#    sys.path.append(module_path)

from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ddg imports (from ddgclib package)
from ddgclib import *
from hyperct import Complex
from ddgclib._curvatures import *
from ddgclib._capillary_rise_flow import *
from ddgclib._capillary_rise import *
from ddgclib._eos import *
from ddgclib._misc import *
from ddgclib._plotting import *

# barycentric imports (from ddgclib.barycentric subpackage)
from ddgclib.barycentric._duals import compute_vd, triang_dual, _set_boundary
from ddgclib.visualization.plot_dual import plot_dual_mesh_2D, plot_dual_mesh_3D
from ddgclib._plotting import plot_dual

# compute duals (and Delaunay dual from a set of points)
from ddgclib.barycentric._duals import compute_vd, _merge_local_duals_vector, triang_dual, plot_dual

# Plots
from ddgclib.barycentric._duals import  plot_dual_mesh_2D

# Geometry and dual computations
from ddgclib.barycentric._duals import area_of_polygon, e_star, volume_of_geometric_object, plot_dual, v_star

# Boundary geometry
from ddgclib.barycentric._duals import  _set_boundary, _find_plane_equation, _find_intersection, _reflect_vertex_over_edge

# Area computations
from ddgclib.barycentric._duals import d_area

# Volume computations (including helper functions)
from ddgclib.barycentric._duals import _signed_volume_parallelepiped, _volume_parallelepiped
# DDG gradient operations on primary edges (for continuum)
from ddgclib.barycentric._duals import dP, du, dudt



def run_cap_rise_simulation(refinements, r, height, h_jurin, params=None):
    """
    Single function to run the capillary rise simulation for a given refinement level.

    Parameters:
    - refinements: int, the refinement level (1, 2, or 3)
    - r: float, radius (assumed defined externally)
    - height: float, height (assumed defined externally)
    - h_jurin: float, Jurin height (assumed defined externally)
    - params: dict, optional parameters per refinement. If None, uses default.

    Default params include t_final, screenshot_dir, and look_at positions.
    Stability parameters (cdist, merge tol, movement tolerances, speeds) are fixed
    as they are essential and identical across refinements in the provided code.
    """
    # Default parameters per refinement (varying ones)
    if params is None:
        length_scale = 1e-3
        default_params = {
            1: {
                't_final': 10,
                'screenshot_dir': './fig/cap_rise/sparse_short/',
                'initial_look_at': (0., -10. * length_scale, 2. * length_scale),
                'loop_look_at': (0., -6. * length_scale, 2. * length_scale)
            },
            2: {
                't_final': 10,
                'screenshot_dir': './fig/cap_rise/medium_short/',
                'initial_look_at': (0., -10. * length_scale, 2. * length_scale),
                'loop_look_at': (0., -6. * length_scale, 2. * length_scale)
            },
            3: {
                't_final': 35,
                'screenshot_dir': './fig/cap_rise/fine_short/',
                'initial_look_at': (0., -10. * length_scale, 2. * length_scale),
                'loop_look_at': (0., -6. * length_scale, 2. * length_scale)
            }
        }
        p = default_params[refinements]
    else:
        p = params[refinements]
        length_scale = p.get('length_scale', 1e-3)  # Allow override if needed for stability

    t_final = p['t_final']
    screenshot_dir = p['screenshot_dir']

    # Stability parameters (fixed across refinements; can be overridden in params if needed)
    init_cdist = 1e-8
    recompute_cdist = 1e-7
    merge_tol = 1e-12
    boundary_tol = 1e-10
    interior_d_add = 1e-4
    boundary_speed = 0.2e-3
    interior_speed = 0.1e-3
    dt = 0.05
    initial_plot_scale = 1e3
    point_radii = 1e-5

    # Initiate refinement and compute initial duals
    HC = cube_to_tube(r, refinements=refinements, height=height)
    compute_vd(HC, cdist=init_cdist)

    # Select a vertex to monitor
    vi = HC.V[(0.0, 0.0, 0.0)]

    # Initial plot
    ps.init()
    plot_dual(vi, HC, length_scale=initial_plot_scale, point_radii=point_radii)
    ps.look_at(p['initial_look_at'], (0., 0., 0.))

    # Dynamic loop
    t = 0
    i = 0
    menv = False
    while t < t_final:
        # Move all vertices on boundary
        for v in HC.V:
            if t < 0.2:
                if v.x_a[2] < 0.0:
                    continue
            if v.x_a.all() == vi.x_a.all():
                menv = True

            # Distance from centre
            v_xz = np.array([v.x_a[0], v.x_a[1]])
            d = np.linalg.norm(v_xz - np.array([0, 0]))
            if np.abs(d - r) < boundary_tol:
                vt_new = v.x_a.copy()
                vt_new[2] += boundary_speed * (h_jurin - vt_new[2])
                vt_new = tuple(vt_new)
                v_new = HC.V.move(v, vt_new)
            if menv:
                # vi = v_new  # TODO: needed? (kept commented as in original)
                menv = False

        # Move special monitored vertex (disabled as in original)
        if 0:
            vt_new = vi.x_a.copy()
            vt_new[2] += 1e-5
            vt_new = tuple(vt_new)
            v_new = HC.V.move(vi, vt_new)
            vi = v_new

        # Move interior vertices
        if t > 1:
            for v in HC.V:
                if t < 0.2:
                    if v.x_a[2] < 0.0:
                        continue
                # Distance from centre
                v_xz = np.array([v.x_a[0], v.x_a[1]])
                d = np.linalg.norm(v_xz - np.array([0, 0]))
                if np.abs(d - r) < boundary_tol:
                    continue
                d += interior_d_add
                vt_new = v.x_a.copy()
                vt_new[2] += interior_speed * (h_jurin - vt_new[2]) * (d / r)
                vt_new = tuple(vt_new)
                v_new = HC.V.move(v, vt_new)

        # Postprocessing
        HC.V.merge_all(merge_tol)
        # Clear the dual cache
        for v in HC.V:
            v.vd = set()

        HcVd = copy.copy(HC.Vd)
        for vd in HcVd:
            HC.Vd.cache.pop(vd.x)

        # Recompute the dual
        compute_vd(HC, cdist=recompute_cdist)

        # Plot in polyscope and update the frame
        plot_dual(vi, HC, length_scale=length_scale, point_radii=point_radii)
        ps.look_at(p['loop_look_at'], (0., 0., 0.))
        ps.frame_tick()  # renders one UI frame, returns immediately
        ps.screenshot(f"{screenshot_dir}cap_rise_{i}.png")

        # Update the time step
        t += dt
        i += 1
        print(f't = {t}')
        if t > t_final:  # Break the while loop at the end
            ps.window_requests_close()
            break


def cube_to_tube(r, refinements=1, height=5e-3):
    # Construct the initial cube
    lb = -0.5
    ub = 0.5
    domain = [(lb, ub), ] * 3
    # symmetry = [0, 1, 1]
    HC = Complex(3, domain=domain, symmetry=None)
    HC.triangulate()
    for i in range(refinements):
        HC.refine_all()

    # NEW
    # Compute boundaries
    bV = set()
    for v in HC.V:
        if ((v.x_a[0] == lb or v.x_a[1] == lb or v.x_a[2] == lb) or
                (v.x_a[0] == ub or v.x_a[1] == ub or v.x_a[2] == ub)):
            bV.add(v)

    # boundaries which exclude the interior vertices on the top/bottom of
    # the cyllinder
    bV_sides = set()
    for v in HC.V:
        # if ((v.x_a[0] == lb or v.x_a[0] == ub) and
        #    (v.x_a[1] == lb or v.x_a[1] == ub)):
        if ((v.x_a[0] == lb or v.x_a[0] == ub) or
                (v.x_a[1] == lb or v.x_a[1] == ub)):
            bV_sides.add(v)
            # Special side boundary property
            v.side_boundary = True
        else:
            v.side_boundary = False
    # print(f'bV_sides = {bV_sides}')

    for bv in bV:
        _set_boundary(bv, True)
    for v in HC.V:
        if not (v in bV):
            _set_boundary(v, False)

    # for bv in bV:
    #    print(f'bv = {bv.x}')
    # Move the vertices to the tube radius
    for v in bV_sides:
        r_eff = r  # Trancated radius projection
        nv = np.zeros(3)
        theta = math.atan2(v.x_a[1], v.x_a[0])
        nv[0] = r_eff * np.cos(theta)
        nv[1] = r_eff * np.sin(theta)
        if 1:
            h_eff = height  # * ((1 - np.cos(np.pi * v.x[2]**0.5)) / 2)
        nv[2] = h_eff * (v.x[2] - 0.5)  # * 1e-3
        print(f'nv[2] = {nv[2]}')
        HC.V.move(v, tuple(nv))

    for v in HC.V:
        # if v.boundary:
        if v.side_boundary:
            continue
        d = np.linalg.norm(v.x_a[:2])  # This is already a normalized distance for 0.5 bounds
        # Power law scaling:
        if 0:
            n = 0.5  # 0.6  # Aribtrarily chosen power law scaling, should be n<=r
            r_eff = d ** n * r  # Trancated radius projection
        # log law scaling:
        if 0:
            r_eff = r * (np.log(d + 1) / np.log(2))
        # Sinusoidal scaling:
        if 1:
            r_eff = (r * ((1 - np.cos(np.pi * d ** 0.5)) / 2))

        # r_eff = r/d  # Trancated radius projection
        nv = np.zeros(3)
        theta = math.atan2(v.x_a[1], v.x_a[0])
        nv[0] = r_eff * np.cos(theta)
        nv[1] = r_eff * np.sin(theta)
        # Height scaling
        if 1:
            h_eff = height  # * ((1 - np.cos(np.pi * v.x[2]**0.5)) / 2)

        nv[2] = h_eff * (v.x[2] - 0.5)  # * 1e-3
        print(f'nv[2] = {nv[2]}')
        HC.V.move(v, tuple(nv))

    # print(f'post move:')
    # for bv in bV:
    #    print(f'bv = {bv.x}')
    return HC


def _cap_rise_meniscus_init(r, theta_i, gamma, N=4, refinement=0,
                            cdist=1e-10, equilibrium=True):
    """
    Helper function to generate the initial film
    :param r:
    :param theta_i:
    :param gamma:
    :param N:
    :param refinement:
    :return:
    """
    Theta = np.linspace(0.0, 2 * np.pi, N)  # range of theta
    R = r / np.cos(theta_i)  # = R at theta = 0
    # Exact values:
    K_f = (1 / R) ** 2
    H_f = 1 / R + 1 / R  # 2 / R
    dp_exact = gamma * (2 / R)  # Pa      # Young-Laplace equation  dp = - gamma * H_f = - gamma * (1/R1 + 1/R2)
    F = []
    nn = []
    F.append(np.array([0.0, 0.0, R * np.sin(theta_i) - R]))
    nn.append([])
    ind = 0
    for theta in Theta:
        ind += 1
        # Define coordinates:
        # x, y, z = sphere(R, theta, phi)
        F.append(np.array([r * np.sin(theta), r * np.cos(theta), 0.0]))
        # Define connections:
        nn.append([])
        if ind > 0:
            nn[0].append(ind)
            nn[ind].append(0)
            nn[ind].append(ind - 1)
            nn[ind].append((ind + 1) % N)

    # clean F
    for f in F:
        for i, fx in enumerate(f):
            if abs(fx) < 1e-15:
                f[i] = 0.0

    F = np.array(F)
    nn[1][1] = ind

    # Construct complex from the initial geometry:
    HC = construct_HC(F, nn)
    v0 = HC.V[tuple(F[0])]
    # Compute boundary vertices
    V = set()
    for v in HC.V:
        V.add(v)
    bV = V - set([v0])
    for i in range(refinement):
        V = set()
        for v in HC.V:
            V.add(v)
        HC.refine_all_star(exclude=bV)
        # New boundary vertices:
        for v in HC.V:
            if v.x[2] == 0.0:
                bV.add(v)

    # Move to spherical cap
    for v in HC.V:
        z = v.x_a[2]
        z_sphere = z - R * np.sin(theta_i)  # move to origin
        # z_sphere = R * np.cos(phi)  # For a sphere centered at origin
        phi_v = np.arccos(z_sphere / R)
        plane_dist = R * np.sin(phi_v)
        # Push vertices on the z-slice the required distance
        z_axis = np.array([0.0, 0.0, z])  # axial centre
        vec = v.x_a - z_axis
        s = np.abs(np.linalg.norm(vec) - plane_dist)
        nvec = normalized(vec)[0]
        nvec = v.x_a + s * nvec
        HC.V.move(v, tuple(nvec))
        vec = nvec - z_axis
        np.linalg.norm(vec)

    # Rebuild set after moved vertices (appears to be needed)
    bV = set()
    for v in HC.V:
        if v.x[2] == 0.0:
            bV.add(v)

    if not equilibrium:
        # Move to zero, for mean flow simulations
        VA = []
        for v in HC.V:
            if v in bV:
                continue
            else:
                VA.append(v.x_a)

        VA = np.array(VA)
        for i, v_a in enumerate(VA):
            v = HC.V[tuple(v_a)]
            v_new = tuple(v.x_a - np.array([0.0, 0.0, v.x_a[2]]))
            HC.V.move(v, v_new)

    return F, nn, HC, bV, K_f, H_f


def cap_rise_init_dyn(r, theta_i, gamma, N=4, refinement=0, depth_dist=0.06, depth_ref=3, cdist=1e-12,
                      equilibrium=True):
    """

    :param r:
    :param theta_i: Initial angle
    :param gamma:
    :param N:
    :param refinement:
    :param cdist:
    :param equilibrium:
    :return:
    """
    F, nn, HC, bV, K_f, H_f = _cap_rise_meniscus_init(r, theta_i, gamma, N, refinement, cdist=1e-12, equilibrium=False)

    # Save the points and boundary points
    V_points = []
    bV_points = []
    for v in HC.V:
        V_points.append(v.x_a)

    for bv in bV:
        print(f'bv.x_a = {bv.x_a}')
        bV_points.append(bv.x_a)

    V_points_film = np.array(V_points)
    bV_points_film = np.array(bV_points)
    print(f'V_points_film.shape = {V_points_film.shape}')
    print(f'bV_points_film.shape = {bV_points_film.shape}')
    # Extend the meniscus film down to a tube below the water line:
    dx = depth_dist / depth_ref
    h = 0  # Height layer tracker
    for i in range(depth_ref):
        # Add new points
        for v in V_points_film:
            v_new = copy.copy(v)
            v_new[2] = v_new[2] + h
            V_points.append(v_new)

        # Track new boundary points
        for bv in bV_points_film:
            bv_new = copy.copy(bv)
            bv_new[2] = bv_new[2] + h
            bV_points.append(bv_new)

        h -= dx  # Update next layer height

    V_points = np.array(V_points)
    bV_points_a = np.array(bV_points)

    # Reconstruct the complex using these values:
    HC = Complex(3)
    for va in V_points:
        v = HC.V[tuple(va)]
        # Set default boundary off
        v.boundary = False

    tri = Delaunay(V_points)
    plot_complex_3d_mat(V_points)
    for t in tri.simplices:
        for v1i in t:
            v1a = V_points[v1i]
            v1 = HC.V[tuple(v1a)]
            for v2i in t:
                v2a = V_points[v2i]
                v2 = HC.V[tuple(v2a)]
                v1.connect(v2)

    # Add boundary conditions:
    if 0:
        for vba in bV_points_a:
            vb = HC.V[tuple(vba)]
            vb.boundary = True

        # All the top and bottom points must also be boundary points (in the future use boundary function):

        h += dx  # Reset height to last layer in loop
        for v in V_points_film:
            v_new = copy.copy(v)
            v1 = HC.V[tuple(v_new)]
            v1.boundary = True
            v_new[2] = v_new[2] + h
            v2 = HC.V[tuple(v_new)]
            v2.boundary = True
            bV_points.append(v1.x_a)
            bV_points.append(v2.x_a)

        bV_points = np.array(bV_points)
        print(f'V_points.shape = {V_points.shape}')
        print(f'bV_points.shape = {bV_points.shape}')

    # Clean up:
    HC.V.merge_all(cdist)

    bV = set()
    for vba in bV_points:
        vb = HC.V[tuple(vba)]
        bV.add(vb)
    print(f'bV = {bV}')
    return HC, bV


