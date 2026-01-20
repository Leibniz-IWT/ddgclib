import copy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import polyscope as ps
from interruptingcow import timeout

# ddg imports
from ddgclib import *
from ddgclib._complex import *
from ddgclib._curvatures import *
from ddgclib._capillary_rise_flow import *
from ddgclib._eos import *
from ddgclib._misc import *
from ddgclib._plotting import *


# -----------------------------
# FLAGS (toggle features here)
# -----------------------------
RUN_SIMULATION = False       # run time evolution loop
PLOT_MATPLOTLIB = False      # plot using matplotlib
PLOT_PLOTLY = False          # plot using plotly
PLOT_POLYSCOPE = True        # plot using polyscope
CHECK_CURVATURE = True       # compute curvature and print error


# -----------------------------
# Physical constants (water in air at 20 deg C)
# -----------------------------
gamma = 0.0728  # N/m, surface tension
rho   = 1000   # kg/m3, density
g     = 9.81   # m/s2, gravity


# -----------------------------
# Geometry & initial conditions
# -----------------------------
r = 1.4e-5                 # tube radius (m)
h = 0.0130                 # initial film height
theta_p = 20 * np.pi/180.0 # contact angle (radians)

# Discrete mesh parameters
N = 5
refinement = 0
equilibrium = True
cdist = 1e-10

# Convert to long double for numerical accuracy
r = np.array(r, dtype=np.longdouble)
theta_p = np.array(theta_p, dtype=np.longdouble)


# -----------------------------
# Core numerical update
# -----------------------------
def mean_flow(HC, bV, params, tau, print_out=False):
    """
    Move vertices according to surface tension + gravity forces.

    - Interior vertices: move in z direction based on mean curvature and hydrostatic pressure.
    - Boundary vertices: apply force from contact angle mismatch (approximate capillary force).
    """
    (gamma, rho, g, r, theta_p, K_f, h) = params

    # Compute curvature quantities on the mesh
    (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
     HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache,
     HNdA_i_Cij_cache, Theta_i_cache) = HC_curvatures(HC, bV, r, theta_p, printout=0)

    bV_new = set()

    for v in HC.V:
        if v in bV:
            # ---- boundary vertex update ----
            # Compute a simple capillary force based on geodesic curvature
            R_approx = 1 / np.sqrt(K_H_i_cache[v.x])
            theta_p_approx = np.arccos(np.min([r / R_approx, 1]))
            h = R_approx - r * np.tan(theta_p_approx)
            A_approx = 2 * np.pi * R_approx * h

            Xi = 1  # topological Euler characteristic for a disk
            kg_ds = 2 * np.pi * Xi - K_H_i_cache[v.x] * (A_approx)
            ds = 2 * np.pi * r
            k_g = kg_ds / ds

            phi_est = np.arctan(R_approx * k_g)

            gamma_bt = gamma * (np.cos(phi_est) - np.cos(theta_p)) * np.array([0, 0, 1.0])
            l_a = 2 * np.pi * r / len(bV)
            F_bt = gamma_bt * l_a

            new_vx = v.x + 1e-1 * F_bt
            new_vx = tuple(new_vx)

            HC.V.move(v, new_vx)
            bV_new.add(HC.V[new_vx])

        else:
            # ---- interior vertex update ----
            H = HN_i_cache[v.x]

            # Hydrostatic pressure term
            height = np.max([v.x_a[2], 0.0])
            df = gamma * H - (rho * g * height)

            f_k = v.x_a + np.array([0, 0, tau * df])
            f_k[2] = np.max([f_k[2], 0.0])  # do not go below z=0

            new_vx = tuple(f_k)
            HC.V.move(v, new_vx)

    return HC, bV_new


def incr(HC, bV, params, tau=1e-5, plot=False, verbosity=1):
    """
    Single time-step update.
    """
    HC.dim = 3
    print_out = (verbosity == 2)

    HC, bV = mean_flow(HC, bV, params, tau=tau, print_out=print_out)

    if verbosity == 1:
        current_Jurin_err(HC)

    if plot:
        pass  # placeholder for polyscope update

    return HC, bV


def current_Jurin_err(HC):
    """
    Compute min height vs Jurin height.
    """
    h_final = np.min([v.x_a[2] for v in HC.V])
    print(f'h_final = {h_final} m')
    print(f'h (Jurin) = {h_jurin} m')
    print(f'h_final - h (Jurin) = {h_final - h_jurin} m')

    try:
        print(f'Error: {100 * abs((h_final - h_jurin)/ h_jurin)} %')
    except ZeroDivisionError:
        pass


def plot_polyscope(HC):
    """
    Visualize the surface in Polyscope.
    """
    ps.init()
    ps.set_up_dir("z_up")

    HC.dim = 2
    HC.vertex_face_mesh()

    points = np.array(HC.vertices_fm)
    triangles = np.array(HC.simplices_fm_i)

    # Register point cloud + mesh
    ps_cloud = ps.register_point_cloud("my points", points)
    surface = ps.register_surface_mesh("my mesh", points, triangles,
                                       color=(0.5, 0.5, 0.5),
                                       edge_width=1.0,
                                       edge_color=(0.0, 0.0, 0.0),
                                       smooth_shade=False)

    ps.show()


# -----------------------------
# INITIALIZATION
# -----------------------------
# Analytical curvature for a spherical cap
H_f, K_f, dA, k_g_f, dC = analytical_cap(r, theta_p)
h_jurin = 2 * gamma * np.cos(theta_p) / (rho * g * r)

# Create initial mesh for capillary rise
F, nn, HC, bV, K_f, H_f = cap_rise_init_N(r, theta_p, gamma, N=N,
                                          refinement=refinement,
                                          equilibrium=equilibrium)

params = (gamma, rho, g, r, theta_p, K_f, h)
HC.V.merge_all(cdist=cdist)


# -----------------------------
# MAIN ACTIONS (controlled by flags)
# -----------------------------
if RUN_SIMULATION:
    steps = 800
    for i in range(steps):
        HC, bV = incr(HC, bV, params, tau=0.1, plot=False)


if PLOT_MATPLOTLIB:
    HC.plot_complex()
    plt.show()


if PLOT_PLOTLY:
    import plotly.graph_objects as go
    HC.dim = 2
    HC.vertex_face_mesh()
    points = np.array(HC.vertices_fm)
    triangles = np.array(HC.simplices_fm_i)

    fig = go.Figure(data=[
        go.Mesh3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
        )
    ])
    fig.show()


if PLOT_POLYSCOPE:
    plot_polyscope(HC)


if CHECK_CURVATURE:
    (HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i, HNdA_ij_dot_hnda_i,
     K_H_2, HNdA_i_Cij) = int_curvatures(HC, bV, r, theta_p, printout=0)
