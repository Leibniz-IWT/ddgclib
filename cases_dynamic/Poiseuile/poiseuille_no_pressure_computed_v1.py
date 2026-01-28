# compared to version 0, dual_edge_flux is now barycentric-aware, used in mass_imbalance and mass divergence is bounded
# In the future, we need pressure correction to avoid mass divergence


import numpy as np
from scipy.spatial import Delaunay
import os, sys
module_path = os.path.abspath(os.path.join('../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from ddgclib._complex import *
from ddgclib._sphere import *

def distance(a, b):
    return np.linalg.norm(a - b)

def barycenter(vertices):
    A, B, C = vertices
    return (A + B + C) / 3

def compute_vd(HC, cdist=1e-10):
    """
    Computes the dual vertices of a primal vertex cache HC.V on
    each dim - 1 simplex.

    Currently only dim = 2 is supported

    cdist: float, tolerance for where a unique dual vertex can exist

    """
    # Construct dual cache
    HC.Vd = VertexCacheField()

    # Construct dual neighbour sets
    for v in HC.V:
        v.vd = set()

    for v1 in HC.V:
        for v2 in v1.nn:
            # Find all v2.nn also connected to v1:
            v1nn_u_v2nn = v1.nn.intersection(v2.nn)
            for v3 in v1nn_u_v2nn:
                verts = np.zeros([3, 2])
                verts[0] = v1.x_a
                verts[1] = v2.x_a
                verts[2] = v3.x_a
                # Compute the barycenter:
                cd = barycenter(verts)
                # Check for uniqueness:
                for vd_i in HC.Vd:
                    dist = np.linalg.norm(vd_i.x_a - cd)
                    if dist < cdist:
                        cd = vd_i.x_a

                vd = HC.Vd[tuple(cd)]
                # Connect to all primal vertices
                for v in [v1, v2, v3]:
                    v.vd.add(vd)
                    vd.nn.add(v)

    return HC

def triang_dual(points, plot_delaunay=False):
    """
    Compute the Delaunay triangulation plus the dual points. Put into hyperct complex object.
    """
    tri = Delaunay(points)
    if plot_delaunay:
        import matplotlib.pyplot as plt
        plt.triplot(points[:,0], points[:,1], tri.simplices)
        plt.plot(points[:,0], points[:,1], 'o')
        plt.show()

    HC = Complex(2)
    for s in tri.simplices:
        for v1i in s:
            for v2i in s:
                if v1i == v2i:
                    continue
                else:
                    v1 = tuple(points[v1i])
                    v2 = tuple(points[v2i])
                    HC.V[v1].connect(HC.V[v2])

    return HC, tri

def plot_dual_mesh_2D(HC, tri):
    """
    Plot the dual mesh and show edge connectivity. Blue is the primary mesh. Orange is the dual mesh.
    """
    import matplotlib.pyplot as plt

    dual_points = []
    for vd in HC.Vd:
        dual_points.append(vd.x_a)
    dual_points = np.array(dual_points)

    for v in HC.V:
        for v2 in v.nn:
            v1vdv2vd = v.vd.intersection(v2.vd)
            if len(v1vdv2vd) == 1:
                continue
            v1vdv2vd = list(v1vdv2vd)
            x = [v1vdv2vd[0].x[0], v1vdv2vd[1].x[0]]
            y = [v1vdv2vd[0].x[1], v1vdv2vd[1].x[1]]
            plt.plot(x, y, color='orange')

        for vd in v.vd:
            x = [v.x[0], vd.x[0]]
            y = [v.x[1], vd.x[1]]
    plt.triplot(points[:,0], points[:,1], tri.simplices, color='tab:blue')
    plt.plot(points[:,0], points[:,1], 'o', color='tab:blue')
    plt.plot(dual_points[:,0], dual_points[:,1], 'o', color='tab:orange')

    plt.show()

def get_incident_triangles(vp1, HC):
    triangles = []
    visited = set()
    for vp2 in vp1.nn:
        for vp3 in vp1.nn.intersection(vp2.nn):
            sorted_ids = sorted([id(vp1), id(vp2), id(vp3)])
            tri_key = tuple(sorted_ids)
            if tri_key not in visited:
                visited.add(tri_key)
                triangles.append((vp1, vp2, vp3))
    return triangles

def d_area(vp1):
    """
    Compute the barycentric dual area exactly as sum (area_tri / 3) over incident triangles.
    """
    area = 0.0
    for tri in get_incident_triangles(vp1, HC):
        A = tri[0].x_a
        B = tri[1].x_a
        C = tri[2].x_a
        area_tri = 0.5 * np.abs((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1]))
        area += area_tri / 3.0
    return area

def incom_Poi(domain, refinements=2):
    """
    Compute the triangulate of a 2D incompressible Poiseuile flow
    """
    HC = Complex(2, domain)
    HC.triangulate()
    for i in range(refinements):
        HC.refine_all()

    points = []
    for v in HC.V:
        points.append(v.x_a)
    points = np.array(points)
    tri = Delaunay(points)
    return points

def u_x_analytical(y):
    return 4 * y * (1 - y)

def u_plane_analytical(x):
    y = x[1]
    return (G / (2 * mu)) * y * (h - y)

def v_error(HC):
    MSE = 0
    N = 0
    for v in HC.V:
        if is_boundary(v):
            continue  # Skip boundaries as they are fixed
        u_anal = u_plane_analytical(v.x_a)
        MSE += (v.u[0] - u_anal)**2
        N += 1
    return MSE / N if N > 0 else 0

def set_zero_vel(HC):
    for v in HC.V:
        v.u = np.array([0.0, 0.0])

def mass_IC(HC):
    for v in HC.V:
        area = d_area(v)
        V = area * 1
        v.m = rho * V

def plot_field(p, u, xlim_lb=-1, xlim_ub=10, scale=1):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    positions = p
    velocities = u

    x, y = positions[:, 0], positions[:, 1]
    u_v, v_v = velocities[:, 0], velocities[:, 1]

    magnitude = np.sqrt(u_v**2 + v_v**2)
    norm = Normalize(vmin=magnitude.min(), vmax=magnitude.max())

    plt.figure(figsize=(8, 8))
    quiver_plot = plt.quiver(x, y, u_v, v_v, magnitude, norm=norm, cmap='viridis', scale=scale)
    plt.scatter(x, y, s=0.1)
    cbar = plt.colorbar(quiver_plot)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label('Magnitude of Velocity (m/s)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.xlim(xlim_lb, xlim_ub)
    plt.ylim(0, 1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.show()

def plot_discrete_field(p, u, HC, tri, xlim_lb=0, xlim_ub=1, scale=1e-6, save=None, title=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    dual_points = []
    for vd in HC.Vd:
        dual_points.append(vd.x_a)
    dual_points = np.array(dual_points)

    fig, ax = plt.subplots(figsize=(8, 8))
    for v in HC.V:
        for v2 in v.nn:
            v1vdv2vd = v.vd.intersection(v2.vd)
            if len(v1vdv2vd) == 1:
                continue
            v1vdv2vd = list(v1vdv2vd)
            x = [v1vdv2vd[0].x[0], v1vdv2vd[1].x[0]]
            y = [v1vdv2vd[0].x[1], v1vdv2vd[1].x[1]]
            ax.plot(x, y, color='orange')

        for vd in v.vd:
            x = [v.x[0], vd.x[0]]
            y = [v.x[1], vd.x[1]]
    ax.triplot(points[:,0], points[:,1], tri.simplices, color='tab:blue')
    ax.plot(points[:,0], points[:,1], 'o', color='tab:blue')
    ax.plot(dual_points[:,0], dual_points[:,1], 'o', color='tab:orange')

    positions = p
    velocities = u

    x, y = positions[:, 0], positions[:, 1]
    u_v, v_v = velocities[:, 0], velocities[:, 1]

    magnitude = np.sqrt(u_v**2 + v_v**2)
    norm = Normalize(vmin=magnitude.min(), vmax=magnitude.max())

    quiver_plot = ax.quiver(x, y, u_v, v_v, magnitude, norm=norm, cmap='viridis', scale=scale)
    ax.scatter(x, y, s=0.1)
    cbar = plt.colorbar(quiver_plot)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label('Magnitude of Velocity (m/s)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(xlim_lb, xlim_ub)
    ax.set_ylim(0, 1)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if title:
        ax.set_title(title)
    if save:
        plt.savefig(save)
    plt.show()

def plot_velocity_profile(HC, save=None, title=None):
    import matplotlib.pyplot as plt
    y_num = []
    u_num = []
    for v in HC.V:
        y_num.append(v.x_a[1])
        u_num.append(v.u[0])

    # Sort for plotting
    idx = np.argsort(y_num)
    y_num = np.array(y_num)[idx]
    u_num = np.array(u_num)[idx]

    y_anal = np.linspace(0, h, 100)
    u_anal = (G / (2 * mu)) * y_anal * (h - y_anal)

    plt.figure(figsize=(6, 6))
    plt.plot(u_num, y_num, 'o', label='Numerical')
    plt.plot(u_anal, y_anal, '-', label='Analytical')
    plt.xlabel('u_x (m/s)')
    plt.ylabel('y (m)')
    plt.legend()
    if title:
        plt.title(title)
    if save:
        plt.savefig(save)
    plt.show()

def min_edge_length(HC):
    min_d = np.inf
    seen = set()
    for v in HC.V:
        for nn in v.nn:
            key = frozenset([id(v), id(nn)])
            if key in seen:
                continue
            seen.add(key)
            d = np.linalg.norm(v.x_a - nn.x_a)
            if d < min_d:
                min_d = d
    return min_d

def is_boundary(v, y_lb=0, y_ub=1, eps=1e-8):
    if abs(v.x_a[1] - y_lb) < eps or abs(v.x_a[1] - y_ub) < eps:
        return True
    return False

def get_cotan_weight(v1, v2, HC):
    common = v1.nn.intersection(v2.nn)
    cot_sum = 0.0
    for v3 in common:
        a = v1.x_a - v3.x_a
        b = v2.x_a - v3.x_a
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            continue
        dot = np.dot(a, b)
        cross = a[0]*b[1] - a[1]*b[0]
        cos_theta = dot / (norm_a * norm_b)
        sin_theta = abs(cross) / (norm_a * norm_b)
        if sin_theta < 1e-12:
            continue  # degenerate
        cot_theta = cos_theta / sin_theta
        cot_sum += cot_theta
    return cot_sum / 2.0

def compute_lap(HC, field='u', comp=0):
    lap_dict = {}
    for v in HC.V:
        a_v = d_area(v)
        if a_v < 1e-12:
            lap_dict[v] = 0.0
            continue
        sum_term = 0.0
        for nn in v.nn:
            w = get_cotan_weight(v, nn, HC)
            u_v = v.__dict__[field][comp]
            u_nn = nn.__dict__[field][comp]
            sum_term += w * (u_nn - u_v)
        lap_v = sum_term / a_v
        lap_dict[v] = lap_v
    return lap_dict

def compute_residual(HC):
    res = 0.0
    N = 0
    for v in HC.V:
        if is_boundary(v):
            continue
        res += np.linalg.norm(v.du)
        N += 1
    return res / N if N > 0 else 0

def dual_edge_and_normal(v, vn):
    """
    Returns:
    - dual edge length |e*|
    - unit normal n_ij (oriented)
    """
    # dual vertices shared by v and vn
    vd_shared = list(v.vd.intersection(vn.vd))
    if len(vd_shared) != 2:
        return None, None

    # dual edge vector
    e_star = vd_shared[1].x_a - vd_shared[0].x_a
    L_star = np.linalg.norm(e_star)

    if L_star < 1e-14:
        return None, None

    # primal edge
    t = vn.x_a - v.x_a

    # normal via 90° rotation
    n = np.array([t[1], -t[0]])
    n /= np.linalg.norm(n)

    return L_star, n


def dual_edge_flux(v, vn):
    L_star, n = dual_edge_and_normal(v, vn)
    if L_star is None:
        return 0.0

    # midpoint velocity
    u_ij = 0.5 * (v.u + vn.u)

    # flux
    return L_star * np.dot(u_ij, n)

def discrete_divergence(v):
    div = 0.0
    for vn in v.nn:
        F = dual_edge_flux(v, vn)
        div += F
    return div / d_area(v)

def mass_imbalance(HC):
    divs = []
    for v in HC.V:
        if is_boundary(v):
            continue
        divs.append(abs(discrete_divergence(v)))
    return np.max(divs), np.mean(divs)


def plot_dual_fluxes(HC, scale=1e-4):
    import matplotlib.pyplot as plt

    X, Y, FX, FY = [], [], [], []

    for v in HC.V:
        for vn in v.nn:
            if id(v) > id(vn):
                continue

            L_star, n = dual_edge_and_normal(v, vn)
            if L_star is None:
                continue

            u_ij = 0.5 * (v.u + vn.u)
            flux_vec = L_star * np.dot(u_ij, n) * n

            xm = 0.5 * (v.x_a + vn.x_a)
            X.append(xm[0]); Y.append(xm[1])
            FX.append(flux_vec[0]); FY.append(flux_vec[1])

    plt.figure(figsize=(7,7))
    plt.quiver(X, Y, FX, FY, scale=scale)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Dual-edge velocity fluxes")
    plt.axis("equal")
    plt.show()



# Parameters
mu = 8.90 * 1e-4  # Pa·s
L = 1  # m
Q = 1  # m3 / s
R = 1  # m
A = 1  # m2
dP_anal = (8 * np.pi * mu * L * Q) / (A**2)
P_in = 101325 # Pa
def P_ic(x):
    return P_in - ((8 * np.pi * mu * x * Q) / (A**2))

h = R
Q = 1
G = (Q * 12 * mu) / (h**3)
rho = 1000

def u_ic(x):
    y = x[1]
    return (G / (2 * mu)) * y * (h - y)

def P_IC(HC):
    for v in HC.V:
        P_i = P_ic(v.x_a[0])
        v.P = np.array([P_i, P_i])

# Run
x_lb = 0
x_ub = R
y_lb = 0
y_ub = R
domain = [(x_lb, x_ub), (y_lb, y_ub)]
points = incom_Poi(domain, refinements=2)  # Increased refinement for better accuracy
HC, tri = triang_dual(points)
HC = compute_vd(HC)
# plot_dual_mesh_2D(HC, tri)  # Optional

Areas = []
for vp1 in HC.V:
    area = d_area(vp1)
    Areas.append(area)

mass_IC(HC)
set_zero_vel(HC)
# P_IC(HC)  # Not used in evolution

# Plot initial
p = np.array([v.x_a for v in HC.V])
u = np.array([v.u for v in HC.V])
plot_discrete_field(p, u, HC, tri, scale=1e-5, title='Initial Velocity Field')
plot_velocity_profile(HC, title='Initial Velocity Profile')

# Time evolution with standard CFD features: convergence monitoring, residual, error history
nu = mu / rho
f = np.array([G / rho, 0.0])  # Body force equivalent to pressure gradient
dx = min_edge_length(HC)
dt = 0.1 * dx**2 / nu  # Explicit stability
max_steps = 3000
tol_error = 1e-6
tol_res = 1e-8
error_history = []
res_history = []

step = 0
while step < max_steps:
    lap_u0 = compute_lap(HC, 'u', 0)
    lap_u1 = compute_lap(HC, 'u', 1)
    for v in HC.V:
        v.du = np.zeros(2)  # For residual
        if is_boundary(v):
            v.u = np.array([0.0, 0.0])
            continue
        du = dt * (f + nu * np.array([lap_u0[v], lap_u1[v]]))
        v.du = du
        v.u += du

    curr_error = v_error(HC)
    curr_res = compute_residual(HC)
    error_history.append(curr_error)
    res_history.append(curr_res)
    # check mass imbalance
    max_div, mean_div = mass_imbalance(HC)


    if step % 1000 == 0:
        print(f"Step {step}, MSE Error = {curr_error:.2e}, Residual = {curr_res:.2e}, max mass imbalance = {max_div:.2e}")

    if curr_error < tol_error and curr_res < tol_res:
        print(f"Converged at step {step}")
        break

    step += 1

# Plot convergence history
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.semilogy(error_history, label='MSE Error')
plt.semilogy(res_history, label='Residual')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend()
plt.title('Convergence History')
plt.show()

# Plot final
p = np.array([v.x_a for v in HC.V])
u = np.array([v.u for v in HC.V])
plot_discrete_field(p, u, HC, tri, scale=1e-5, title='Final Velocity Field')
plot_velocity_profile(HC, title='Final Velocity Profile')

print(f'Final MSE Error = {v_error(HC):.2e}')