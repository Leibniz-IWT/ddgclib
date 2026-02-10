# moved from ddgclib
# caprise.py

import numpy as np
import matplotlib.pyplot as plt

# ddg imports
# from ddgclib import *
from hyperct import *
from ddgclib._curvatures import HC_curvatures, int_curvatures, Gauss_Bonnet, analytical_cap
from ddgclib._capillary_rise_flow import kmean_flow, film_init, mean_flow
from ddgclib._capillary_rise import cape_rise_plot, plot_c_outd, new_out_plot_cap_rise, out_plot_cap_rise, out_plot_cap_rise_boundary, cap_rise_init_N
# from ddgclib._eos import *
from ddgclib._misc import *
from ddgclib._plotting import plot_polyscope


# Execution flags

RUN_CAP_RISE_PLOT = False
RUN_THETA_RISE_PLOT = True
RUN_NEW_THETA_FORMULATION = False
RUN_PLOT_COMPLEX = True
RUN_OLD_CURVATURES = False
RUN_OLD_GAUSS_BONNET = False
RUN_NEW_BOUNDARY_FORMULATION = True
RUN_MEAN_CURVATURE_FLOW = False
RUN_FILM_MEAN_FLOW = False
RUN_GAUSS_BONNET_PLOT = False
RUN_MEAN_FLOW = False
RUN_POLYSCOPE_VIEW = True



# Physical parameters

gamma = 0.0728  # N/m (water surface tension at 20 °C)
rho = 1000.0    # kg/m^3 (density)
g = 9.81        # m/s^2 (gravity)


# Capillary rise parameters

r = 1.0            # tube radius (m)
theta_p = 20.0 * np.pi / 180.0  # contact angle (rad)

# mesh resolution
N = 7
refinement = 1
cdist = 1e-10

# convert to long double for precision
r = np.array(r, dtype=np.longdouble)
theta_p = np.array(theta_p, dtype=np.longdouble)



# Helper functions

def add_sphere_cap(axes, a, theta_p):
    """
    Plot a spherical cap on the existing axes.
    """
    def cap_ratio(r, a, h):
        surface_cap = np.pi * (a ** 2 + h ** 2)
        surface_sphere = 4.0 * np.pi * r ** 2
        return surface_cap / surface_sphere

    def find_radius(a, h):
        return (a ** 2 + h ** 2) / (2 * h)

    h = a * np.cos(theta_p)
    R = a / np.cos(theta_p)
    h = R - R * np.sin(theta_p)
    r = R

    p = cap_ratio(r, a, h)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, theta_p, 100)

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = -r * np.outer(np.ones(np.size(u)), np.cos(v))
    z = z + R - h

    axes.plot_surface(x, y, z, rstride=4, cstride=4, alpha=0.3)
    return axes



# Capillary rise plot with surrounding cylinder

if RUN_CAP_RISE_PLOT:
    fig, axes, HC = cape_rise_plot(r, theta_p, gamma, N=N, refinement=refinement)

    axes = add_sphere_cap(axes, r, theta_p)
    axes.set_xlabel('x (m)')
    axes.set_ylabel('y (m)')
    axes.set_zlabel('z (m)')

    # additional 2D profile plot (optional)
    fig, ax = plt.subplots()
    x = np.linspace(0, 1.0, 1000)
    h = r * np.cos(theta_p)
    R = r / np.cos(theta_p)
    h = R - R * np.sin(theta_p)

    y = (h - np.sqrt(h ** 2 - x ** 2)) / 2
    y = np.sqrt(r ** 2 + x ** 2) - 1.0
    ymax = np.max(y)
    y = y - ymax

    ax.plot(x, y, alpha=0.5, linewidth=2)

    # plot boundary points
    for v in HC.V:
        print(f'v = {v.x_a}')

    x2 = [0.0, 0.6380986, 1.0]
    y2 = [-ymax,
          np.sqrt(r ** 2 + x2[1] ** 2) - 1.0 - ymax,
          np.sqrt(r ** 2 + x2[2] ** 2) - 1.0 - ymax]

    ax.plot(x2, y2, linestyle='-', marker='o', linewidth=2)
    ax.fill_between(x2[0:2], y2[0:2], 0, alpha=0.3)
    ax.fill_between(x2[1:], y2[1:], 0, alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')



# Plot theta rise

if RUN_THETA_RISE_PLOT:
    c_outd_list, c_outd, vdict, X = out_plot_cap_rise(
        N=N, r=r, gamma=gamma, refinement=refinement
    )
    plot_c_outd(c_outd_list, c_outd, vdict, X)



# Plot theta rise with new formulation (2021-07)

if RUN_NEW_THETA_FORMULATION:
    c_outd_list, c_outd, vdict, X = new_out_plot_cap_rise(
        N=N, r=r, gamma=gamma, refinement=refinement
    )
    keyslabel = {
        'K_f': {'label': '$K$', 'linestyle': '-', 'marker': None},
        'K_H_i': {'label': '$\\widehat{K_i}$', 'linestyle': 'None', 'marker': 'o'},
        'H_f': {'label': '$H$', 'linestyle': '--', 'marker': None},
        'HN_i': {'label': '$\\widehat{H_i}$', 'linestyle': 'None', 'marker': 'D'},
    }

    plot_c_outd(c_outd_list, c_outd, vdict, X, keyslabel=keyslabel)



# Plot the current complex

if RUN_PLOT_COMPLEX:
    F, nn, HC, bV, K_f, H_f = cap_rise_init_N(
        r, theta_p, gamma,
        N=N,
        refinement=refinement,
        cdist=cdist,
        equilibrium=True
    )

    HC.V.merge_all(cdist=cdist)
    fig, axes, fig_s, axes_s = HC.plot_complex(point_color=db, line_color=lb)

    axes.set_zlim(-2 * r, 2 * r)

    R = r / np.cos(theta_p)
    K_f = (1 / R) ** 2
    H_f = 2 / R

    print(f'R = {R}')
    print(f'K_f = {K_f}')
    print(f'H_f = {H_f}')
    print(f'theta_p = {theta_p} rad')

    
    # Old interior curvature (deprecated)
    
    if RUN_OLD_CURVATURES:
        (HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i,
         HNdA_ij_dot_hnda_i, K_H_2, HNdA_i_Cij) = int_curvatures(
            HC, bV, r, theta_p, printout=1
        )

    
    # Old Gauss-Bonnet (angle defect)
    
    if RUN_OLD_GAUSS_BONNET:
        chi, KdA, k_g = Gauss_Bonnet(HC, bV, print_out=True)

    
    # New boundary formulation (current)
    
    if RUN_NEW_BOUNDARY_FORMULATION:
        print("New boundary formulation:")

        H_f, K_f, dA, k_g_f, dC = analytical_cap(r, theta_p)

        (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
         HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache,
         HNdA_i_Cij_cache, Theta_i_cache) = HC_curvatures(
            HC, bV, r, theta_p, printout=0
        )

        # Plot curvature errors
        plt.figure()
        yr = np.array(K_H_i) - K_f
        xr = list(range(len(yr)))
        plt.plot(xr, yr, 'o', label='K_H_i - K_f')

        yr = np.array(HN_i) - H_f
        plt.plot(xr, yr, 'x', label='HN_i - H_f')

        plt.ylabel('Difference')
        plt.xlabel('Vertex No.')
        plt.legend()



# New Mean flow

if RUN_MEAN_CURVATURE_FLOW:
    R = r / np.cos(theta_p)
    K_f = (1 / R) ** 2
    H_f = 2 / R
    h_jurin = 2 * gamma * np.cos(theta_p) / (rho * g * r)

    F, nn, HC, bV, K_f, H_f = cap_rise_init_N(
        r, theta_p, gamma, N=N, refinement=refinement, equilibrium=True
    )

    params = (gamma, rho, g, r, theta_p, K_f, 0.0)
    tau = 1e-8

    HC, bV = kmean_flow(HC, bV, params, tau=tau, print_out=0)

    h_final = 0.0
    for v in HC.V:
        h_final = np.min([h_final, v.x_a[2]])

    print(f'h_final = {h_final} m')
    print(f'h_jurin = {h_jurin} m')
    print(f'h_final - h_jurin = {h_final - h_jurin} m')

    fig, axes, fig_s, axes_s = HC.plot_complex(point_color=db, line_color=lb)
    axes.set_zlim(-2 * r, h_jurin + 2 * r)



# Film mean flow

if RUN_FILM_MEAN_FLOW:
    R = r / np.cos(theta_p)
    K_f = (1 / R) ** 2
    H_f = 2 / R
    h_jurin = 2 * gamma * np.cos(theta_p) / (rho * g * r)

    HC, bV = film_init(r, h_jurin)
    HC.V.merge_all(cdist=cdist)

    fig, axes, fig_s, axes_s = HC.plot_complex(point_color=db, line_color=db)

    h_0 = 2 * h_jurin

    params = (gamma, rho, g)
    HC = mean_flow(HC, bV, h_0=h_0, params=params, tau=1e-5)
    HC.V.merge_all(cdist=cdist)

    h_f = 2 * h_jurin
    for v in HC.V:
        if v in bV:
            continue
        h_f = min(v.x[2], h_f)

    fig, axes, fig_s, axes_s = HC.plot_complex(point_color=db, line_color=db)
    print(f'Final capillary height: {h_f}')
    print(f'Expected rise: {h_jurin}')



# Gauss-Bonnet plot

if RUN_GAUSS_BONNET_PLOT:
    c_outd_list, c_outd, vdict, X = out_plot_cap_rise_boundary(
        N=N, r=r, refinement=2
    )
    plot_c_outd(c_outd_list, c_outd, vdict, X, ylabel=r'$\pi \cdot m$ or $\pi \cdot m^{-1}$')



# Polyscope view

if RUN_POLYSCOPE_VIEW:
    fn = f"r_{r}_bV_{len(bV)}"
    ps = plot_polyscope(HC, fn=fn, up="z_up", stl=True)
    ps.show()


plt.show()
