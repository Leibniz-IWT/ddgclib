import numpy as np
import scipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from ipywidgets import *
from matplotlib.widgets import Slider
from matplotlib.widgets import Slider

# Parameters for a water droplet in air at standard laboratory conditions
gamma = 0.0728  # N/m, surface tension of water at 20 deg C
rho = 1000  # kg/m3, density
g = 9.81  # m/s2

from ddgclib import *
from ddgclib._complex import *
from ddgclib._curvatures import *  # plot_surface#, curvature
from ddgclib._capillary_rise_flow import * #plot_surface#, curvature
from ddgclib._eos import *
from ddgclib._misc import *
from ddgclib._plotting import *

# Colour scheme for surfaces
db = np.array([129, 160, 189]) / 255  # Dark blue
lb = np.array([176, 206, 234]) / 255  # Light blue

cdist = 1e-10

# Replot error data
Nmax = 21
#lp_error = np.zeros(Nmax - 3)
Nlist = list((range(3, Nmax)))
Nlist = list((range(4, Nmax)))

erange = []
lp_error = []
Nmax = 21
refinement = 1
phi = 2
r = 1  # m, Radius of the capillary tube
theta_p = 20 * np.pi / 180.0  # Three phase contact angle
R = r / np.cos(theta_p)  # = R at theta = 0

N = 8
fig, axes, HC = cape_rise_plot(r, theta_p, gamma, N=N, refinement=refinement)

axes.grid(False)
axes.set_xticks([])
axes.set_yticks([])
axes.set_zticks([])
plt.axis('off')
N = 5
fig, axes, HC = cape_rise_plot(r, theta_p, gamma, N=N, refinement=refinement)
axes.grid(False)
axes.set_xticks([])
axes.set_yticks([])
axes.set_zticks([])
plt.axis('off')

#for N in range(4, Nmax + 1):
for N in range(5, Nmax + 1):

    F, nn, HC, bV, K_f, H_f = cap_rise_init_N(r, theta_p, gamma, N=N,
                                              refinement=refinement,
                                              cdist=cdist
                                              )

    HC.V.merge_all(cdist=cdist)

    if 0:
        fig, axes, fig_s, axes_s = HC.plot_complex()
        axes.set_zlim(-r, 0.0)
        axes.set_zlim(-2*r, 2*r)

    if 0:
        fig, axes, HC = cape_rise_plot(r, theta_p, gamma, N=N, refinement=0)

    (HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i,  HNdA_ij_dot_hnda_i,
     K_H_2, HNdA_i_Cij) = int_curvatures(HC, bV, r, theta_p, printout=0)
    # Laplacian error
    P_L = gamma*(2.0/R)  # Analytical solution
    print(f'P_L = {P_L}')
    print(f'HNdA_ij_dot_hnda_i = {HNdA_ij_dot_hnda_i}')
    H_dis = HNdA_ij_dot_hnda_i[0]
    L_error = 100 *(P_L - gamma * H_dis)/(P_L)
    lp_error.append(abs(L_error) + 1e-13)
    if 0:
        plt.figure()
        yr = K_H_2 - K_f
        xr = list(range(len(yr)))
        plt.plot(xr, yr, label='K_H_2 - K_f')
        yr = HNdA_ij_dot_hnda_i - H_f
        print(f'yr = {yr}')
        plt.plot(xr, yr, label='HNdA_ij_dot_hnda_i - H_f')
        plt.ylabel('Difference')
        plt.xlabel('Vertex No.')
        plt.legend()

    max_int_e = 0.0
    ern = 0.0
    for v in bV:
        for v2 in v.nn:
            if v2 in bV:
                a = v.x_a
                b = v2.x_a
                if N == 14:
                    print(f'numpy.linalg.norm(a - b) = {numpy.linalg.norm(a - b)}')
                    continue
                break
        ern = 2*numpy.linalg.norm(a - b)
        max_int_e = ern
        break

    erange.append(max_int_e/r)



erange
# Computed from step size on Euclidean metric for cap rise:
geo_error = erange

plot.figure()
# plt.plot(N, lp_error, 'x', label='$\Delta p \frac{\Delta p - \Delta p}{\Delta p}$')
if 1:
    plt.plot(Nlist, lp_error, 'x')
    plt.plot(Nlist, lp_error, 'x', label='Young-Laplace error: $(\Delta p - \Delta\hat{ p})/\Delta p $')
    plt.plot(Nlist, geo_error, 'X', label='Integration error (Trapezoidal rule $O(h^3)$)')

if 0:
    plt.semilogy(Nlist, lp_error, 'x')
    plt.semilogy(Nlist, lp_error, 'x', label='Young-Laplace error: $(\Delta p - \Delta\hat{ p})/\Delta p $')
    plt.semilogy(Nlist, geo_error, 'X', label='Integration error (Trapezoidal rule $O(h^3)$)')#

plt.legend()
plt.xlabel(r'N (number of boundary vertices)')
plt.ylabel(r'Error (%)')

import matplotlib

matplotlib.pyplot.xticks(Nlist)

plt.tick_params(axis='y', which='minor')



if 0:
    plot.figure()
    fig, ax1 = plt.subplots()

    color = 'tab:orange'
    ax1.set_ylabel('Young-Laplace error: $\Delta p - \Delta\hat{ p} $', color=color)  # we already handled the x-label with ax1
    #ax1.semilogy(Nlist, lp_error, 'x', color=color)
    ax1.plot(Nlist, lp_error, 'x', color=color)
    ax1.tick_params(axis='y', labelcolor=color)


    color = 'tab:red'

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_xlabel(r'N (number of boundary vertices)')
    ax2.set_ylabel('Integration error (Trapezoidal rule $O(h^3)$)', color=color)
    ax2.plot(Nlist, geo_error, 'x', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    matplotlib.pyplot.xticks(Nlist)

    plt.ylabel(r'Error (%)')

plt.show()












