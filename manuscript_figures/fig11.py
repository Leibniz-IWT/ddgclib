import numpy as np
import scipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from ipywidgets import *
from matplotlib import pyplot as plt, ticker as mticker

# ddg imports
# Allow for relative imports from main library:
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from ddgclib import *
from ddgclib._complex import *
from ddgclib._curvatures import *  # plot_surface#, curvature
from ddgclib._capillary_rise_flow import * #plot_surface#, curvature
from ddgclib._eos import *
from ddgclib._misc import *
from ddgclib._plotting import *


# Parameters for a water droplet in air at standard laboratory conditions
gamma = 0.0728  # N/m, surface tension of water at 20 deg C
rho = 1000  # kg/m3, density
g = 9.81  # m/s2

# Colour scheme for surfaces
db = np.array([129, 160, 189]) / 255  # Dark blue
lb = np.array([176, 206, 234]) / 255  # Light blue

cdist = 1e-12

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
r = 0.5e-3  # m, Radius of the capillary tube
theta_p = 20.001 * np.pi / 180.0  # Three phase contact angle
R = r / np.cos(theta_p)  # = R at theta = 0

# Note the capillary rise systems can be visualized with this code at any N/refinement:
if 0:
    N = 8
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

    (HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i,  HNdA_ij_dot_hnda_i,
     K_H_2, HNdA_i_Cij) = int_curvatures(HC, bV, r, theta_p, printout=0)
    # Laplacian error
    P_L = gamma*(2.0/R)  # Analytical solution
    H_dis = HNdA_ij_dot_hnda_i[0]
    L_error = 100 *(P_L - gamma * H_dis)/(P_L)
    lp_error.append(abs(L_error))
    max_int_e = 0.0
    ern = 0.0
    for v in bV:
        for v2 in v.nn:
            if v2 in bV:
                a = v.x_a
                b = v2.x_a
                ern = 0.5 * numpy.linalg.norm(a - b) ** 2 / (12 * (2)**2)
                max_int_e += ern

    erange.append( (max_int_e/r)/len(bV) )

# Computed from step size on Euclidean metric for cap rise:
geo_error = erange

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax2 = ax.twinx()
print(f'lp_error = {lp_error}')
ln1 = ax2.loglog(Nlist, np.abs(lp_error), 'x', color='tab:blue',
             label='Young-Laplace error: $(F_{cap} - \hat{F_{cap}}) / F_{cap} $')

ax2.set_ylabel(r'Young-Laplace error (%)')
ax.set_xlabel(r'$n$ (number of boundary vertices)')

eps = (geo_error[-1] - geo_error[0])/(Nlist[-1] - Nlist[0])
print(f' eps = {eps**(-1)}')
ln2 = ax.loglog(Nlist, geo_error, 'X', color='tab:orange',
          label=f'Integration error (p-normed trapezoidal rule $O(h^2)$)')
ax.set_ylabel(f'Integration error (%)')

x, y = Nlist, geo_error
x, y = np.float32(x), np.float32(y)
z = numpy.polyfit(x, y, 1)
z = np.polyfit(np.log10(x), np.log10(y), 1)

p = numpy.poly1d(z)
p = np.poly1d(z)
# the line equation:
print("y=%.6fx+(%.6f)" % (z[0], z[1]))
ax2.set_ylim([1e-15, 1e-12])
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

plt.tick_params(axis='y', which='minor')


ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())

plt.tick_params(axis='y', which='minor')

plt.show()












