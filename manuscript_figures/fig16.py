# Imports and physical parameters
import numpy as np
import scipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from ipywidgets import *
from matplotlib.widgets import Slider

# ddg imports
# Allow for relative imports from main library:
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from ddgclib._complex import Complex
from ddgclib import *
from ddgclib._complex import *
from ddgclib._curvatures import * #plot_surface#, curvature
from ddgclib._capillary_rise_flow import * #plot_surface#, curvature
from ddgclib._hyperboloid import *
from ddgclib._catenoid import *
from ddgclib._ellipsoid import *
from ddgclib._eos import *
from ddgclib._misc import *
from ddgclib._plotting import *
from ddgclib._case2 import *

# Parameters for a water droplet in air at standard laboratory conditions
gamma = 0.0728  # N/m, surface tension of water at 20 deg C
rho = 1000  # kg/m3, density
g = 9.81  # m/s2

# Numerical parameters #Stated this is what to plaay
r = 1
theta_p = 20 * np.pi/180.0  # rad, three phase contact angle
refinements =  [2, 3, 4, 5]   # NOTE: 2 is the minimum refinement needed for the complex to be manifold

# data containers:
lp_error = []
lp_error_2 = []
geo_error = []

for refinement in refinements:
    # Compute all curvatures:
    (HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i,  HNdA_ij_dot_hnda_i,
         K_H_2, HNdA_i_Cij) = int_curvatures(HC, bV, r, theta_p, printout=0)

    # Process the results:
    c_outd2 = []  #TODO not needed?
    HN_i_2 = []
    HNdA_i_list = []
    C_ij_i_list = []
    for v in HC.V:
        if v in bV:
            continue
        nullp = np.zeros(3)
        nullp[2] = v.x_a[2]
        N_f0 = v.x_a - nullp  # First approximation of normal vectors
        N_f0 = normalized(N_f0)[0]
        F, nn = vectorise_vnn(v)
        c_outd2 = b_curvatures_hn_ij_c_ij(F, nn, n_i=N_f0)
        HN_i_2.append(c_outd2['HN_i'])
        HNdA_i_list.append(c_outd2['HNdA_i'])
        C_ij_i_list.append(c_outd2['C_ij'])


    # Sum all the curvature tensor errors in the postive 'upwards' direction
    sum_HNdA_i = 0.0
    for hndA_i, c_ij in zip(HNdA_i_list, C_ij_i_list):
        hndA_i_c_ij = hndA_i / np.sum(c_ij)
        sum_HNdA_i += np.dot(hndA_i_c_ij, [0, 0, 1])

    # Print out results:
    print(f'Number of vertices in the complex: {HC.V.size()}')  #
    print(f'Number of boundary vertices in the complex: {len(bV)}')  # n
    print(f'Integrated curvature sum_HNdA_i in (0, 0, 1) direction = {sum_HNdA_i}')

    lp_error.append(sum_HNdA_i)
    # Sum all the curvature tensor errors in the negative 'upwards' direction
    for hndA_i, c_ij in zip(HNdA_i_list, C_ij_i_list):
        hndA_i_c_ij = hndA_i / np.sum(c_ij)
        sum_HNdA_i += np.dot(hndA_i_c_ij, [0, 0, -1])

    lp_error_2.append(sum_HNdA_i)
    print(f'Integrated curvature sum_HNdA_i in (0, 0, -1) direction = {sum_HNdA_i}')

    # Compute the geometric error
    max_int_e = 0.0
    ern = 0.0
    for v in bV:
        for v2 in v.nn:
            if v2 in bV:  # Explain
                a = v.x_a
                b = v2.x_a
                break
        ern = 0.5 * numpy.linalg.norm(a - b) ** 2
        max_int_e = ern
        break

    print(f'geo erange = {ern }')
    geo_error.append(ern)


# Plot the final results
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax2 = ax.twinx()
ln1 = ax2.loglog(Nlist, np.abs(lp_error), 'x', color='tab:blue',
             label='Capillary force error: $(F_{cap} - \hat{F_{cap}}) / F_{cap} $')

ax2.set_ylabel(r'Capillary force error (%)')
ax.set_xlabel(r'$n$ (number of boundary vertices)')

ln2 = ax.loglog(Nlist, geo_error, 'X', color='tab:orange',
          label='Integration error (p-normed trapezoidal rule $O(h^2)$)')

ax.set_ylabel(r'Integration error (%)')

ax2.set_ylim([1e-15, 1e-18])
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
plt.tick_params(axis='y', which='minor')

x, y = Nlist, geo_error
x, y = np.float32(x), np.float32(y)
z = numpy.polyfit(x, y, 1)
z = np.polyfit(np.log10(x), np.log10(y), 1)

p = numpy.poly1d(z)
p = np.poly1d(z)
# the line equation:
print("y=%.6fx+(%.6f)" % (z[0], z[1]))

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

plt.tick_params(axis='y', which='minor')
if 1:
    from matplotlib import pyplot as plt, ticker as mticker

    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())
