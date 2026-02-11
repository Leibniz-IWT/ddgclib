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
from ddgclib import *
from hyperct import *
from ddgclib._curvatures import * #plot_surface#, curvature
from ddgclib._capillary_rise_flow import * #plot_surface#, curvature
from ddgclib._eos import *
from ddgclib._misc import *
from ddgclib._plotting import *
if 0:
    plt.rcParams.update({
        "text.usetex": True,
        #"font.family": "sans-serif",
        #"font.sans-serif": ["Calibri"]
    })
    # for Palatino and other serif fonts use:
    plt.rcParams.update({
        "text.usetex": True,
        #"font.family": "serif",
        #"font.serif": ["Calibri"],
    })
    # It's also possible to use the reduced notation by directly setting font.family:
    plt.rcParams.update({
        "text.usetex": True,
        #"font.family": "Calibri",
        #'font.size': 14
    })


# Parameters for a water droplet in air at standard laboratory conditions
gamma = 0.0728  # N/m, surface tension of water at 20 deg C
rho = 1000  # kg/m3, density
g = 9.81  # m/s2

# Parameters from EoS:
#T_0 = 273.15 + 25  # K, initial tmeperature
#P_0 = 101.325  # kPa, Ambient pressure
#gamma = IAPWS(T_0)  # N/m, surface tension of water at 20 deg C
#rho_0 = eos(P=P_0, T=T_0)  # kg/m3, density

# Capillary rise parameters
#r = 0.5e-2  # Radius of the droplet sphere
r = 2.0  # Radius of the droplet sphere
r = 2.0e-6  # Radius of the tube
r = 2.0e-6  # Radius of the tube
r = 0.5e-6  # Radius of the tube
r = 1.4e-5  # Radius of the tube
r = 1.4e-5  # Radius of the tube
r = 0.5e-3  # Radius of the tube (20 mm)
#r = 0.5  # Radius of the tube (20 mm)
#r = 1 # Radius of the tube (20 mm)


#r = 0.5e-5  # Radius of the droplet sphere
#theta_p = 45 * np.pi/180.0  # Three phase contact angle
theta_p = 20 * np.pi/180.0  # Three phase contact angle
#phi = 0.0
N = 8
N = 5
#N = 6
N = 7
#N = 5
#N = 12
refinement = 1
#N = 20
#cdist = 1e-10
cdist = 1e-10

r = np.array(r, dtype=np.longdouble)
theta_p = np.array(theta_p, dtype=np.longdouble)

# PLot theta rise with New fomulation over Phi_C (2021-07)
# NOTE: THIS IS CURRENTLY USED IN THE MANUSCRIPT:
if 1:
    #TODO: Check out the N_f0 vectors here! They are NOT the same as the ones
    #      used in the default HC_curvatures function
    c_outd_list, c_outd, vdict, X = new_out_plot_cap_rise(N=N, r=r,
        gamma=gamma, refinement=refinement)
    keyslabel = {'K_f': {'label': '$K$',
                         'linestyle':  '-',
                         'marker': None},
                'K_H_i': {'label': '$\widehat{K_i}$',
                         'linestyle':  'None',
                          'marker': 'o'},
                'H_f': {'label': '$H$',
                        'linestyle':  '--',
                        'marker': None},
                'HN_i': {'label': '$\widehat{H_i}$',
                         'linestyle': 'None',
                         'marker': 'D'},
    }

    plot_c_outd(c_outd_list, c_outd, vdict, X, keyslabel=keyslabel)

    fig = plt.figure()
    # ax = fig.add_subplot(2, 1, 1)
    ax = fig.add_subplot(1, 1, 1)

    ind = 0
    Lines = {}
    fig.legend()

    K_fl = []
    H_fl = []
    Theta_p = np.linspace(0.0, 0.5 * np.pi, 100)
    for theta_p in Theta_p:
        # Contruct the simplicial complex, plot the initial construction:
        # F, nn = droplet_half_init(R, N, phi)
        R = r / np.cos(theta_p)  # = R at theta = 0
        # Exact values:
        K_f = (1 / R) ** 2
        H_f = 1 / R + 1 / R  # 2 / R
        # dp_exact = gamma * H_f

        F, nn, HC, bV, K_f, H_f = cap_rise_init_N(r, theta_p, gamma, N=N,
                                                  refinement=refinement)
        K_fl.append(K_f)
        H_fl.append(H_f)

    key = 'K_f'
    value = K_fl
    # Normal good for K = 1 vs. loglog plots
    ax.plot(Theta_p* 180 / np.pi, value,
            color='tab:blue',
            marker=keyslabel[key]['marker'],
            linestyle=keyslabel[key]['linestyle'],
            label=keyslabel[key]['label'], alpha=0.7)

    key = 'K_H_i'
    value = vdict[key]
    ax.plot(X, value,
            color='tab:purple',
            marker=keyslabel[key]['marker'],
            linestyle=keyslabel[key]['linestyle'],
            markerfacecolor='None',
            label=keyslabel[key]['label'], alpha=0.7)

    plot.xlabel(r'Contact angle $\Theta_{C}$ ($^\circ$)')
    plot.ylabel(r'Gaussian curvature ($m^{-2})$ $10^{-6})$')

    ax2 = ax.twinx()


    key = 'H_f'
    value = H_fl
    ax2.plot(Theta_p* 180 / np.pi, value,
            color='tab:orange',
            marker=keyslabel[key]['marker'],
            linestyle=keyslabel[key]['linestyle'],
            label=keyslabel[key]['label'], alpha=0.7)

    key = 'HN_i'
    value = vdict[key]
    ax2.plot(X, value,
            color='tab:red',
            marker=keyslabel[key]['marker'],
            linestyle=keyslabel[key]['linestyle'],
            markerfacecolor='None',
            label=keyslabel[key]['label'], alpha=0.7)



    plt.ylabel('Mean normal curvature ($m^{-1}$)')

  #  plt.ylim((0, max(max( vdict['K_H_i']), max( vdict['HN_i']) + 1e-6)))
  #  ax.set_ylim((0, max(vdict['K_H_i']) + 1))
  #  ax2.set_ylim((0, max(vdict['HN_i'])))
    ax.legend(#bbox_to_anchor=(0.15, 0.15),
              loc="lower left",
              #bbox_transform=fig.transFigure,
        ncol=2
    )
    ax2.legend(#loc="lower left",
             # bbox_transform=fig.transFigure,
        ncol=2
    )
    # interact(update);

    #import matplotlib.ticker as mticker
    #f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    #g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
    #plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(g))

plt.savefig('fig12_x10.png', bbox_inches='tight', dpi=600)
plt.show()







