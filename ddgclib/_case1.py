## Imports and physical parameters
# std library
import numpy as np
import scipy

# plotting
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from ipywidgets import *
from matplotlib.widgets import Slider

# ddg imports
from . import *
from ._complex import *
from ._curvatures import * #plot_surface#, curvature
from ._capillary_rise_flow import * #plot_surface#, curvature
from ._capillary_rise import * #plot_surface#, curvature
from ._eos import *
from ._misc import *
from ._plotting import *


def plot_cap_rise_over_theta(c_outd_list, c_outd, vdict, X):
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

    # Plot the values from new_out_plot_cap_rise
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

        F, nn, HC, bV, K_f, H_f = cap_rise_init_N(r, theta_p, gamma, N=N,
                                                  refinement=refinement)
        K_fl.append(K_f)
        H_fl.append(H_f)

    key = 'K_f'
    value = K_fl
    ax.plot(Theta_p* 180 / np.pi, value,
            marker=keyslabel[key]['marker'],
            linestyle=keyslabel[key]['linestyle'],
            label=keyslabel[key]['label'], alpha=0.7)

    key = 'K_H_i'
    value = vdict[key]
    ax.plot(X, value,
            marker=keyslabel[key]['marker'],
            linestyle=keyslabel[key]['linestyle'],
            markerfacecolor='None',
            label=keyslabel[key]['label'], alpha=0.7)

    key = 'H_f'
    value = H_fl
    ax.plot(Theta_p* 180 / np.pi, value,
            marker=keyslabel[key]['marker'],
            linestyle=keyslabel[key]['linestyle'],
            label=keyslabel[key]['label'], alpha=0.7)

    key = 'HN_i'
    value = vdict[key]
    ax.plot(X, value,
            marker=keyslabel[key]['marker'],
            linestyle=keyslabel[key]['linestyle'],
            markerfacecolor='None',
            label=keyslabel[key]['label'], alpha=0.7)

    if 1:
        plot.xlabel(r'Contact angle $\Theta_{C}$')
        plot.ylabel(r'Gaussian curvature ($m^{-1}$)')
        ax2 = ax.twinx()
        plt.ylabel('Mean normal curvature ($m^{-1}$)')
    else:
        plot.xlabel(r'Contact angle $\Theta_{C}$')
        plot.ylabel(r'$K$ ($m^{-1}$)')
        ax2 = ax.twinx()
        plt.ylabel('$H$ ($m^{-1}$)')

    plt.ylim((0, 2))
    ax.legend(bbox_to_anchor=(0.15, 0.15), loc="lower left",
              bbox_transform=fig.transFigure, ncol=2)
    # interact(update);

