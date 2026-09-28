"""
Plots using the data output from running droplet.py for the given refinements
 and parameters sets. To obtain this data droplet.py must be run switching the
 different parameter sets on/off for the 3 refinements listed in the file.

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from ipywidgets import *
from matplotlib.widgets import Slider
from matplotlib import pyplot as plt, ticker as mticker

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
# Our formulation (droplet.py)
N_list = [7, 19, 61]  # Total number of vertices at end
bV_list = [6, 12, 24]  #
# Average data norm error:

data_error = [2.5271702590869453e-06,
              2.5271699752705548e-06,
              9.175761973317646e-06]

# Average relative angle error
phi_rel_error = [2.8381639047890193e-13,
                 0.0003696689325014279,
                 0.02953080254880999]

# Surface evolver
se_N_list = [12, 39, 141]
se_data_error = [0.010929308788062657, 0.003524668110821649,
                 0.0022713988668246385]

# Avg data error = 0.0020213679022338337
# HC.V.size() = 9873


if 0:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()
    ln2 = ax2.loglog(N_list, data_error, 'x', color='tab:blue',
                     #label=r'Relative error in DDG formulation $\min _{i, j}  \frac{|| \bf{x_i} - \bf{x_j^*} ||}{\bf{x_j^*}}$')
                     label=r'Relative error in DDG formulation')

    ax2.set_ylabel(r'Relative error (DDG formulation)')
    ax.set_xlabel(r'$n$ (number of vertices)')
    ln1 = ax.loglog(se_N_list, se_data_error, 'X', color='tab:orange',
                    #label=r'Surface Evolver error $\min _{i, j}  \frac{|| \bf{x_i} - \bf{x_j^*} ||}{\bf{x_j^*}}$')
                    label=r'Relative error in Surface Evolver')
    ax.set_ylabel(f'Relative error (Surface Evolver)')


    ax2.set_ylim([1e-6, 5e-5])
    ax.set_ylim([1e-3, 1e-1])
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)

    plt.tick_params(axis='y', which='minor')

    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())

    plt.tick_params(axis='y', which='minor')
else:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax = ax.twinx()
    ln2 = ax.loglog(N_list, data_error, 'x', color='tab:blue',
                     #label=r'Relative error in DDG formulation $\min _{i, j}  \frac{|| \bf{x_i} - \bf{x_j^*} ||}{\bf{x_j^*}}$')
                     label=r'Relative error in DDG formulation')

    ax.set_ylabel(r'Relative error (DDG formulation)')
    ax.set_xlabel(r'$n$ (number of vertices)')
    ln1 = ax.loglog(se_N_list, se_data_error, 'X', color='tab:orange',
                    #label=r'Surface Evolver error $\min _{i, j}  \frac{|| \bf{x_i} - \bf{x_j^*} ||}{\bf{x_j^*}}$')
                    label=r'Relative error in Surface Evolver')
    ax.set_ylabel(f'Relative error')


    #ax2.set_ylim([1e-6, 5e-5])
    #ax.set_ylim([1e-3, 1e-1])
    ax.set_ylim([1e-6, 1e-1])
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)

    plt.tick_params(axis='y', which='minor')

    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())

    plt.tick_params(axis='x', which='minor')

if 0:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax2 = ax.twinx()
    ln1 = ax.loglog(N_list, data_error, 'x', color='tab:blue',
                     label='Error 1 $')

    ax.set_ylabel(r'Young-Laplace error (%)')
    ax.set_xlabel(r'$n$ (number of vertices)')
    ln2 = ax.loglog(se_N_list, se_data_error, 'X', color='tab:orange',
                    label=f'Error 2 surface evovler)')
    ax.set_ylabel(f'Integration error (%)')

    #ax.set_ylim([1e-1, 1e-7])
    lns = ln2 + ln1
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)

    plt.tick_params(axis='y', which='minor')

    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    #ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())

    plt.tick_params(axis='y', which='minor')

    #plt.show()

plt.show()