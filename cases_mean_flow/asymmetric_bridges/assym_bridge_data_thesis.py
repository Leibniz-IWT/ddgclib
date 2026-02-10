# Imports and physical parameters
import numpy as np
import scipy

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
from hyperct import Complex
from ddgclib import *
from hyperct import *
from ddgclib._curvatures import * #plot_surface#, curvature
from ddgclib._capillary_rise_flow import * #plot_surface#, curvature
from ddgclib._hyperboloid import *
from ddgclib._catenoid import *
from ddgclib._ellipsoid import *
from ddgclib._eos import *
from ddgclib._misc import *
from ddgclib._plotting import *
#from ddgclib._case2 import *

# FONT:
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'CMU Serif, Times New Roman'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams["mathtext.fontset"]= 'cm'  # ValueError: Key mathtext.fontset: 'serif' is not a valid value for mathtext.fontset; supported values are ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']
matplotlib.rcParams["mathtext.default"] = 'it'
#matplotlib.rcParams["mathtext.fontset"]
#

errors_point_wise_1 = [2.392492038148525, 10.837791012382999, 46.310079254685476, 190.1675853457836,
                       769.0103098098057, 3090.9577885704543
                       ]
errors_point_wise_2 = [-1.7618285302889447e-19, -4.431676380034499e-18, -6.0986372202309624e-18, 9.23198149871407e-17,
                       -5.455434281403937e-16, 2.9201088162095212e-15
                       ]
# np.sum of the `sum_HNdA_i (total)` data:
errors_total = [-3.25260652e-19, 3.6591823320000005e-19, 1.6805133640000001e-18, 4.201283422e-18,
                6.564505389999999e-18, -5.30581438e-18
               ]
verts = [36, 136, 528, 2080, 8256, 32896]
bverts = []

"""
-
HC.V.size() = 36
len(bV) = 0
sum_HNdA_i (point-wise) = 2.392492038148525
sum_HNdA_i _ 2  (point-wise) = -1.7618285302889447e-19
-
HC.V.size() = 36
len(HNdA_i_list) = 36
len(bV) = 0
sum_HNdA_i (total) = [-1.08420217e-19  1.08420217e-19 -3.25260652e-19]
sum_HNdA_i _ 2  (total) = [3.25260652e-19 1.19262239e-18 3.25260652e-19]
geo erange = 0.0
[polyscope] Backend: openGL3_glfw -- Loaded openGL version: 4.6 (Core Profile) Mesa 22.1.7

-
HC.V.size() = 136
len(bV) = 0
sum_HNdA_i (point-wise) = 10.837791012382999
sum_HNdA_i _ 2  (point-wise) = -4.431676380034499e-18
-
HC.V.size() = 136
len(HNdA_i_list) = 136
len(bV) = 0
sum_HNdA_i (total) = [ 1.35525272e-20 -4.33680869e-19  7.86046575e-19]
sum_HNdA_i _ 2  (total) = [-7.86046575e-19 -7.86046575e-19  8.13151629e-20]
geo erange = 0.0
[polyscope] Backend: openGL3_glfw -- Loaded openGL version: 4.6 (Core Profile) Mesa 22.1.7
-
HC.V.size() = 528
len(bV) = 0
sum_HNdA_i (point-wise) = 46.310079254685476
sum_HNdA_i _ 2  (point-wise) = -6.0986372202309624e-18
-
HC.V.size() = 528
len(HNdA_i_list) = 528
len(bV) = 0
sum_HNdA_i (total) = [-4.47233396e-19  1.84314369e-18  2.84603070e-19]
sum_HNdA_i _ 2  (total) = [-1.15196481e-18  1.45012041e-18 -2.84603070e-19]
geo erange = 0.0
-
HC.V.size() = 2080
len(bV) = 0
sum_HNdA_i (point-wise) = 190.1675853457836
sum_HNdA_i _ 2  (point-wise) = 9.23198149871407e-17
-
HC.V.size() = 2080
len(HNdA_i_list) = 2080
len(bV) = 0
sum_HNdA_i (total) = [9.69005692e-19 1.58564568e-18 1.64663205e-18]
sum_HNdA_i _ 2  (total) = [-3.45589442e-19  8.80914265e-20  8.80914265e-20]
geo erange = 0.0
-
HC.V.size() = 8256
len(bV) = 0
sum_HNdA_i (point-wise) = 769.0103098098057
sum_HNdA_i _ 2  (point-wise) = -5.455434281403937e-16
-
HC.V.size() = 8256
len(HNdA_i_list) = 8256
len(bV) = 0
sum_HNdA_i (total) = [-1.50433051e-18  1.31798327e-17 -5.11099680e-18]
sum_HNdA_i _ 2  (total) = [ 3.37627333e-18  1.81214229e-17 -9.31736242e-20]
geo erange = 0.0
-
HC.V.size() = 32896
len(bV) = 0
sum_HNdA_i (point-wise) = 3090.9577885704543
sum_HNdA_i _ 2  (point-wise) = 2.9201088162095212e-15
-
HC.V.size() = 32896
len(HNdA_i_list) = 32896
len(bV) = 0
sum_HNdA_i (total) = [-4.07592254e-18  2.04643160e-18 -3.27632344e-18]
sum_HNdA_i _ 2  (total) = [-6.26804381e-19  5.44472778e-18  2.40557357e-19]
geo erange = 0.0
"""

if 1:
    errors_point_wise_1 = np.abs(errors_point_wise_1)
    errors_point_wise_2 = np.abs(errors_point_wise_2)

    Nlist = verts
   # Nlist = [8, 16, 32, 64, 128]  # BOUNDARY Vertices (DELETE)
    #Nlist = bverts
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax2 = ax.twinx()
    ln1 = ax.loglog(Nlist, errors_point_wise_1*100, 'x', color='tab:blue',
                     label=r'Point-wise errors in $\mathbf{x_{+}}=(0, 0, 1)$ direction  ($\frac{\widehat{H\mathbf{N}} d A_{i} \cdot \mathbf{x_{+}}}{C_i}$)')
    ln2 = ax.loglog(Nlist, errors_point_wise_2*100, 'X', color='tab:red',
                     label=r'Point-wise errors in $\mathbf{x_{-}}=(0, 0, -1)$ direction  ($\frac{\widehat{H\mathbf{N}} d A_{i} \cdot \mathbf{x_{-}}}{C_i}$)')

    #ax2.set_ylabel(r'Young-Laplace error (%)')
    ax.set_xlabel(r'$n$ (number of vertices)')
    #ax.set_xlabel(r'$n$ (number boundary of vertices)')

    ln3 = ax.loglog(Nlist,  np.abs(errors_total)*100, 'o', color='tab:orange',
                    label=r'Integrated errors (sum of vector components in $\widehat{H\mathbf{N}} d A_{i}$ )')
    ax.set_ylabel(f'Integration error (%)')
    ax.set_ylabel(f'Error (%)')

    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    plt.tick_params(axis='y', which='minor')

    #ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())

    # ax2.set_ylim([1e-15, 1e-13])
   # ax.set_ylim([1e-20, 1e4])
#    ax2.set_ylim([1e-20, 1e4])

    plt.tick_params(axis='y', which='minor')
    plt.tick_params(axis='x', which='major', reset=True)
    #  ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    #print(help(plt.tick_params))
    #ax2.set_ylabel(r'Young-Laplace error (%)')

    #plt.savefig('./fig/errors.png', bbox_inches='tight', dpi=600)
    plt.savefig('./fig/assym_bridge_data_errors.pdf', bbox_inches='tight', dpi=600)

if 1:
    plt.show()

if 0:
    sref=0
    HC, bV, K_f, H_f, neck_verts, neck_sols = catenoid_N(r, theta_p, gamma, abc,
                                    sref=sref,
                                    refinement=4,
                                    cdist=1e-5, equilibrium=True)

    # Plot in polyscope and save .png image
    if 1:
        fn = f'./fig/assym_bridge_smooth.png'
        ps = plot_polyscope(HC, fn=fn)
if 1:
    HC.V.print_out()
    ps.show()
    #ax2.set_ylabel(r'Young-Laplace error (%)')
