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
from ddgclib._complex import *
from ddgclib._curvatures import * #plot_surface#, curvature
from ddgclib._capillary_rise_flow import * #plot_surface#, curvature
from ddgclib._eos import *
from ddgclib._misc import *
from ddgclib._plotting import *


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
r = 0.5  # Radius of the tube (20 mm)
r = 1 # Radius of the tube (20 mm)

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
refinement = 0
#N = 20
#cdist = 1e-10
cdist = 1e-10

r = np.array(r, dtype=np.longdouble)
theta_p = np.array(theta_p, dtype=np.longdouble)

##################################################
# PLot theta rise
if 1:
    # First generate the smooth data
    domain = Theta_p = np.linspace(0.0, 0.5*np.pi, 100)
    c_outd_list, c_outd, vdict, X = out_plot_cap_rise(N=N, r=r, gamma=gamma, refinement=refinement,
                                                      domain=domain)
    ylabel = r'$m$ or $m^{-1}$'
    keyslabel = None
    keyslabel = {'K_f': {'label': '$K$',
                         'linestyle':  '-',
                         'marker': None},
                'H_f': {'label': '$H$',
                        'linestyle':  '--',
                        'marker': None},
    }

    print(f'vdict.keys() = {vdict.keys()}')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ind = 0
    Lines = {}
    fig.legend()
    key = 'K_f'
    value = vdict['K_f']

    # Normal good for K = 1 vs. loglog plots
    ax.plot(Theta_p * 180 / np.pi, value,
            marker=keyslabel[key]['marker'],
            linestyle=keyslabel[key]['linestyle'],
            label=keyslabel[key]['label'], alpha=0.7)

    key = 'H_f'
    value = vdict['H_f']
    ax.plot(Theta_p * 180 / np.pi, value,
            marker=keyslabel[key]['marker'],
            linestyle=keyslabel[key]['linestyle'],
            label=keyslabel[key]['label'], alpha=0.7)




    # Next genereate discrete dat apoints
    domain = Theta_p = np.linspace(0.0, 0.5*np.pi, 10)
    c_outd_list, c_outd, vdict, X = out_plot_cap_rise(N=N, r=r, gamma=gamma, refinement=refinement,
                                                      domain=domain)
    ylabel = r'$m$ or $m^{-1}$'
    keyslabel = None


    """
    vdict.keys() = dict_keys(['K_f', 'K/C_ijk', ' 0.5 * KNdA_ij_sum / C_ijk', '- 0.5 * KNdA_ij_dot / C_ijk',
                              '((K/C_ijk)^0.5 + (K/C_ijk)^0.5)', 'H_f', '2 * H_i/C_ijk = H_ij_sum/C_ijk',
                              ' -(1 / 2.0) * HNdA_ij_dot/C_ijk', '(1/2)*HNdA_ij_sum/C_ijk', 'K_H', 'K_H 2'])
    """
    keyslabel = {'K/C_ijk': {'label': r'$\sum_{i \in s t\left(v_{i}\right)} \frac{\Omega_{i}}{C_{i j k}}$',
                         'linestyle':  'None',
                          'marker': 'o'},
                ' 0.5 * KNdA_ij_sum / C_ijk': {'label': r'$\int_{\star s t\left(v_{i}\right)} \frac{\langle K, N\rangle}{C_{i j k}} d A=\frac{\frac{1}{2} \sum_{i j \in S_{\mathrm{t}(i)}} \frac{\varphi_{i j}}{\ell_{i j}}\left(\mathbf{f}_{\mathrm{j}}-\mathbf{f}_{\mathrm{i}}\right)}{C_{i j k}}$',
                          'linestyle':  'None',
                          'marker': 'x'},
                 '((K/C_ijk)^0.5 + (K/C_ijk)^0.5)': {'label': r'$2 \sqrt{\frac{\Omega_{i}}{C_{i j k}}}$',
                          'linestyle': 'None',
                          'marker': 'o'},
                '2 * H_i/C_ijk = H_ij_sum/C_ijk': {'label': r'$\sum_{i j \in s t\left(v_{i}\right)}\left(H_{i j}\right)=\frac{\sum_{i j \in s t\left(v_{i}\right)}\left(\frac{1}{2} \ell_{i j} \varphi_{i j}\right)}{C_{i j k}}$',
                          'linestyle': 'None',
                          'marker': 'o'},
                ' -(1 / 2.0) * HNdA_ij_dot/C_ijk': {'label': r'$\int_{\star s t\left(v_{i}\right)} \frac{H \cdot N}{C_{i j k}} d A=\frac{\frac{1}{2} \sum_{i j \in s t\left(v_{j}\right)}\left(\cot \alpha_{i j}+\cot \beta_{i j}\right)\left(\mathbf{f}_{\mathbf{i}}-\mathbf{f}_{\mathbf{j}}\right)}{C_{i j k}}$',
                         'linestyle': 'None',
                         'marker': 'o'},
                '(1/2)*HNdA_ij_sum/C_ijk': {'label': r'$\int_{\star s t\left(v_{i}\right)} \frac{\langle H, N\rangle}{C_{i j k}} d A=\frac{\frac{1}{2} \sum_{i j \in s t\left(v_{j}\right)}\left(\cot \alpha_{i j}+\cot \beta_{i j}\right)\left(\mathbf{f}_{\mathbf{i}}-\mathbf{f}_{\mathbf{j}}\right)}{C_{i j k}}$',
                         'linestyle': 'None',
                         'marker': 'X'},

    }




    keys = keyslabel.keys()
    for key in keys:

        value = vdict[key]
        ax.plot(Theta_p * 180 / np.pi, value,
                marker=keyslabel[key]['marker'],
                linestyle=keyslabel[key]['linestyle'],
                label=keyslabel[key]['label'], alpha=0.7)








    #plt.ylim((0, max(max( vdict['K_H_i']), max( vdict['HN_i']))))
    ax.legend(#bbox_to_anchor=(0.15, 0.15),
              loc="upper right",
              bbox_transform=fig.transFigure, ncol=3)

    #fig.legend(ncol=3)




    if 0:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        lstyles = ['-', '--', '-.', ':']





        mod = len(lstyles)
        ind = 0
        Lines = {}
        fig.legend()
        for key, value in vdict.items():
            if keyslabel is None:
                line, = ax.plot(X, value,
                                linestyle=lstyles[ind],
                                label=key, alpha=0.7)

            else:
                line, = ax.plot(X, value,
                                marker=keyslabel[key]['marker'],
                                linestyle=keyslabel[key]['linestyle'],
                                label=keyslabel[key]['label'], alpha=0.7)

            Lines[key] = line
            # plot.plot(X, value, linestyle=lstyles[ind], label=key, alpha=0.7)
            ind += 1
            ind = ind % mod

        plot.xlabel(r'Contact angle $\Theta_{C}$')
        plot.ylabel(ylabel)
        # fig.legend(bbox_to_anchor=(1, 0.5), loc='right', ncol=2)
        fig.legend(ncol=2)
        # interact(update);


# PLot theta rise with New fomulation over Phi_C (2021-07)
# NOTE: THIS IS CURRENTLY USED IN THE MANUSCRIPT:
if 0:
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

    # Plot the values from new_out_plot_cap_rise
    if 1:
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
        if 1:
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

        else:
            ax.semilogy(Theta_p * 180 / np.pi, value,
                    marker=keyslabel[key]['marker'],
                    linestyle=keyslabel[key]['linestyle'],
                    label=keyslabel[key]['label'], alpha=0.7)

            key = 'K_H_i'
            value = vdict[key]
            ax.semilogy(X, value,
                    marker=keyslabel[key]['marker'],
                    linestyle=keyslabel[key]['linestyle'],
                    markerfacecolor='None',
                    label=keyslabel[key]['label'], alpha=0.7)

            key = 'H_f'
            value = H_fl
            ax.semilogy(Theta_p * 180 / np.pi, value,
                    marker=keyslabel[key]['marker'],
                    linestyle=keyslabel[key]['linestyle'],
                    label=keyslabel[key]['label'], alpha=0.7)

            key = 'HN_i'
            value = vdict[key]
            ax.semilogy(X, value,
                    marker=keyslabel[key]['marker'],
                    linestyle=keyslabel[key]['linestyle'],
                    markerfacecolor='None',
                    label=keyslabel[key]['label'], alpha=0.7)

        if 1:
            plot.xlabel(r'Contact angle $\Theta_{C}$ ($^\circ$)')
            plot.ylabel(r'Gaussian curvature ($m^{-2}$)')
            ax2 = ax.twinx()
            plt.ylabel('Mean normal curvature ($m^{-1}$)')
        else:
            plot.xlabel(r'Contact angle $\Theta_{C}$')
            plot.ylabel(r'$K$ ($m^{-1}$)')
            ax2 = ax.twinx()
            plt.ylabel('$H$ ($m^{-1}$)')

        plt.ylim((0, max(max( vdict['K_H_i']), max( vdict['HN_i']))))
        ax.legend(bbox_to_anchor=(0.15, 0.15), loc="lower left",
                  bbox_transform=fig.transFigure, ncol=2)
        # interact(update);


plt.show()






