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

import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'CMU Serif, Times New Roman'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams["mathtext.fontset"]= 'cm'  # ValueError: Key mathtext.fontset: 'serif' is not a valid value for mathtext.fontset; supported values are ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']
matplotlib.rcParams["mathtext.default"] = 'it'


if 0:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    # for Palatino and other serif fonts use:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })
    # It's also possible to use the reduced notation by directly setting font.family:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        'font.size': 14
    })

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri"]})
    # for Palatino and other serif fonts use:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Calibri"],
    })
    # It's also possible to use the reduced notation by directly setting font.family:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Calibri",
        'font.size': 14
    })

plt.rcParams.update({
    "text.usetex": True,
   # "font.family": "sans-serif",
   # "font.sans-serif": ["Calibri"]
})

font_size = 12

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
    # Save
    Theta_p_smooth =  domain
    vdicsmooth = {'K_f':vdict['K_f'], 'H_f':vdict['H_f']}

    # Next genereate discrete dat apoints
    domain = Theta_p = np.linspace(0.0, 0.5*np.pi, 10)
    c_outd_list, c_outd, vdict, X = out_plot_cap_rise(N=N, r=r, gamma=gamma, refinement=refinement,
                                                      domain=domain)
    ylabel = r'$m$ or $m^{-1}$'
    keyslabel = None

    # Redefine smooth data:
    vdict['K_f'] = vdicsmooth['K_f']
    vdict['H_f'] = vdicsmooth['H_f']

    # Begin plotting
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ind = 0
    Lines = {}
    fig.legend()
    ylabel = r'$m$ or $m^{-1}$'


    # First plot the Gaussia curvatures:
    keyslabel = {
                'K_f': {'label': '$K$',
                        'linestyle': '-',
                        'marker': None,
                        'color':'tab:blue'},
                'K/C_ijk': {'label': r'$\sum_{i \in s t\left(v_{i}\right)} \frac{\Omega_{i}}{C_{i j k}}$',
                            'linestyle':  'None',
                            'marker': '^',
                            'color':'tab:green'},
                ' 0.5 * KNdA_ij_sum / C_ijk': {'label': r'$\int_{\star s t\left(v_{i}\right)} \frac{\langle K, N\rangle}{C_{i j k}} d A=\frac{\frac{1}{2} \sum_{i j \in S_{\mathrm{t}(i)}} \frac{\varphi_{i j}}{\ell_{i j}}\left(\mathbf{f}_{\mathrm{j}}-\mathbf{f}_{\mathrm{i}}\right)}{C_{i j k}}$',
                          'linestyle':  'None',
                          'marker': "D",
                          'color':'tab:purple'},

    }


    keys = keyslabel.keys()
    for key in keys:
        value = vdict[key]
        try:
            ax.plot(Theta_p * 180 / np.pi, value,
                    marker=keyslabel[key]['marker'],
                    linestyle=keyslabel[key]['linestyle'],
                    label=keyslabel[key]['label'],
                    color=keyslabel[key]['color'],
                    alpha=0.7)
        except ValueError:
            ax.plot(Theta_p_smooth* 180 / np.pi, value,
                    marker=keyslabel[key]['marker'],
                    linestyle=keyslabel[key]['linestyle'],
                    color=keyslabel[key]['color'],
                    label=keyslabel[key]['label'], alpha=0.7)

    plt.ylabel(r'Gaussian curvature ($m^{-2})$', fontsize=font_size)
    #plt.ylim((0, max(max( vdict['K_H_i']), max( vdict['HN_i']))))
    ax.legend(#bbox_to_anchor=(0.15, 0.15),
              #loc="upper center",
              #loc="lower left",
        loc="center right",
              bbox_transform=fig.transFigure, ncol=1)

    # Next we plot the mean normal curvatures:
    plt.xlabel(r'Contact angle $\Theta_{C}$ ($^\circ$)', fontsize=font_size)
    ax2 = ax.twinx()
    keyslabel = {

                'H_f': {'label': '$H$',
                        'linestyle': '--',
                        'marker': None,
                        'color':'tab:orange'},
                '2 * H_i/C_ijk = H_ij_sum/C_ijk': {'label': r'$\sum_{i j \in s t\left(v_{i}\right)}\left(H_{i j}\right)=\frac{\sum_{i j \in s t\left(v_{i}\right)}\left(\frac{1}{2} \ell_{i j} \varphi_{i j}\right)}{C_{i j k}}$',
                          'linestyle': 'None',
                          'marker': "s",
                          'color':'tab:red'},
                ' -(1 / 2.0) * HNdA_ij_dot/C_ijk': {'label': r'$\int_{\star s t\left(v_{i}\right)} \frac{H \cdot N}{C_{i j k}} d A=\frac{\frac{1}{2} \sum_{i j \in s t\left(v_{j}\right)}\left(\cot \alpha_{i j}+\cot \beta_{i j}\right)\left(\mathbf{f}_{\mathbf{i}}-\mathbf{f}_{\mathbf{j}}\right)}{C_{i j k}}$',
                         'linestyle': 'None',
                         'marker': "H",
                          'color':'tab:brown'},
                '(1/2)*HNdA_ij_sum/C_ijk': {'label': r'$\int_{\star s t\left(v_{i}\right)} \frac{\langle H, N\rangle}{C_{i j k}} d A=\frac{\frac{1}{2} \sum_{i j \in s t\left(v_{j}\right)}\left(\cot \alpha_{i j}+\cot \beta_{i j}\right)\left(\mathbf{f}_{\mathbf{i}}-\mathbf{f}_{\mathbf{j}}\right)}{C_{i j k}}$',
                         'linestyle': 'None',
                         'marker': 'x',
                         'color':'tab:olive'},
                '((K/C_ijk)^0.5 + (K/C_ijk)^0.5)': {'label': r'$2 \sqrt{\frac{\Omega_{i}}{C_{i j k}}}$',
                          'linestyle': 'None',
                          'marker': 'o',
                          'color': 'tab:pink'},
    }

    keys = keyslabel.keys()
    for key in keys:
        value = vdict[key]
        try:
            ax2.plot(Theta_p * 180 / np.pi, value,
                    marker=keyslabel[key]['marker'],
                    #markersize=1,
                    linestyle=keyslabel[key]['linestyle'],
                    color=keyslabel[key]['color'],
                    label=keyslabel[key]['label'], alpha=0.7)
        except ValueError:
            ax2.plot(Theta_p_smooth* 180 / np.pi, value,
                    marker=keyslabel[key]['marker'],
                    #markersize=1,
                    linestyle=keyslabel[key]['linestyle'],
                    color=keyslabel[key]['color'],
                    label=keyslabel[key]['label'], alpha=0.7)


   # ax2.legend(#bbox_to_anchor=(0.15, 0.15),
   #           loc="upper right",
   #           bbox_transform=fig.transFigure, ncol=3)

    # Finally add the plot labels
    #fig.legend(ncol=3)
    #plt.xlabel('Contact angle $\Theta_{C}$')
    plt.xlabel(r'Contact angle $\it{ \Theta_{C}$ ($^\circ}$)', fontsize=font_size)


    #ax2.set_ylim()
    #ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))

    plt.ylabel('Mean normal curvature ($m^{-1}$)', fontsize=font_size
               )
    plt.legend(fontsize=font_size-2)

fig.set_size_inches(10,6)

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
plt.savefig('./figs6_thesis.pdf', bbox_inches='tight', dpi=600)
plt.show()






