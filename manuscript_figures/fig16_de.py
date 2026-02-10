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

# Parameters for a water droplet in air at standard laboratory conditions
gamma = 0.0728  # N/m, surface tension of water at 20 deg C
rho = 1000  # kg/m3, density
g = 9.81  # m/s2

# Numerical parameters #Stated this is what to plaay
r = 1
theta_p = 20 * np.pi/180.0  # rad, three phase contact angle
refinements = []# [2, 3, 4, 5, 6]   # NOTE: 2 is the minimum refinement needed for the complex to be manifold

# data containers:
lp_error = []
lp_error_2 = []
geo_error = []# Imports and physical parameters
import numpy as np
import scipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from ipywidgets import *
from matplotlib.widgets import Slider

# ddg imports

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

matplotlib.rcParams['font.size'] = 16

# Parameters for a water droplet in air at standard laboratory conditions
gamma = 0.0728  # N/m, surface tension of water at 20 deg C
rho = 1000  # kg/m3, density
g = 9.81  # m/s2

r = 1
theta_p = 20 * np.pi/180.0  # Three phase contact angle
#phi = 0.0
N = 8
#N = 5
N = 7

#F, nn, HC, bV, K_f, H_f = hyperboloid_N(r, theta_p, gamma, N=4, refinement=0, cdist=1e-10, equilibrium=True)

a, b, c = 1, 0.0, 1
abc = (a, b, c)
#HC, bV, K_f, H_f, neck_verts, neck_sols = hyperboloid_N(r, theta_p, gamma, abc, N=4, refinement=2, cdist=1e-10, equilibrium=True)
HC, bV, K_f, H_f, neck_verts, neck_sols = catenoid_N(r, theta_p, gamma, abc,
                                # refinement=2,
                                refinement=3,
                                # refinement=4,
                                #refinement=6,
                                #refinement=1,
                                cdist=1e-5, equilibrium=True)


if 0:
    cdist = 0.8
    HC.V.merge_all(cdist=cdist)
print(f'K_f = {K_f}')
print(f'H_f = {H_f}')

# active latest 10.10.20213
if 1:
    def int_curvatures(HC, bV, r, theta_p, printout=False):
        HNdA_ij = []
        HNdA_i = []
        HNdA_ij_sum = []
        HNdA_ij_dot = []
        C_ijk = []
        A_ijk = []
        N_i = []
        c_N_i = []
        int_V = []
        HNda_v_cache = {}
        C_ijk_v_cache = {}
        K_H_cache = {}
        HNdA_i_Cij = []
        HNdA_ij_dot_hnda_i = []
        for v in HC.V:
            if v in bV:
                continue
            else:
                #R = r / np.cos(theta_p)
                #N_f0 = np.array([0.0, 0.0, R * np.sin(theta_p)]) - v.x_a # First approximation

                nullp = np.zeros(3)
                nullp[2] = v.x_a[2]
                N_f0 = v.x_a - nullp # First approximation
                # N_f0 = v.x_a #- nullp # First approximation

                N_f0 = normalized(N_f0)[0]
                N_i.append(N_f0)
                F, nn = vectorise_vnn(v)
                # Compute discrete curvatures
                # c_outd = curvatures(F, nn, n_i=N_f0)
                #c_outd = curvatures_hn_i(F, nn, n_i=N_f0)
                c_outd = curvatures_hn_ij_c_ij(F, nn, n_i=N_f0)


                print(f"---")
                print(f"c_outd['n_i'] = {c_outd['n_i']}")
                print(f"c_outd['NdA_i'] = {c_outd['NdA_i']}")
                if 1:
                    #print(f"HNdA_i_Cij= {c_outd['HNdA_ij_Cij']}")
                    sum_HNdA_ij_Cij = np.sum(c_outd['HNdA_ij_Cij'], axis=0)
                    print(f"np.sum(HNdA_i_Cij, axis=0)) = {sum_HNdA_ij_Cij }")
                    print(f"np.sum(HNdA_i_Cij) = {np.sum(c_outd['HNdA_ij_Cij'])}")
                    HNdA_ij_Cij_dot_NdA_i = np.dot(c_outd['NdA_i'], sum_HNdA_ij_Cij )
                    print(f"HNdA_ij_Cij_dot_NdA_i = {HNdA_ij_Cij_dot_NdA_i}")
                    print(f"np.sum(HNdA_ij_Cij_dot_NdA_i ) = {np.sum(HNdA_ij_Cij_dot_NdA_i)}")
                    HNdA_ij_Cij_dot_n_i = np.dot(c_outd['n_i'], sum_HNdA_ij_Cij)
                    print(
                        f"np.sum(HNdA_ij_Cij_dot_n_i ) = {np.sum(HNdA_ij_Cij_dot_n_i)}")

                # test for plots:
                HNdA_ij_dot_hnda_i.append(
                                            np.sum(HNdA_ij_Cij_dot_NdA_i)
                )

                new_HNdA_ij_dot_hnda_i = HNdA_ij_dot_hnda_i
                #########################################
                HNda_v_cache[v.x] = c_outd['HNdA_ij']
                HNdA_i.append(c_outd['HNdA_i'])
                HNdA_ij.append(c_outd['HNdA_ij'])
                HNdA_ij_sum.append(np.sum(c_outd['HNdA_ij']))
                HNdA_ij_dot.append(np.sum(np.dot(c_outd['HNdA_ij'], c_outd['n_i'])))

                HNdA_i_Cij.append(c_outd['HNdA_ij_Cij'])



                #print(f"np.sum(HNdA_i_Cij) = {np.dot(c_outd['HNdA_ij'], c_outd['n_i'])}")
               # print(f'HNdA_ij = {HNdA_ij}')
               # print(f'C_ijk = {C_ijk}')
                # New normalized dot produic
                #HNdA_ij_dot.append(np.sum(np.dot(c_outd['HNdA_ij'], c_outd['n_i'])))
                #print(c_outd['C_ijk'])
                C_ijk.append(np.sum(c_outd['C_ijk']))
                C_ijk_v_cache[v.x] = np.sum(c_outd['C_ijk'])
                A_ijk.append(np.sum(c_outd['A_ijk']))
                #KdA += c_outd['Omega_i']  # == c_outd['K']
                int_V.append(v)
                # New
                #h_disc = (1 / 2.0) * np.sum(np.dot(c_outd['HNdA_ij'], c_outd['n_i'])) / np.sum(c_outd['C_ijk'])
                # h_disc = (1 / 2.0) * np.sum(np.dot(c_outd['HNdA_ij'], N_f0)) / np.sum(c_outd['C_ijk'])


                h_disc = (1 / 2.0) * np.sum(
                    np.dot(c_outd['HNdA_ij'], c_outd['n_i'])) / np.sum(
                    c_outd['C_ijk'])
                if 0:
                    print(f"c_outd['C_ijk'] = {c_outd['C_ijk']}")
                    print(f"c_outd['HNdA_ij'] = {c_outd['HNdA_ij']}")
                    print(f"c_outd['HNdA_ij'] / c_outd['C_ijk']  = "
                          f"{np.dot(c_outd['HNdA_ij'], c_outd['n_i']) / c_outd['C_ijk']}")
                    print(f'h_disc = {h_disc}')

                K_H_cache[v.x] = (h_disc / 2.0) ** 2

                c_N_i.append(c_outd['n_i'])

        H_disc = (1 / 2.0) * np.array(HNdA_ij_dot) / C_ijk
        K_H = (H_disc / 2.0)**2
        K_H = (H_disc )**2

        # Adjust HNdA_ij_sum and HNdA_ij_dot
        HNdA_ij_sum = 0.5 * np.array(HNdA_ij_sum) / C_ijk
        HNdA_ij_dot = 0.5 * np.array(HNdA_ij_dot) / C_ijk

        # New normalized dot product odeas
        HNdA_ij_dot_hnda_i = []
        K_H_2 = []
        HN_i = []

        # Old method that works with convex surfaces
        if 1:
            for hnda_ij, c_ijk, n_i in zip(HNdA_ij, C_ijk, N_i):
           # for hnda_ij, c_ijk, n_i in zip(HNdA_ij, C_ijk, c_N_i):
                if 1:
                    hnda_i = np.sum(hnda_ij, axis=0)
                    # print(f'hnda_i = {hnda_i}')
                    n_hnda_i = normalized(hnda_i)[0]
                    hndA_ij_dot_hnda_i = 0.5 * np.sum(np.dot(hnda_ij, n_hnda_i)) / c_ijk
                elif 0: # Appears to be more accurate, sadly
                    hndA_ij_dot_hnda_i = 0.5 * np.sum(np.dot(hnda_ij, n_i)) / c_ijk

                # Prev. converging working, changed on 2021-06-22:
                elif 0:
                    hndA_ij_dot_hnda_i = 0.5 * np.sum(hnda_ij) / c_ijk

                # Latest attempt 2021-06-22:
                elif 0:
                    print(f'hnda_ij = {hnda_ij}')
                    print(f'c_ijk = {c_ijk}')
                    sum_HNdA_ij_Cij = np.sum(hnda_ij, axis=0)
                    print(f'sum_HNdA_ij_Cij = {sum_HNdA_ij_Cij}')
                    hndA_ij_dot_hnda_i = 0.5 * np.linalg.norm(sum_HNdA_ij_Cij) / c_ijk

                # Latest formulation
                elif 0:
                    pass
                    #hndA_ij_dot_hnda_i = 0.5 * sum_HNdA_ij_Cij
                HNdA_ij_dot_hnda_i.append(hndA_ij_dot_hnda_i)
                #k_H_2 = (hndA_ij_dot_hnda_i/2.0) ** 2
                k_H_2 = (hndA_ij_dot_hnda_i/2.0) ** 2  # Modifcation 14.01.2020
                K_H_2.append(k_H_2)

                HN_i.append(0.5 * np.sum(np.dot(hnda_ij, n_i)) / c_ijk)
                hnda_i_sum = 0.0
                for hnda_ij_sum in HNdA_ij_sum:
                    hnda_i_sum += hnda_ij_sum

        HNdA_ij_dot_hnda_i

        if printout:
            print(f'H_disc = {H_disc}')
            print(f'HNdA_i = {HNdA_i}')
            print(f'HNdA_ij = {HNdA_ij}')
            print(f'HNdA_ij_sum = {HNdA_ij_sum}')
            print(f'HNdA_ij_dot = {HNdA_ij_dot}')

            print(f'=' * len('Discrete (New):'))
            print(f'Discrete (New):')
            print(f'=' * len('Discrete (New):'))
            print(f'K_H = {K_H}')
            print('-')
            print('New:')
            print('-')
            print(f'hnda_i_sum = {hnda_i_sum}')
            print(f'K_H_2 = {K_H_2}')
            print(f'C_ijk = {C_ijk}')
            print(f'np.sum(C_ijk) = {np.sum(C_ijk)}')
            print(f'A_ijk = {A_ijk}')
            print(f'np.sum(A_ijk) = {np.sum(A_ijk)}')
            print(f'K_H_2 dA = {np.sum(K_H_2) * np.sum(C_ijk)}')
            print(f'K_H_2 dA = {np.sum(K_H_2) * np.sum(A_ijk)}')
            print(f'np.sum(K_H_2) = {np.sum(K_H_2)}')
            print(f'HNdA_ij_dot_hnda_i = {np.array(HNdA_ij_dot_hnda_i)}')
            print(f'np.sum(HNdA_ij_dot_hnda_i) = {np.sum(np.array(HNdA_ij_dot_hnda_i))}')

            print(f'K_H_2 - K_f = {K_H_2 - K_f}')
            print(f'HNdA_ij_dot_hnda_i  - H_f = {HNdA_ij_dot_hnda_i -  H_f}')

            print(f'np.sum(C_ijk) = {np.sum(C_ijk)}')

        #HNdA_ij_dot_hnda_i = new_HNdA_ij_dot_hnda_i
        return HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i, HNdA_ij_dot_hnda_i,  K_H, K_H_2, HNdA_i_Cij

    (HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i,  HNdA_ij_dot_hnda_i, K_H,
         K_H_2, HNdA_i_Cij) = int_curvatures(HC, bV, r, theta_p, printout=0)

    # Method, new test 2021-07-02
    # Normals modification 2022-01-14
    if 1:
        c_outd2 = []
        HN_i_2 = []
        HNdA_i_list = []
        C_ij_i_list = []
        for v in HC.V:
            if v in bV:
                continue
            nullp = np.zeros(3)
            nullp[2] = v.x_a[2]
            N_f0 = v.x_a - nullp  # First approximation
            # N_f0 = v.x_a #- nullp # First approximation

            N_f0 = normalized(N_f0)[0]
            #N_i.append(N_f0)
            F, nn = vectorise_vnn(v)

            c_outd2 = b_curvatures_hn_ij_c_ij(F, nn, n_i=N_f0)
            HN_i_2.append(c_outd2['HN_i'])
            HNdA_i_list.append(c_outd2['HNdA_i'])
            C_ij_i_list.append(c_outd2['C_ij'])

        if 0:
            # Use the second approximation for the normals
            for v in HC.V:
                if v in bV:
                    continue
                nullp = np.zeros(3)
                nullp[2] = v.x_a[2]
                N_f0 = v.x_a - nullp  # First approximation
                # N_f0 = v.x_a #- nullp # First approximation

                N_f0 = normalized(N_f0)[0]
                #N_i.append(N_f0)
                F, nn = vectorise_vnn(v)

                c_outd2 = b_curvatures_hn_ij_c_ij(F, nn, n_i=N_f0)
                HN_i_2.append(c_outd2['HN_i'])
                HNdA_i_list.append(c_outd2['HNdA_i'])
                C_ij_i_list.append(c_outd2['C_ij'])

    # plot results
    if 1:
         print('-')
         print(f'=' * len('Discrete (New):'))
         K_f = np.array(K_f)
         H_f = np.array(H_f)  # Redefine H --> H_f
         print(f'K_H_2 - K_f = {K_H_2 - K_f}')
         print(f'K_f = { K_f}')
         print(f'len(HN_i) = {len(HN_i)}')
         fig = plt.figure()
         # ax = fig.add_subplot(2, 1, 1)
         ax = fig.add_subplot(1, 1, 1)
        # yr = (-np.array(K_H_2) - K_f)/K_f
         yr = (np.array(K_H) - K_f)/K_f
         xr = list(range(len(yr)))
         ax.plot(xr[:27], np.abs(yr)[:27], 'o',
                 label=r'$|| \left( \hat{K}_i(u, v) - K(u, v) \right) /K(u, v) ||$')
         print(f'HNdA_ij_dot_hnda_i  - H_f = {HNdA_ij_dot_hnda_i - H_f}')
       #  yr = HNdA_ij_dot_hnda_i - H_f
         yr = HN_i - H_f
         #yr = hndA_ij_dot_hnda_i - H_f
         # yr = HNdA_ij_Cij_dot_NdA_i - H_f
         ax.plot(xr[:27], np.abs(yr)[:27], 'x',
                 label='$|| \hat{HN_i}(u, v)  - H(u, v) ||$')

         yr = HN_i_2  - H_f
       #  plt.plot(xr, yr, 'x', label='$\hat{H}N_i_2(u, v)  - H(u, v)$')

         l_hnda_i_cij = []
         for hnda_i_cij in HNdA_i_Cij:
             print(f'h = {hnda_i_cij}')

             l_hnda_i_cij.append(np.sum(hnda_i_cij))

         #print(f'HNdA_i_Cij = {HNdA_i_Cij}')
         yr = l_hnda_i_cij - H_f
         #plt.plot(xr, yr, 'x', label='HNdA_i_Cij - $H$')
        # yr = HNdA_i_Cij
        # plt.plot(xr, yr, label='HNdA_i_Cij - H_f')
         plt.ylabel('Difference (Normal mean curvature ($m^{-1}$))')
         ax2 = ax.twinx()
        # plt.ylabel('Difference (Gaussian curvature ($m^{-2}$))')
         plt.ylabel('Relative difference (Gaussian curvature (-))')
         ax.set_xlabel('Vertex No.')
         ax.legend(loc="upper left",
              #bbox_transform=fig.transFigure,
         ncol=1)
         ax.set_ylim([0, 3.5])  # Prev 1.4
         ax2.set_ylim([0, 3.5])


sum_HNdA_i = 0.0
for hndA_i, c_ij in zip(HNdA_i_list, C_ij_i_list):
    #print(f'hndA_i = {hndA_i}')
    hndA_i_c_ij = hndA_i / np.sum(c_ij)

    sum_HNdA_i += np.dot(hndA_i_c_ij, [0, 0, 1])
   # sum_HNdA_i += np.dot(hndA_i_c_ij, [0, 0, -1])

print('-')
print(f'HC.V.size() = {HC.V.size()}')
print(f'len(bV) = {len(bV)}')
print(f'sum_HNdA_i = {sum_HNdA_i}')

for hndA_i, c_ij in zip(HNdA_i_list, C_ij_i_list):
    hndA_i_c_ij = hndA_i / np.sum(c_ij)
    sum_HNdA_i += np.dot(hndA_i_c_ij, [0, 0, -1])

print(f'sum_HNdA_i _ 2  = {sum_HNdA_i}')

if 1:
    max_int_e = 0.0
    ern = 0.0
    for v in bV:
        # for v in HC.V:
        for v2 in v.nn:
            if v2 in bV:
                a = v.x_a
                b = v2.x_a
                if N == 14:
                    print(
                        f'numpy.linalg.norm(a - b) = {numpy.linalg.norm(a - b)}')
                    continue
                break
        # ern = 2*numpy.linalg.norm(a - b)
        ern = 0.5 * numpy.linalg.norm(a - b) ** 2
        max_int_e = ern
        break

    #erange.append(max_int_e / r)
    print(f'geo erange = {ern }')




# Plot complex
if 0:
    fig_complex, ax_complex, fig_surface, ax_surface = HC.plot_complex(point_color=db, line_color=lb,
         complex_color_f=lb, complex_color_e=db)

    # Solid nanoparticles:
    if 1:

        np_dist = 4.5
        np_1 = 0.0  # position of nanoparticle 1
        np_1 = -np_dist  # position of nanoparticle 1
        np_1 = np.array([-np_dist , 0.0, 0.0])  # position of nanoparticle 1
        R_1 = 2.0  # nm, radius of nanoparticle 1
        delta_1 = 0.6  # nm, hydrate layer thickness of nanoparticle 1
        #np_2 = np_dist  # nm, radius of nanoparticle 2
        np_2 = np.array([np_dist , 0.0, 0.0])  # nm, radius of nanoparticle 2
        R_2 = 2.0  # nm, radius of nanoparticle 2
        delta_2 = 0.6  # nm, hydrate layer thickness of nanoparticle 2

        u = np.linspace(0, 2 * np.pi, N)
        v = np.linspace(0, np.pi, N)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        # x = np.cos(u) * np.sin(v)
        # y = np.sin(u) * np.sin(v)
        # z = np.cos(v)
        ax_complex.plot_surface(x * R_1, y * R_1, (z + np_2[0] / 2.0) * R_1,
                                linewidth=0.0,
                                color=do, alpha=0.3
                                # , cstride=stride, rstride=stride
                                )

        ax_complex.plot_surface(x * R_1, y * R_1, (z + np_1[0] / 2.0) * R_1,
                                linewidth=0.0,
                                color=do, alpha=0.3
                                # , cstride=stride, rstride=stride
                                )

#HNdA_ij_dot_hnda_i
# Cross section of meniscus neck

# neck_sols = [(0.0, -1.0), (0.0, -1.0), (0.0, -1.0), (0.0, -1.0)]
#TODO: H_f is always precisely zero at this point? Why?
if 1:
     A_m = 2 * np.pi * a**2
     U_m = 2 * np.pi * a
    # dP_f = -gamma * H_f_neck
     dP_f = -gamma * 0.0  # H_f_neck == 0.0 always
     dP = -gamma
     F_cap = dP * A_m + 2 * gamma * U_m
     F_cap_f = dP_f * A_m + 2 * gamma * U_m
     #

     # Errors
     plt.figure()
     # plt.plot(N, lp_error, 'x', label='$\Delta p \frac{\Delta p - \Delta p}{\Delta p}$')

     if 0:
         Nmax = 21
         lp_error = np.zeros(Nmax - 3)
         Nlist = list((range(3, Nmax)))

         for v in HC.V:
             if v in bV:
                 continue
             max_lp
             #numpy.linalg.norm(a - b)


         plt.plot(Nlist, lp_error, 'x')
         plt.plot(Nlist, lp_error, 'x',
                   label='Young-Laplace error: $(\Delta p - \Delta\hat{ p})/\Delta p $')
         plt.plot(Nlist, geo_error, 'X',
                   label='Integration error (Trapezoidal rule $O(h^3)$)')


# [36, 120, 528, 2080]
#len(bV) = 8, 16, 32, 64
plot.figure()
# plt.plot(N, lp_error, 'x', label='$\Delta p \frac{\Delta p - \Delta p}{\Delta p}$')

"""
#NOTE: Data values recorded data from refinements 2-5
Change refinement on top of the script and then read from CLI output ex:
-
HC.V.size() = 136
len(bV) = 16
sum_HNdA_i = 2.4001525593397854e-16
sum_HNdA_i _ 2  = -6.2341624917916505e-19
"""

#Nlist = [8, 16, 32, 64]
Nlist = [8, 16, 32, 64]  # BOUNDARY Vertices
N_total = [36, 136, 528, 2080]
lp_error = [0, 0, 0, 0]
lp_error = np.array(lp_error) +  np.random.rand(4) * 4 *np.random.rand(4) * 1e-12

lp_error = [4.5902409477605044e-17,
            2.4001525593397854e-16,
            -3.5415463964239e-16,
            -1.652188585596348e-16
            ]
lp_error_2 = [-4.0657581468206416e-20,
              -6.2341624917916505e-19,
               2.710505431213761e-19,
               5.624298769768554e-18
              ]
# geo_erro should be proportion to area based
#geo_error = [a/(2**2+1), a/(2**3+1), a/(2**4+1), a/(2**5+1)]  # area based
geo_error = [5.533830997888883,
             1.6208215733413345,
             0.4212378025628348,
             0.1063310109203463

             ]
if 0:
    fig,  (ax1, ax2) = plt.subplots(1, 2)
    ax2.plot(Nlist, np.abs(lp_error), 'x')
    ax1.semilogy(Nlist, np.abs(lp_error), 'x')
   # ax1.plot(Nlist, np.abs(lp_error), 'x')
   # ax2.semilogy(Nlist, np.abs(lp_error), 'x')
    #plt.plot(Nlist, lp_error, 'x', label='Young-Laplace error: $(\Delta p - \Delta\hat{ p})/\Delta p $')
    ax2.plot(Nlist, np.abs(lp_error), 'x',
             label='Capillary force error: $(F_{cap} - \hat{F_{cap}}) / F_{cap} $')
    ax2.plot(Nlist, geo_error, 'X', label='Integration error (p-normed trapezoidal rule $O(h^3)$)')

    ax1.semilogy(Nlist, np.abs(lp_error), 'x',
             label='Capillary force error: $(F_{cap} - \hat{F_{cap}}) / F_{cap} $')
    ax1.semilogy(Nlist, geo_error, 'X', label='Integration error (p-normed trapezoidal rule $O(h^3)$)')

    plt.xlabel(r'N (number of boundary vertices)')
    plt.ylabel(r'Error (%)')
    plt.tick_params(axis='y', which='minor')

if 1:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax2 = ax.twinx()
    ln1 = ax.loglog(Nlist, np.abs(lp_error), 'x', color='tab:blue',
                 label=r'Kapillarkraftfehler: $(F_{cap} - \hat{F_{cap}}) / F_{cap} $')

    ax2.set_ylabel(r'Kapillarkraftfehler (%)')
    ax.set_xlabel(r'$n$ (Anzahl der Scheitelpunkte)')

    #ln2 = ax.loglog(Nlist, geo_error, 'X', color='tab:orange',
    #          label='Integration error (p-normed trapezoidal rule $O(h^2)$)')

    ax.set_ylabel(r'Integration error (%)')

    ax.set_ylim([1e-15, 1e-18])
    lns = ln1# + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    plt.tick_params(axis='y', which='minor')

    x, y = Nlist, geo_error
    x, y = np.float32(x), np.float32(y)
    z = numpy.polyfit(x, y, 1)
    z = np.polyfit(np.log10(x), np.log10(y), 1)

    p = numpy.poly1d(z)
    #ax.loglog(x, 10 **p(x), "r--")
    p = np.poly1d(z)
    # the line equation:
    print("y=%.6fx+(%.6f)" % (z[0], z[1]))

    lns = ln1 #+ ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=1
              )

    plt.tick_params(axis='y', which='minor')
    if 1:
        from matplotlib import pyplot as plt, ticker as mticker

        ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
        #ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())
        # ax.xaxis.get_major_formatter().set_scientific(False)
        # ax2.xaxis.get_major_formatter().set_scientific(False)
        # ax.tic

   # fig.set_size_inches(10,6)

    p#lt.savefig('fig15.png', dpi=600)
    plt.tight_layout()
    plt.savefig('fig15.png', bbox_inches='tight', dpi=600)
    plt.show()


if 0:
    plt.show()



def plot_polyscope(HC):
    # Initialize polyscope
    ps.init()
    ps.set_up_dir("z_up")

    do = coldict['db']
    lo = coldict['lb']
    HC.dim = 2  # The dimension has changed to 2 (boundary surface)
    HC.vertex_face_mesh()
    points = np.array(HC.vertices_fm)
    triangles = np.array(HC.simplices_fm_i)
    ### Register a point cloud
    # `my_points` is a Nx3 numpy array
    my_points = points
    ps_cloud = ps.register_point_cloud("my points", my_points)
    ps_cloud.set_color(tuple(do))
    #ps_cloud.set_color((0.0, 0.0, 0.0))
    verts = my_points
    faces = triangles
    ### Register a mesh
    # `verts` is a Nx3 numpy array of vertex positions
    # `faces` is a Fx3 array of indices, or a nested list
    surface = ps.register_surface_mesh("my mesh", verts, faces,
                                         color=do,
                                         edge_width=1.0,
                                         edge_color=(0.0, 0.0, 0.0),
                                         smooth_shade=False)

    # Add a scalar function and a vector function defined on the mesh
    # vertex_scalar is a length V numpy array of values
    # face_vectors is an Fx3 array of vectors per face
    if 0:
        ps.get_surface_mesh("my mesh").add_scalar_quantity("my_scalar",
                vertex_scalar, defined_on='vertices', cmap='blues')
        ps.get_surface_mesh("my mesh").add_vector_quantity("my_vector",
                face_vectors, defined_on='faces', color=(0.2, 0.5, 0.5))

    # View the point cloud and mesh we just registered in the 3D UI
    #ps.show()

    np_dist = 4.0
    radius = 3.0

    points = np.array([[0, 0, np_dist],
                       [0, 0, -np_dist]])

    cloud = ps.register_point_cloud("my points", points)
    cloud.set_enabled(False)  # disable
    cloud.set_enabled()  # default is true
    cloud.set_radius(radius, relative=False)  # radius in absolute world units
    cloud.set_color(tuple(do))  # rgb triple on [0,1]
    print(f'tuple(do) = tuple(do)')
    #cloud.set_material("candy")
 #   cloud.set_transparency(1.0)
    if 0:
        cloud1 = ps.register_point_cloud("my points 1", points, enabled=False,
                                # material='candy',
                                radius=radius, color=tuple(do),
                                # transparency = 0.5
                                )

    #print(help(ps.register_point_cloud))
    if 1:
        cloud2 = ps.register_point_cloud("my points 2", points)
        cloud2.set_enabled(False)  # disable
        cloud2.set_enabled()  # default is true
        cloud2.set_radius(radius + 0.35, relative=False)  # radius in absolute world units
        #cloud2.set_radius(radius + 0.4, relative=False)  # radius in absolute world units

        cloud2.set_color(tuple(lb))  # rgb triple on [0,1]

    print(f'tuple(do) = {tuple(do)}')

    def rgb2hex(r, g, b):
        return "#{:02x}{:02x}{:02x}".format(r, g, b)
    print(f'tuple(do) = {tuple(do)}')
    hex = rgb2hex(int(tuple(do)[0]*255),
                  int(tuple(do)[1]*255),
                  int(tuple(do)[2]*255))
    print(f'rgb2hex( = { hex}')
    ps.show()


if 0:
    plot_polyscope(HC)



# visualize!# Average data norm error:

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
#ps_cloud = ps.register_point_cloud("my points", points)
#ps.show()


             #h = np.sum(np.sum(hnda_i_cij), axis=0)
             #l_hnda_i_cij.append(np.linalg.norm(h))

             #h = np.sum(np.sum(hnda_i_cij), axis=0)
             #l_hnda_i_cij.append(np.linalg.norm(h))

# Run full simulation
if 0:
    Nlist = []

    for refinement in refinements:
        # Construct the new Catenoid
        a, b, c = 1, 0.0, 1  # Geometric parameters of the catenoid
        abc = (a, b, c)
        HC, bV, K_f, H_f, neck_verts, neck_sols = catenoid_N(r, theta_p, gamma, abc, refinement=refinement,
                                                             cdist=1e-5, equilibrium=True)
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
        # Append number of boundary vertices:
        Nlist.append(len(bV))

    # Plot the final results
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax2 = ax.twinx()
    ln1 = ax.loglog(Nlist, np.abs(lp_error), 'x', color='tab:blue',
                 label='Capillary force error: $(F_{cap} - \hat{F_{cap}}) / F_{cap} $')

   # ax2.set_ylabel(r'Capillary force error (%)')
    ax.set_xlabel(r'$n$ (Anzahl der Scheitelpunkte)')

   # ln2 = ax.loglog(Nlist, geo_error, 'X', color='tab:orange',
    #          label='Integration error (p-normed trapezoidal rule $O(h^2)$)')

    ax.set_ylabel(r'Young-Laplace Fehler (%)')

    ax.set_ylim([1e-15, 1e-18])
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

    lns = ln1# + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    plt.tick_params(axis='y', which='minor')
    from matplotlib import pyplot as plt, ticker as mticker

    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    #ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())



plt.tight_layout()
plt.show()
