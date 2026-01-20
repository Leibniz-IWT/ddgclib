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
HC, bV, K_f, H_f, neck_verts, neck_sols = catenoid_N(r, theta_p, gamma, abc, N=4,
    refinement=2,
    # refinement=4,
    #refinement=6,
    #refinement=1,
    cdist=1e-5, equilibrium=True)


if 0:
    cdist = 0.8
    HC.V.merge_all(cdist=cdist)
print(f'K_f = {K_f}')
print(f'H_f = {H_f}')

fig, (ax1, ax2) = plt.subplots(1, 2)

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
                elif 1:
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
                k_H_2 = (hndA_ij_dot_hnda_i/2.0) ** 2
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
        return HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i, HNdA_ij_dot_hnda_i, K_H_2, HNdA_i_Cij

    (HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i,  HNdA_ij_dot_hnda_i,
         K_H_2, HNdA_i_Cij) = int_curvatures(HC, bV, r, theta_p, printout=0)


    # Method, new test 2021-07-02
    if 1:
        c_outd2 = []
        HN_i_2 = []
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

    # plot results
    if 1:
         print('-')
         print(f'=' * len('Discrete (New):'))
         K_f = np.array(K_f)
         H_f = np.array(H_f)
         print(f'K_H_2 - K_f = {K_H_2 - K_f}')
         #plt.figure()
         yr = K_H_2 - K_f
         xr = list(range(len(yr)))
         ax1.plot(xr, yr, 'o', label='$|\hat{K}_i(u, v) - K(u, v)|$')
         print(f'HNdA_ij_dot_hnda_i  - H_f = {HNdA_ij_dot_hnda_i - H_f}')
         yr = HNdA_ij_dot_hnda_i - H_f
         len_1 = len(yr)
         # yr = HNdA_ij_Cij_dot_NdA_i - H_f
         ax1.plot(xr, yr, 'x', label='$|\hat{H}N_i(u, v)  - H(u, v)$|')
         yr = HN_i_2  - H_f
       #  plt.plot(xr, yr, 'x', label='$\hat{H}N_i_2(u, v)  - H(u, v)$')

         l_hnda_i_cij = []
         for hnda_i_cij in HNdA_i_Cij:
             print(f'h = {hnda_i_cij}')

             l_hnda_i_cij.append(np.sum(hnda_i_cij))

             #h = np.sum(np.sum(hnda_i_cij), axis=0)
             #l_hnda_i_cij.append(np.linalg.norm(h))

             #h = np.sum(np.sum(hnda_i_cij), axis=0)
             #l_hnda_i_cij.append(np.linalg.norm(h))

         #print(f'HNdA_i_Cij = {HNdA_i_Cij}')
         yr = l_hnda_i_cij - H_f
         #plt.plot(xr, yr, 'x', label='HNdA_i_Cij - $H$')
        # yr = HNdA_i_Cij
        # plt.plot(xr, yr, label='HNdA_i_Cij - H_f')
         ax1.set_ylabel('Difference')
         ax1.set_xlabel('Vertex No.')
         ax1.set_ylim([0,1.5])
         ax1.legend()


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
                elif 1:
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
                k_H_2 = (hndA_ij_dot_hnda_i/2.0) ** 2
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
        return HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i, HNdA_ij_dot_hnda_i, K_H_2, HNdA_i_Cij


    HC, bV, K_f, H_f, neck_verts, neck_sols = catenoid_N(r, theta_p, gamma, abc,
                                                         N=4,
                                                         refinement=5,
                                                         # refinement=4,
                                                         # refinement=6,
                                                         # refinement=1,
                                                         cdist=1e-5,
                                                         equilibrium=True)


    (HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i,  HNdA_ij_dot_hnda_i,
         K_H_2, HNdA_i_Cij) = int_curvatures(HC, bV, r, theta_p, printout=0)


    # Method, new test 2021-07-02
    if 1:
        c_outd2 = []
        HN_i_2 = []
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

    # plot results
    if 1:
         print('-')
         print(f'=' * len('Discrete (New):'))
         K_f = np.array(K_f)
         H_f = np.array(H_f)
         print(f'K_H_2 - K_f = {K_H_2 - K_f}')
         #plt.figure()
         yr = K_H_2 - K_f
         yr = yr[0:len_1]
         xr = list(range(len(yr)))
         xr = xr[0:len_1]
         ax2.plot(xr, yr, 'o', label='$|\hat{K}_i(u, v) - K(u, v)|$')
         print(f'HNdA_ij_dot_hnda_i  - H_f = {HNdA_ij_dot_hnda_i - H_f}')
         yr = HNdA_ij_dot_hnda_i - H_f
         yr = yr[0:len_1]
         # yr = HNdA_ij_Cij_dot_NdA_i - H_f
         ax2.plot(xr, yr, 'x', label='$|\hat{H}N_i(u, v)  - H(u, v)$|')
         


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
         #plt.ylabel('Difference')
         ax2.set_xlabel('Vertex No.')
         ax2.set_ylim([0,1.5])
         plt.legend()


if 0:
    HC.plot_complex(point_color=db, line_color=lb,
         complex_color_f=lb, complex_color_e=db)


#HNdA_ij_dot_hnda_i
# Cross section of meniscus neck

# neck_sols = [(0.0, -1.0), (0.0, -1.0), (0.0, -1.0), (0.0, -1.0)]
#TODO: H_f is always precisely zero at this point? Why?
if 0:
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

Nlist = [8, 16, 32, 64]
lp_error = [0.0, 0.0, 0.0, 0.0]  # area based
geo_error = [a/(2**2+1), a/(2**3+1), a/(2**4+1), a/(2**5+1)]  # area based
if 0:
    plt.plot(Nlist, lp_error, 'x')
    #plt.plot(Nlist, lp_error, 'x', label='Young-Laplace error: $(\Delta p - \Delta\hat{ p})/\Delta p $')
    plt.plot(Nlist, lp_error, 'x',
             label='Capillary force error: $(F_{cap} - \hat{F_{cap}}) / F_{cap} $')
    plt.plot(Nlist, geo_error, 'X', label='Integration error (p-normed trapezoidal rule $O(h^3)$)')

plt.legend()
plt.xlabel(r'N (number of boundary vertices)')
plt.ylabel(r'Error (%)')
plt.tick_params(axis='y', which='minor')

plt.show()

plot_polyscope(HC)
