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
r = 1e-3  # Radius of the tube (20 mm)
r = 2e-3  # Radius of the tube (20 mm)
#r = 1.0  # Radius of the droplet sphere
#r = 10.0  # Radius of the droplet sphere
r = 1.0  # Radius of the droplet sphere
#r = 0.1  # Radius of the droplet sphere

#r = 0.5e-5  # Radius of the droplet sphere
theta_p = 45 * np.pi/180.0  # Three phase contact angle
theta_p = 20 * np.pi/180.0  # Three phase contact angle
#theta_p = 0 * np.pi/180.0  # Three phase contact angle
#phi = 0.0
N = 8
N = 5
#N = 7
refinement = 0
#N = 20
#cdist = 1e-10
cdist = 1e-10

r = np.array(r, dtype=np.longdouble)
theta_p = np.array(theta_p, dtype=np.longdouble)


#r = decimal.Decimal(r)
#theta_p = decimal.Decimal(theta_p)

##################################################
# Cap rise plot with surrounding cylinder
if 0:
    fig, axes, HC = cape_rise_plot(r, theta_p, gamma, N=N, refinement=1)
##################################################
# PLot theta rise
if 0:
    c_outd_list, c_outd, vdict, X = out_plot_cap_rise(N=N, r=r, gamma=gamma, refinement=refinement)
    plot_c_outd(c_outd_list, c_outd, vdict, X)

# New fomulation
if 1:
    c_outd_list, c_outd, vdict, X = new_out_plot_cap_rise(N=N, r=r, gamma=gamma, refinement=refinement)
    plot_c_outd(c_outd_list, c_outd, vdict, X)

##################################################
##################################################
# Plot the current complex
if 1:
    F, nn, HC, bV, K_f, H_f = cap_rise_init_N(r, theta_p, gamma, N=N,
                                              refinement=refinement,
                                              cdist=cdist,
                                              equilibrium = True
                                              )
    #HC.V.print_out()
    HC.V.merge_all(cdist=cdist)
    #HC.V.print_out()

    fig, axes, fig_s, axes_s = HC.plot_complex(point_color=db, line_color=lb)

    if 0:
        R = r / np.cos(theta_p)  # = R at theta = 0
        K_f = (1 / R) ** 2
        H_f = 1 / R + 1 / R  # 2 / R
        rho = 1000  # kg/m3, density
        g = 9.81  # m/s2
        h_jurin = 2 * gamma * np.cos(theta_p) / (rho * g * r)
        axes.set_zlim(h_jurin-r, h_jurin + r)
        plot.show()

    axes.set_zlim(-r, 0.0)
    axes.set_zlim(-2*r, 2*r)

    R = r / np.cos(theta_p)
    print(f'R = {R}')
    K_f = (1 / R) ** 2
    H_f = 2 / R  # 2 / R

    # Compute the interior mean normal curvatures
    # (uses  curvatures_hn_i
    (HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i,  HNdA_ij_dot_hnda_i,
     K_H_2, HNdA_i_Cij) = int_curvatures(HC, bV, r, theta_p, printout=1)
    print('-')
    print(f'=' * len('Discrete (New):'))
    print(f'K_H_2 - K_f = {K_H_2 - K_f}')
    plt.figure()
    yr = K_H_2 - K_f
    xr = list(range(len(yr)))
    plt.plot(xr, yr, label='K_H_2 - K_f')
    print(f'HNdA_ij_dot_hnda_i  - H_f = {HNdA_ij_dot_hnda_i - H_f}')
    yr = HNdA_ij_dot_hnda_i - H_f
    plt.plot(xr, yr, label='HNdA_ij_dot_hnda_i - H_f')
    plt.ylabel('Difference')
    plt.xlabel('Vertex No.')
    plt.legend()

    # Compute the old discrete Gauss_Bonnet values
    print(f'=' * len('Discrete (Angle defect):'))
    print(f'Discrete (Angle defect):')
    print(f'='*len('Discrete (Angle defect)::'))
    chi, KdA, k_g = Gauss_Bonnet(HC, bV, print_out=True)

    print(f'=' * len('Analytical:'))
    print(f'Analytical:')
    print(f'='*len('Analytical:'))
    H_f, K_f, dA, k_g_f, dC = analytical_cap(r, theta_p)
    print(f'Area of spherical cap = {dA}')
    print(f'H_f  = {H_f }')    # int K_f dA
    print(f'K_f  = {K_f }')    # int K_f dA
    print(f'K_f dA = {K_f * dA}')    # int K_f dA
    # Values for r =1
    #k_g_f = np.cos(theta_p) / np.sin(theta_p)  # The geodisic curvature at a point
    #b_S_length = 2 * np.pi * np.sin(theta_p)  # The length of a boundary of a spherical cap

    print(f'k_g  = {k_g_f }')  # int
    print(f'k_g * dC = {k_g_f * dC  }')  # int
    print(f'K_f dA + int k_g_f dC = { K_f * dA + k_g_f  * dC }')
    print(f'LHS - RHS = { K_f * dA + k_g_f  * dC   - 2 * np.pi * np.array(chi)  }')

    print(f'k_g_f/K_f = {k_g_f/K_f}')
    print(f'k_g_f * dC/K_f dA = {(k_g_f * dC )/(K_f * dA)}')
    print(f'Total vertices: {HC.V.size()}')

    # NEW Compute new boundary formulation
    print(f'=' * len('New boundary formulation:'))
    print(f'New boundary formulation:')
    print(f'=' * len('New boundary formulation:'))

    # HNdA_ij_sum.append(-np.sum(np.dot(c_outd['HNdA_i'], c_outd['n_i'])))

    (HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i, HNdA_ij_dot_hnda_i,
      K_H_2, HNdA_i_Cij, Theta_i) = b_curvatures(HC, bV, r, theta_p, printout=False)
    
    print(f'HNdA_ij_dot_hnda_i = {HNdA_ij_dot_hnda_i}')
    print(f'K_H_2 = {K_H_2}')
    print(f'K_H_2 = {K_H_2}')
    #print(f'HNda_v_cache = {HNda_v_cache}')
    print(f'K_H_cache = {K_H_cache}')
    print(f'C_ijk_v_cache = {C_ijk_v_cache}')
    int_K_H_dA = 0 # Boundary integrated curvature
    for v in bV:
        K_H_dA = K_H_cache[v.x] * C_ijk_v_cache[v.x]
        int_K_H_dA += K_H_dA


    print(f'K_H_dA = {K_H_dA}')
    print(f'int_K_H_dA= {int_K_H_dA}')
    print('-')
    print('Analytical:')
   # k_g_a = 1/r * np.tan(theta_p)
    k_g_a = 1/R * np.tan(theta_p)
    print(f'k_g_a = {k_g_a}')
    l_a = 2 * np.pi * r / len(bV)     # arc length
    print(f'l_a= {l_a}')

    Xi = 1
    # Gauss-Bonnet
    # int_M K dA + int_dM kg ds = 2 pi Xi
    print('-')
    print('Numerical:')
    print(f'K_H_dA - 2 pi Xi  = {K_H_dA - 2 * np.pi * Xi}')
    #kg_ds = K_H_cache[v.x]  * (2 * np.pi * r**2) - 2 * np.pi * Xi
    #kg_ds = K_f  * (2 * np.pi * r**2) - 2 * np.pi * Xi
    #NOTE: Area should be height of spherical cap
    #h = R - r * np.tan(theta_p)
    # Approximate radius of the great shpere K = (1/R)**2:
    R_approx = 1/np.sqrt(K_f)
    theta_p_approx = np.arccos(r / R_approx)  #From R = r / np.cos(theta_p)
   # theta_p_approx = np.arctan(r / R_approx)  #From R = r / np.cos(theta_p)
   # A_approx = 2 * np.pi * r**2 * (1 - np.cos(theta_p_approx))
    #A_approx = 2 * np.pi * R_approx**2 * (1 - np.cos(theta_p_approx))
    h = R_approx - r * np.tan(theta_p_approx)
    #A_approx = 2 * np.pi * (r**2  + h**2)
    A_approx = 2 * np.pi * R_approx * h  # Area of spherical cap
    print(f'R_approx = {R_approx}')
    print(f'theta_p_approx = {theta_p_approx *180/np.pi}')
    print(f'A_approx = {A_approx}')
    #A_approx  # Approximate area of the spherical cap
    kg_ds = 2 * np.pi * Xi -  K_f  * (A_approx)
    #kg_ds = K_H_cache[v.x] - 2 * np.pi * Xi
    print(f'kg_ds =  2 pi Xi -  K_H_cache[v.x]  * np.pi * r**2  '
          f' = {kg_ds}')

    print(f'Theta_i = {Theta_i}')
    #ds =  Theta_i[0] * r  # Arc length
    ds = 2 * np.pi * r  # Arc length of whole spherical cap
    #ds = 2 * np.pi * r  # Arc length of whole spherical cap
    print(f'ds = {ds}')
    k_g = kg_ds / ds #/ 2.0
    print(f'k_g = {k_g}')
    #phi_est = cotan(r * k_g)
   # phi_est = np.arctan(r * k_g)
    phi_est = np.arctan(R * k_g)
    print(f'phi_est  = {phi_est * 180 / np.pi}')
    print(f'phi_est  = {phi_est * 180 / np.pi}')

    #for v in HC.V:
    #    if v in bV:
    #        continue
    #    print(f'v.ind = {v.index}')
    #    print(f'v.x = {v.x}')

    #
   # b_curvatures_hn_ij_c_ij

    if 0:
        for v, kh2, hnda in zip(int_V, K_H_2, HNdA_ij_dot_hnda_i):
            print(f'-')
            print(f'v.x = {v.x}')
            print(f'K_H_2 - K_f = {kh2 - K_f}')
            print(f'HNdA_ij_dot_hnda_i - H_f = {hnda- H_f}')

        print(v.x_a)
        print(type(v.x_a[0]))

    # New boundary curvature development:
    if 0:
        print(f'='*100)
        print(f'=')
        print(f'=')
        print(f'-')
        dp_exact = gamma * (2 / R)
        h = 2 * gamma * np.cos(theta_p) / (rho * g * r)
        print(f'dp_exact = {dp_exact}')
        print(f'rho * g * h = {rho * g * h}')
        print(f'h (Jurin) = { h} m')
        print(f'-')

        v0 = HC.V[(0.0, 0.0, R * np.sin(theta_p) - R)]
        print(f'HNda_v_cache[v0.x] = {HNda_v_cache[v0.x]}')
        hnda_ij = HNda_v_cache[v0.x]
        hnda_i = 0.5 * np.sum(hnda_ij, axis=0) / C_ijk_v_cache[v0.x]
        print(f'hnda_i = {hnda_i}')
        print(f'dp_v0 = {gamma * hnda_i}')
        print(f'dp_exact - dp_v0 = {dp_exact - gamma * hnda_i[2]}')
        print(f'-')
        for vn in v0.nn:
            print(f'v0.x = {v0.x}')
            print(f'vn.nn = {vn.x}')
            hnda_ij = HNda_v_cache[vn.x]

            hnda_i = 0.5 * np.sum(hnda_ij, axis=0) / C_ijk_v_cache[vn.x]

            N_f0 = np.array([0.0, 0.0, R * np.sin(theta_p)]) - vn.x_a  # First approximation
            N_f0 = normalized(N_f0)[0]
            print(f'hnda_i = {hnda_i}')
            print(f'N_f0i = {N_f0}')
            print(f'hnda_i = {np.sum(hnda_i)}')

            #np.sum(np.dot(hnda_i, N_f0), axis=0)
            hnda_dot_i = 0.5 * np.sum(np.dot(HNda_v_cache[vn.x], N_f0),
                                      axis=0) / C_ijk_v_cache[vn.x]
            print('- dot:')
            print(f'hnda_dot_i = {hnda_dot_i}')
            break  # Just test the first neighbour

        print(f'v0.x[2] =  {v0.x[2]}')
        print(f'vn.x[2] =  {vn.x[2]}')
        print(f'vn.x[2] - v0.x[2] = {vn.x[2] - v0.x[2]}')
        height = h + (vn.x[2] - v0.x[2])
        print(f'height =  {height} m')
        #print(f;)
        rhogh = rho * g * height
        print(f'rho * g * height = {rhogh} Pa')

        if 0: # This sucks:
            print('-')
            print('sum')
            dp_v1 = gamma * np.sum(hnda_i)
            print(f'dp_v1 = {dp_v1} Pa')
            print(f'rho * g * h - dp_v1 = {rhogh - dp_v1} Pa')

            print('-')
            print('sum')
            dp_v1 = gamma * np.dot(hnda_i, N_f0)
            print(f'dp_v1 = {dp_v1} Pa')
            print(f'rho * g * h - dp_v1 = {rhogh - dp_v1} Pa')
            # Equlibrium heighjt
            eq_height = dp_v1 / (rho * g)
            print(f'eq_height = dp_v1 / (rho * g) = {eq_height} m')
            print(f'eq_height = dp_v1 / (rho * g - height = {eq_height - height} m')

        print('-')
        print('dot')
        dp_v1 = gamma * hnda_dot_i #np.dot(hnda_dot_i, N_f0)
        print(f'dp_v1 = {dp_v1} Pa')
        print(f'rho * g * h - dp_v1 = {rhogh - dp_v1} Pa')
        # Equlibrium heighjt
        eq_height = dp_v1 / (rho * g)
        print(f'eq_height = dp_v1 / (rho * g) = {eq_height} m')
        print(f'eq_height = dp_v1 / (rho * g - height = {eq_height - height} m')

        print('Boundaries:')
        for v in bV:
            print(f'v.x = {v.x}')
            for vn in v.nn:
                if not (vn in bV):
                    print(f'K_H_cache[vn.x] = {K_H_cache[vn.x]}')
                    print(f'C_ijk_v_cache[vn.x] = {C_ijk_v_cache[vn.x]}')

            break
##################################################
# New Mean flow
##################################################
if 0:
    # Compute analytical ratio
    k_K = k_g_f / K_f
    R = r / np.cos(theta_p)  # = R at theta = 0
    K_f = (1 / R) ** 2
    H_f = 1 / R + 1 / R  # 2 / R
    h_jurin = 2 * gamma * np.cos(theta_p) / (rho * g * r)

    # Prepare film and move it to 0.0
    F, nn, HC, bV, K_f, H_f = cap_rise_init_N(r, theta_p, gamma, N=N,
                                              refinement=refinement,
                                              equilibrium=True
                                              )
    h = 0.0
    params = (gamma, rho, g, r, theta_p, K_f, h)

    for i in range(0):
        #HC = kmean_flow(HC, bV, params, tau=0.0001)
        #HC, bV = kmean_flow(HC, bV, params, tau=0.0001)
        tau = 1e-5
        tau = 1e-8
        #tau = 0.1
        # HC.V.merge_all(cdist=cdist)
        HC, bV = kmean_flow(HC, bV, params, tau=tau, print_out=0)

    #HC.V.print_out()
    h_final = 0.0
    for v in HC.V:
        #h_final = np.max([h_final, v.x_a[2]])
        h_final = np.min([h_final, v.x_a[2]])
   # print(f'h_final = {h} m')
    print(f'h_final = {h_final } m')
    print(f'h (Jurin) = {h_jurin} m')
    print(f'h_final - h (Jurin) = {h_final - h_jurin} m')
    try:
        print(f'Error: h_final - h (Jurin) = {100*abs(h_final - h/h)} %')
    except ZeroDivisionError:
        pass
    fig, axes, fig_s, axes_s = HC.plot_complex(point_color=db, line_color=lb)

    #plt.axis('off')
    #axes.set_zlim(-2 * r, h + 2 * r)
    axes.set_zlim(-2 * r, h_jurin + 2 * r)
    #axes.set_zlim(h -2 * r, h + 2 * r)
##################################################
##################################################
# Gauss Bonnet
if 0:
    c_outd_list, c_outd, vdict, X = out_plot_cap_rise_boundary(N=N, r=r, refinement=2)
    plot_c_outd(c_outd_list, c_outd, vdict, X, ylabel=r'$\pi \cdot  m$ or $\pi \cdot m^{-1}$')

##################################################
# Mean flow:
##################################################
if 0:
    R = r / np.cos(theta_p)  # = R at theta = 0
    # Exact values:
    K_f = (1 / R) ** 2
    H_f = 1 / R + 1 / R  # 2 / R
    h_jurin = 2 * gamma * np.cos(theta_p) / (rho * g * r)
    print(f'Expected rise: {h_jurin}')
    print(f'Hydrostatic pressure at equil = r * g * h_jurin: {r * g * h_jurin} Pa')

    # Initial film height:
    h_0 = h_jurin + (R - R * np.sin(theta_p))
    #h_0 = 0.0
    print(f'h_0  = {h_0 }')

    if 0:
        fig, axes, fig_s, axes_s = HC.plot_complex(point_color=db,
                                                   line_color=db,
                                                   complex_color_f=lb,
                                                   complex_color_e=db
                                                   )

    HC, bV = film_init(r, h_jurin)

    HC.V.merge_all(cdist=cdist)

    if 1:  # TODO: Hack for scale; remove after correct equilibrium is found:
        fig, axes, fig_s, axes_s = HC.plot_complex(point_color=db,
                                                   line_color=db,
                                                   complex_color_f=lb,
                                                   complex_color_e=db
                                                   )

    h_0 = 2 * h_jurin

    # Mean flow
    for i in range(1):
        params = (gamma, rho, g)
        HC = mean_flow(HC, bV, h_0=h_0, params=params, tau=1e-5)
        HC.V.merge_all(cdist=cdist)

    # Find the equilibrium capillary rise:
    h_f = 2 * h_jurin
    for v in HC.V:
        if v in bV:
            continue
        else:
            h_f = min(v.x[2], h_f)

    fig, axes, fig_s, axes_s = HC.plot_complex(point_color=db,
                                               line_color=db,
                                               complex_color_f=lb,
                                               complex_color_e=db
                                               )
    # axes.set_xlim3d(-(0.1*r + r) , 0.1*r + r)
    # axes.set_ylim3d(-(0.1*r + r) , 0.1*r + r)
    # axes.set_zlim3d(-(0.1*r + r) , 0.1*r + 2*r)

    print(f'The final capillary height rise is {h_f} m')
    print(f'Expected rise: {h_jurin}')


##################################################
# New plots:
##################################################
if 0:
    ps = pplot_surface(HC)
    # View the point cloud and mesh we just registered in the 3D UI
    ps.show()

#HC.V.print_out()
# Plot all


#plt.axis('off')

plt.show()







