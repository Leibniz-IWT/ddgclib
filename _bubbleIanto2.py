#!/usr/bin/env python
# coding: utf-8

# Imports and physical parameters
import copy
import sys
import numpy as np
import scipy
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Local library
from ddgclib import *
from ddgclib._complex import *
from ddgclib._sphere import *
from ddgclib._curvatures import * #plot_surface#, curvature
from ddgclib._sessile import *
from ddgclib._capillary_rise_flow import * #plot_surface#, curvature
from ddgclib._eos import *
from ddgclib._plotting import plot_polyscope

def sector_volume(HC):
  #compute the volume of the complex by splitting it into sectors centred on origin
  (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
    HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
    Theta_i_cache) = HC_curvatures_sessile(HC, bV, r, theta_p, printout=0)
  total_bubble_volume=0.0
  for v in HC.V:
    dualArea = np.linalg.norm(C_ij_cache[v.x])
    H = HNdA_i_cache[v.x]
    #ToDo: Convex should be negative, concave parts should be positive.
    N_approx = - normalized(H)[0]
    total_bubble_volume += dualArea * sum( v.x_a[:] * N_approx[:] ) / 3.0
  return total_bubble_volume

def mean_flow(HC, bV, params, tau, print_out=False, fix_xy=False, pinned_line=False):
    (gamma, rho, g, r, theta_p, K_f, h) = params
    if print_out:
        print('.')
    # Compute interior curvatures
    (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
            HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
            Theta_i_cache) = HC_curvatures_sessile(HC, bV, r, theta_p, printout=0)

    total_bubble_volume = sector_volume(HC)
    airPressure = P_0 + P_0 * ( initial_volume - total_bubble_volume ) / total_bubble_volume
    print("airPressure",airPressure)

    # Move boundary vertices:
    bV_new = set()
    for v in HC.V:
        if print_out:
            print(f'v.x = {v.x}')
            if v.x[0] == 0.0 and v.x[1] == 0.0:
                print(f'='*10)
        # Compute boundary movements
        # Note: boundaries are fixed for now, this is legacy:
        if v in bV:
            rati = (np.pi - np.array(Theta_i) / 2 * np.pi)
            #ToDo: THis is not the actual sector ration (wrong angle)
            # rati = np.array(Theta_i) / (2 * np.pi)
            # print(f' rati = 2 * np.pi /np.array(Theta_i)= { rati}')
            print('-'*10)
            print('Boundary vertex:')
            print('-'*10)
            if 0:
                #ToDo: len(bV) is sector fraction
                H_K = HNda_v_cache[v.x] * np.array([0, 0, -1]) * len(bV)
                print(f'K_H in bV = {H_K }')
                K_H = ((np.sum(H_K) / 2.0) / C_ijk_v_cache[v.x] ) ** 2
                K_H = ((np.sum(H_K) / 2.0)  ) ** 2
                print(f'K_H in bV = {K_H}')
                print(f'K_H - K_f in bV = {K_H - K_f}')
            K_H_dA = K_H_i_cache[v.x] * np.sum(C_ij_cache[v.x])
            #ToDo: Adjust for other geometric approximations:
            l_a = 2 * np.pi * r / len(bV)  # arc length
            Xi = 1
            # Gauss-Bonnet: int_M K dA + int_dM kg ds = 2 pi Xi
            # Note: Area should be height of spherical cap
            # h = R - r * 4np.tan(theta_p)
            # Approximate radius of the great shpere K = (1/R)**2:
            #R_approx = 1 / np.sqrt(K_f)
            R_approx = 1 / np.sqrt(K_H_i_cache[v.x])
            theta_p_approx = np.arccos(np.min([r / R_approx, 1]))
            h = R_approx - r * np.tan(theta_p_approx)
            A_approx = 2 * np.pi * R_approx * h  # Area of spherical cap
            # A_approx  # Approximate area of the spherical cap
            kg_ds = 2 * np.pi * Xi - K_H_i_cache[v.x] * (A_approx)
            # ToDo: This is NOT the correct arc length (wrong angle)
            ds = 2 * np.pi * r  # Arc length of whole spherical cap
            k_g = kg_ds / ds  # / 2.0
            if print_out:
                print(f' R_approx * k_g = {R_approx * k_g}')
            phi_est = np.arctan(R_approx * k_g)
            # Compute boundary forces
            # N m-1
            if print_out:
                print(f' phi_est = { phi_est}')
                print(f' theta_p = {theta_p}')
            gamma_bt = gamma * (np.cos(phi_est) - np.cos(theta_p)) * np.array([0, 0, 1.0])
            F_bt = gamma_bt * l_a  # N

            F_bt = np.zeros_like(F_bt) # Fix boundaries for now
            if print_out:
                print(f' F_bt = {F_bt}')
                print(f' phi_est = {phi_est * 180 / np.pi}')
                print(f' F_bt = {F_bt}')
            new_vx = v.x + tau * F_bt
            #new_vx = v.x + tau * ( F_bt + dc )
            new_vx[2] = 0  # Boundary condition fix to z=0 plane
            new_vx = tuple(new_vx)
            if print_out:
                print('.')
                print(f'K_H_i_cache[v.x = {v.x}] = {K_H_i_cache[v.x]}')
                print(f'HNdA_i_cache[v.x = {v.x}] = {HNdA_i_cache[v.x]}')
                print(f' rati = {rati}')
                print(f'HNdA_i_cache[1] * rati[1]  = {HNdA_i_cache[v.x] * rati[1]}')
                print(f'K_H_i = {K_H_i}')
                print(f'K_f = {K_f}')
                print(f'K_H_i_cache[v.x] = {K_H_i_cache[v.x]}')
            # Move line if the contact angle is not pinned: 
            if not pinned_line:
                HC.V.move(v, new_vx)
                bV_new.add(HC.V[new_vx])
            if print_out:
                print(f'K_H_dA= {K_H_dA}')
                print(f'l_a= {l_a}')
                print(f'R_approx = {R_approx}')
                print(f'theta_p_approx = {theta_p_approx * 180 / np.pi}')
                print(f'Theta_i = {Theta_i}')
                print(f'phi_est  = {phi_est * 180 / np.pi}')
            print('-' * 10)
        # Current main code:
        else:
            print('-'*10)
            print('Interior vertex:')
            print('-'*10)
            H = HN_i_cache[v.x] #ToDo: Why is this sometimes negative? Should never be
            if print_out:
                print(f' H = {H}')
            print(f' v.x_a = {v.x_a}')
            df = gamma * H  # Hydrostatic pressure
            if print_out:
                print(f' HNdA_i_cache[v.x] = {HNdA_i_cache[v.x]}')
                print(f' gamma * H = { gamma * H}')
                print(f' HNdA_i_cache[v.x] = {HNdA_i_cache[v.x]}')
                print(f' HN_i = {HNdA_i_cache[v.x]}')
                print(f' rho * g * height = {rho * g * height}')
                print(f' height = {height}')
            H = HNdA_i_cache[v.x]
            #ToDo: Convex should be negative, concave parts should be positive.
            if numpy.linalg.norm(H) > 1e-10:
                N_approx = - normalized(H)[0]
            else:
                N_approx = - normalized(v.x_a)[0]
            waterPressure = P_0 - rho * g * v.x_a[2]
            dualArea = np.linalg.norm(C_ij_cache[v.x])
            #dg = rho * g * height * N_approx  # Assume bubble acting inward 11.08
            dg = (airPressure - waterPressure) * N_approx  * dualArea
            df = gamma * H
            print(f' df = {df}')
            print(f' dg = {dg}')
            # Scale forces to characteristic dimension:
            f_k = v.x_a + tau * ( df + dg )
            print(f'f_k = {f_k }')
            f_k[2] = np.max([f_k[2], 0.0])  # Floor constraint
            # Fix interior vertices constraints (avoid flow-over)
            if fix_xy: # Off: results in singular matrix error
                f_k[0] = v.x_a[0]  # constraint
                f_k[1] = v.x_a[1]  # constraint
            new_vx = tuple(f_k)
            # Move interior complex
            if print_out:
                print('.')
                print(f'HNdA_i_cache[{v.x}] = {HNdA_i_cache[v.x]}')
                print(f'HN_i_cache[{v.x}] = {HN_i_cache[v.x]}')
                print(f'H = {H}')
                print(f'v.x_a  = {v.x_a}')
                print(f'df = {df}')
                print(f'height = {height}')
                print(f'np.max([f_k[2], 0.0]) = {np.max([f_k[2], 0.0])}')
                print(f'f_k = {f_k}')
            HC.V.move(v, new_vx)
            print('-' * 10)
    if print_out:
        print(f'bV_new = {bV_new}')
    return HC, bV_new

def incr(HC, bV, params, tau=1e-5, plot=False, verbosity=1, fix_xy=False, pinned_line=False):
    HC.dim = 3  # Rest in case visualization has changed
    if verbosity == 2:
        print_out = True
    else:
        print_out = False
    # Update the progress
    HC, bV = mean_flow(HC, bV, params, tau=tau, print_out=print_out, fix_xy=fix_xy, pinned_line=pinned_line)
    return HC, bV

def ps_inc(surface, HC):
    HC.dim = 2  # The dimension has changed to 2 (boundary surface)
    HC.vertex_face_mesh()
    points = np.array(HC.vertices_fm)
    triangles = np.array(HC.simplices_fm_i)
    ### Register a point cloud
    # `my_points` is a Nx3 numpy array
    my_points = points
    ps_cloud = ps.register_point_cloud("my points", my_points)
    # ps_cloud.set_color((0.0, 0.0, 0.0))
    verts = my_points
    newPositions = verts
    surface.update_vertex_positions(newPositions)
    try:
        with timeout(0.1, exception=RuntimeError):
            # perform a potentially very slow operation
            ps.show()
    except RuntimeError:
        pass

def spherical_cap_init(RadFoot, theta_p, NFoot=4, refinement=0):
    RadSphere = RadFoot / np.sin(theta_p)
    Cone = []
    nn = []
    Cone.append(np.array([0.0, 0.0, 0.0])) #IC middle vertex
    nn.append([])
    ind = 0
    for phi in np.linspace(0.0, 2 * np.pi, NFoot):
        ind += 1
        Cone.append(np.array([np.sin(phi), np.cos(phi), theta_p])) #IC contact line circle
        # Define connections:
        nn.append([])
        if ind > 0:
            nn[0].append(ind)
            nn[ind].append(0)
            nn[ind].append(ind - 1)
            nn[ind].append((ind + 1) % NFoot)
    # clean Cone
    for f in Cone:
        for i, fx in enumerate(f):
            if abs(fx) < 1e-15:
                f[i] = 0.0
    Cone = np.array(Cone)
    nn[1][1] = ind
    # Construct complex from the cone geometry:
    HC = construct_HC(Cone, nn)   
    v0 = HC.V[tuple(Cone[0])]
    # Compute boundary vertices
    V = set()
    for v in HC.V:
        V.add(v)
    bV = V - set([v0])
    for i in range(refinement):
        V = set()
        for v in HC.V:
            V.add(v)
        HC.refine_all_star(exclude=bV)
    # Move to spherical cap
    for v in HC.V:
        theta = v.x_a[2]
        phi = np.arctan2(v.x_a[1],v.x_a[0])
        x = RadSphere * np.cos(phi) * np.sin(theta)
        y = RadSphere * np.sin(phi) * np.sin(theta)
        z = RadSphere * np.cos(theta) - RadSphere * np.cos(theta_p)
        if abs(z) < RadSphere*1e-6: z = 0.0
        HC.V.move(v, tuple((x,y,z)))
    # Rebuild set after moved vertices (appears to be needed)
    bV = set()
    for v in HC.V:
        if abs(v.x[2]) < RadSphere*1e-6: bV.add(v)
    return HC, bV

# ### Parameters
## Numerical
#To see the initial complex, set steps to 0.
refinement = 1
steps = 4 # Still stable, but not for higher values
tau = 1 #0.001  # 0.1 works

## Physical
P_0 = 101.325  # kPa, Ambient pressure
gamma = 72.8e-3  # # N/m
rho_0 = 998.2071  # kg/m3, density, STP
#rho_1 = 1.0 # kg/m3, density of air
g = 9.81  # m/s2
pinned_line = True  # pin the three phase contact if set to True
theta_p = 2*(63 / 75) * 20 + 30  # 46.8  40 #IC
theta_p = theta_p * np.pi / 180.0
r =  ((44 / 58) * 0.25 + 1.0) * 1e-3  # height mm --> m  Radius 10e-6
#h =  ((3 / 58) * 0.25 + 0.5) * 1e-3  # height mm --> m 10e-6
h = r / np.sin(theta_p) * ( 1 - np.cos(theta_p) )
Volume = np.pi * (3 * r ** 2 * h + h ** 3) / 6.0  # Volume in m3 (Segment of a sphere, see note above)
print(f'theta_p = {theta_p * 180.0 / np.pi}')
print(f'r = {r * 1e3} mm')
print(f'h = {h * 1e3} mm')
print(f'Volume = {Volume} m^3')
# define params tuple used in solver:
rho = rho_0
R = r / np.cos(theta_p)  # = R at theta = 0
# Exact values:
K_f = (1 / R) ** 2
params =  (gamma, rho, g, r, theta_p, K_f, h)
#le = V ^ (1 / 3)
print(f'r = {r}')
# Colour scheme for surfaces
db = np.array([129, 160, 189]) / 255  # Dark blue
lb = np.array([176, 206, 234]) / 255  # Light blue


HC,bV = spherical_cap_init(r, theta_p, NFoot=10, refinement=3)
initial_volume = sector_volume(HC)
#Surface energy minimisation
for i in range(steps):
  plot_polyscope(HC)
  print("i",i)
  #if i%10==0:plot_polyscope(HC, data_points)
  HC, bV = incr(HC, bV, params, tau=tau, plot=0, verbosity=0, pinned_line=pinned_line)
  cdist=r*1e-5
  HC.V.merge_all(cdist=cdist)
#plot_polyscope(HC)
#plt.show()
