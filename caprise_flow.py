import copy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
#import plotly.graph_objects as go
#from mayavi import mlab
#import gudhi
import polyscope as ps

from ddgclib._complex import Complex


# ddg imports
from ddgclib import *
from ddgclib._complex import *
from ddgclib._curvatures import * #plot_surface#, curvature
from ddgclib._capillary_rise_flow import * #plot_surface#, curvature
from ddgclib._eos import *
from ddgclib._misc import *
from ddgclib._plotting import *


from interruptingcow import timeout


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

h = 0.0 # Initial film height (TODO: Set higher)

#r = 0.5e-5  # Radius of the droplet sphere
theta_p = 45 * np.pi/180.0  # Three phase contact angle
theta_p = 20 * np.pi/180.0  # Three phase contact angle
#phi = 0.0
N = 8
N = 5
N = 6
N = 7
N = 5
#N = 12
refinement = 0
#N = 20
#cdist = 1e-10
cdist = 1e-10

r = np.array(r, dtype=np.longdouble)
theta_p = np.array(theta_p, dtype=np.longdouble)

def mean_flow(HC, bV, params, tau, print_out=False):
    (gamma, rho, g, r, theta_p, K_f, h) = params
    print('.')
    # Define the normals (NOTE: ONLY WORKS FOR CONVEX SURFACES)
    # Note that the cache HC.V is an OrderedDict:
    N_i = []

    f = []
    for i, v in enumerate(HC.V):
        if v in bV:
            continue
        f.append(v.x_a)

    # Compute interior curvatures
    #(HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i, HNdA_ij_dot_hnda_i,
    # K_H_2, HNdA_i_Cij) = int_curvatures(HC, bV, r, theta_p, printout=False)

    (HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i, HNdA_ij_dot_hnda_i,
      K_H_2, HNdA_i_Cij, Theta_i) = HC_curvatures(HC, bV, r, theta_p, printout=0)
    print('.')
    print(f'HNda_v_cache = {HNda_v_cache}')
    print(f'HNdA_ij_dot_hnda_i = {HNdA_ij_dot_hnda_i}')
    print(f'C_ijk_v_cache = {C_ijk_v_cache}')

    print(f'H_f = {H_f}')
    print(f'K_H_2 = {K_H_2}')
    print(f'K_f = {K_f}')
    print(f'K_H_2 - K_f = {K_H_2 - K_f}')
    print(f'K_H_2/K_f = {K_H_2/K_f}')

    if 0:
        H = np.nan_to_num(HNdA_ij_dot_hnda_i)  # TODO: Future use HN_i
        H = np.nan_to_num(HN_i)  # TODO: Future use HN_i


        # Add gravity force rho * g * h
        height = []
        for v in HC.V:
            if (v in bV):
                continue
            height.append(np.max([v.x_a[2], 0.0]))

        height = np.array(height)
        df = df - (rho * g * height)  # * np.array([0.0, 0.0, -1.0])

        #    print(f'height = {height}')
        f = np.array(f)
        if 0:
            dF = np.zeros_like(f)
            dF[:, 2] = df
            # f_k = f + tau * dF # Previously: f_k = f + tau * df
           # dF_max = np.max(dF)
            #t = tau
            #t = 0.5 * r / (dF_max + r)  # tau
            #tdF = t * dF
            #for i, tdf in enumerate(tdF):
            #    if np.abs(tdf[2]) > r:  # Alleviate floating point errors
                    # tdF[i, 2] = np.sign(tdf[2]) * 0.05 * r
            #        tdF[i, 2] = np.sign(tdf[2]) * 0.05 * h
           # f_k = f + tdF  # Previously: f_k = f + tau * df

        f_k = f + tau * df

    # Interior vertex forces
    VA = []
    # Move boundary vertices:
    bV_new = set()
    for v in HC.V:
        # Compute boundary movements
        if v in bV:
            print(f'K_H_cache[v.x = {v.x}] = {K_H_cache[v.x]}')
            print(f'HNda_v_cache[v.x = {v.x}] = {HNda_v_cache[v.x]}')
            rati = (np.pi - np.array(Theta_i) / 2 * np.pi)
            #TODO: THis is not the actual sector ration (wrong angle)
            # rati = np.array(Theta_i) / (2 * np.pi)
            # print(f' rati = 2 * np.pi /np.array(Theta_i)= { rati}')
            print(f' rati = {rati}')
            # rati =  (2 * np.pi  - np.array(Theta_i))/np.pi
            # print(f' rati = (2 * np.pi  - Theta_i)/np.pi = { rati}')
            print(f'HNdA_i[1] * rati[1]  = {HNda_v_cache[v.x] * rati[1]}')

            #TODO: len(bV) is sector fraction
            H_K = HNda_v_cache[v.x] * np.array([0, 0, -1]) * len(bV)
            print(f'K_H in bV = {H_K }')
            K_H = ((np.sum(H_K) / 2.0) / C_ijk_v_cache[v.x] ) ** 2
            K_H = ((np.sum(H_K) / 2.0)  ) ** 2
            print(f'K_H in bV = {K_H}')
            print(f'K_H - K_f in bV = {K_H - K_f}')
            K_H_dA = K_H_cache[v.x] * C_ijk_v_cache[v.x]
            print(f'K_H_dA= {K_H_dA}')
            l_a = 2 * np.pi * r / len(bV)  # arc length
            print(f'l_a= {l_a}')
            Xi = 1
            # Gauss-Bonnet: int_M K dA + int_dM kg ds = 2 pi Xi
            # NOTE: Area should be height of spherical cap
            # h = R - r * 4np.tan(theta_p)
            # Approximate radius of the great shpere K = (1/R)**2:
            R_approx = 1 / np.sqrt(K_f)
            theta_p_approx = np.arccos(
                r / R_approx)
            h = R_approx - r * np.tan(theta_p_approx)
            A_approx = 2 * np.pi * R_approx * h  # Area of spherical cap
            print(f'R_approx = {R_approx}')
            print(f'theta_p_approx = {theta_p_approx * 180 / np.pi}')
            print(f'A_approx = {A_approx}')
            # A_approx  # Approximate area of the spherical cap
            kg_ds = 2 * np.pi * Xi - K_f * (A_approx)

            print(f'Theta_i = {Theta_i}')
            # TODO: This is NOT the correct arc length (wrong angle)
            ds = 2 * np.pi * r  # Arc length of whole spherical cap
            print(f'ds = {ds}')
            k_g = kg_ds / ds  # / 2.0
            print(f'k_g = {k_g}')
            phi_est = np.arctan(R_approx * k_g)
            print(f'phi_est  = {phi_est * 180 / np.pi}')

            #ADD FORCE HERE:


            HC.V.move(HC.V[tuple(v_a)], tuple(dK[i]))
            bV_new.add(HC.V[tuple(dK[i])])
            if print_out:
                print(f'dK[i] = {dK[i]}')
        else:
            print(f'HNda_v_cache[{v.x}] = {HNda_v_cache[v.x]}')
            H = np.dot(HNda_v_cache[v.x], np.array([0, 0, 1]))
            height = np.max([v.x_a[2], 0.0])
            df = gamma * H  # Hydrostatic pressure
            df = df - (rho * g * height)
            f_k = f + tau * df
            print(f'f_k = {f_k}')
            VA.append(v.x_a)
            # Move interior complex
            # HC.V.move(HC.V[tuple(v_a)], tuple(f_k[i]))

    if print_out:
        print(f'bV_new = {bV_new}')

    return HC, bV_new


    return HC, bV

def incr(HC, bV, params, tau=1e-5):
    HC.dim = 3  # Rest in case visualization has changed

    # Update the progress
    HC, bV = mean_flow(HC, bV, params, tau=tau, print_out=0)

    # Compute progress of Capillary rise:
    current_Jurin_err(HC)

    # Plot in Polyscope:
    ps_inc(surface, HC)

def current_Jurin_err(HC):
    h_final = 0.0
    for v in HC.V:
        # h_final = np.max([h_final, v.x_a[2]])
        h_final = np.min([h_final, v.x_a[2]])
    # print(f'h_final = {h} m')
    print(f'h_final = {h_final} m')
    print(f'h (Jurin) = {h_jurin} m')
    print(f'h_final - h (Jurin) = {h_final - h_jurin} m')
    try:
        print(f'Error: h_final - h (Jurin) = {100 * abs(h_final - h / h)} %')
    except ZeroDivisionError:
        pass
    return

def ps_inc(surface, HC):
    #F, nn, HC, bV, K_f, H_f = cap_rise_init_N(r, theta_p, gamma, N=N,
    #                                          refinement=refinement,
    #                                          equilibrium=True
    #                                          )

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
        with timeout(0.3, exception=RuntimeError):
            # perform a potentially very slow operation
            ps.show()
    except RuntimeError:
        pass

if 1:
    # Define HC
    # Compute analytical ratio
    H_f, K_f, dA, k_g_f, dC = analytical_cap(r, theta_p)
    print(f'H_f, K_f, dA, k_g_f, dC = {H_f, K_f, dA, k_g_f, dC}')
    k_K = k_g_f / K_f  #TODO: We no longer need this?
    h_jurin = 2 * gamma * np.cos(theta_p) / (rho * g * r)

    # Prepare film and move it to 0.0
    F, nn, HC, bV, K_f, H_f = cap_rise_init_N(r, theta_p, gamma, N=N,
                                              refinement=refinement,
                                              equilibrium=1
                                              )
    h = 0.0
    params = (gamma, rho, g, r, theta_p, K_f, h)
    # HC.V.print_out()
    HC.V.merge_all(cdist=cdist)


    # HC, bV = kmean_flow(HC, bV, params, tau=tau, print_out=0)




# Matplotlib and plotly
if 0:
    print('Finished, plotting...')
    if 0:
        HC.plot_complex()
        plt.show()


    # First possibility: plotly
    if 1:
        import plotly.graph_objects as go

        fig = go.Figure(data=[
            go.Mesh3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
            )
        ])
        fig.show()
    # Second possibility: matplotlib
    if 1:

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                        triangles=triangles)
        plt.show()

# Polyscope
if 1:

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
    try:
        with timeout(0.3, exception=RuntimeError):
            # perform a potentially very slow operation
            ps.show()
    except RuntimeError:
        pass

#
if 1:
    incr(HC, bV, params)
