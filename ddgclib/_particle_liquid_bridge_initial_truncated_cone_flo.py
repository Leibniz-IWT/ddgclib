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
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from ddgclib._catenoid import *
from ddgclib._truncated_cone import *
from ddgclib._plotting import *
from timeit import default_timer as timer


def curvature(v):
    F, nn = vectorise_vnn(v)
    #curvature_dict = b_curvatures_hn_ij_c_ij(F, nn)
    curvature_dict = b_curvatures_hn_ij_c_ij_play(F, nn)
    HNdA_i = curvature_dict['HNdA_i']
    return HNdA_i

'''
in this part of the code, the boundarys will be closed with the function close_boundary.

'''
def close_boundary(HC, boundary_top, boundary_bottom):
    v_avg_top = np.zeros(3)
    for v in boundary_top:
        v_avg_top += v.x_a

    v_avg_top = v_avg_top/len(boundary_top)
    v_avg_bottom = np.zeros(3)
    for v in boundary_bottom:
        v_avg_bottom += v.x_a

    v_avg_bottom = v_avg_bottom/len(boundary_bottom)

    vat = HC.V[tuple(v_avg_top)]
    for v in boundary_top:
        v.connect(vat)

    vab = HC.V[tuple(v_avg_bottom)]
    for v in boundary_bottom:
        v.connect(vab)

#    boundary_top.add(vat) # from stefan
#    boundary_bottom.add(vab) # from stefan

    boundary_top.append(vat)
    boundary_bottom.append(vab)
    return HC, boundary_top, boundary_bottom

#def fun_liquid_bridge_truncated_cone(d_l, d_u, length, refinement, tau, t_f, gamma):


def volume(v):
    F, nn = vectorise_vnn(v)
    curvature_dict = b_curvatures_hn_ij_c_ij_play(F, nn)
    #print(curvature_dict)
    V_ijk = curvature_dict['V_ijk']
    N_i = curvature_dict['N_i']
    #print(N_i)
    return V_ijk


def fun_liquid_bridge_truncated_cone_N(d_l, d_u, length, refinement, tau, t_f, gamma):
    '''
    length = length of the inital cone
    tau = size of time step
    t_f = breaking condition in the while-loop = last time step
    d = diameter of the initial cyliner
    '''
    starttime = timer()

    r_l = d_l / 2
    r_u = d_u / 2

    t = 0  #  initial time (=constant)

    v_l = 0
    v_u = length +v_l
    '''
    -----------------------------------------------------------
    From here, the mesh for an open truncated cone is meshed
    '''

    HC, bV, K_f, H_f, neck_verts, neck_sols = truncated_cone_initial_N(r_l, r_u, length, refinement)

    '''
    -----------------------------------------------------------
    closing the open mesh
    '''

    # Define bottom boundary condition
    boundary_bottom = []
    for v in HC.V:
        if v.x[2] == v_l:
            boundary_bottom.append(v)

    # Define top boundary condition
    boundary_top = []
    for v in HC.V:
        if v.x[2] == v_u:
            boundary_top.append(v)

    HC, boundary_top, boundary_bottom = close_boundary(HC, boundary_top, boundary_bottom)
    '''
    ----------------------------------------------------
    Dummy parameter for the residual
    '''
    # runvariable for the iteration tracking
    residual_iteration = 0
    iteration_list = []

    # list to track the progress from df
    df_list_max = []
    df_list_min = []

    # list to track the progress of HNdA
    dHNdA_list_min = []
    dHNdA_list_max = []

    res_list = []

    volume_list = []



    '''
    -----------------------------------------------------------
    From here, the optimization is started
    '''

    i = 0


    while t < t_f:
        # Loop through all interior vertices
        # calculating the volume of the entire bridge

        V_current = 0

        for v in HC.V:
            V_ijk = volume(v)
            #print(V_ijk)
            V_current += np.sum(V_ijk)

        V_current = V_current/12
        volume_list.append(V_current)



        cHCV = copy.copy(HC.V)

        # dummy list to find the maximum df
        df_list_dummy       = []
        dHNdA_list_dummy    = []
        for v in cHCV:
            # exclude boundaries
            if v in set(boundary_bottom):
                continue
            # Define top boundary condition
            if v in set(boundary_top):
                continue
            # Compute curvature for each interior vertex
            HNdA_i = curvature(v)
            # HNdA_i, V_i, N_i = ????(v)

            # split loop, compute V_total
            # start new    for v in cHCV:
            #
            #print(HNdA_i)
            # Step in time
            df = -gamma * HNdA_i  # Add compressive force F_c = N_i * (V_initial - V_total)
            v_new = v.x_a - tau/2 * df
            HC.V.move(v, tuple(v_new))

            '''
            Try to implement a Residual with the mean curvature
            '''
            #print(HNdA_i)
            #print(np.linalg.norm(df))
            df_list_dummy.append(np.linalg.norm(df))
            dHNdA_list_dummy.append(np.linalg.norm(HNdA_i))
            #print(df)
        #print(HNdA_i)
        #print(max(df_list_dummy))
        # if V_ne
        # save the maximum and minimum datas for the Residuals
        df_list_max.append(max(df_list_dummy))
        df_list_min.append(min(df_list_dummy))

        # save the numbers of iteration for lately plotting
        residual_iteration = residual_iteration +1
        iteration_list.append(residual_iteration)

        dHNdA_list_max.append(max(dHNdA_list_dummy))
        dHNdA_list_min.append(min(dHNdA_list_dummy))

        '''
        if i == 0:
            continue
        else:
            res = abs((df_list_max[i-1]-df_list_max[i])/df_list_max[i])

        '''



        res = abs((df_list_max[i-1]-df_list_max[i])/df_list_max[i])
        res_list.append(res)


        t = tau + t
        i = i+1

        if i % 100 == 0:
            print(i)

    endtime = timer()

    return(HC, iteration_list, df_list_min, df_list_max,res_list, starttime, endtime, volume_list)





'''
Function for calculating the volume of the mesh
'''



def save_complex(HC,filename):
    for v in HC.V:
        print('---')
        v_new = []
        for i, vi in enumerate(v.x):
            print(type(vi))
            v_new.append(float(vi))
            #v.x[i] = float(vi)
        HC.V.move(v, tuple(v_new))
        v.x_a = np.array(v.x_a, dtype='float')
        #v.x_a = v.x_a.astype(float)
        for vi in v.x:
            print(type(vi))
    print(type(v.x_a))
    HC.save_complex(fn = filename)


def fun_liquid_bridge_load_N(length, tau, t_f, gamma, filename):
    '''
    length = length of the inital cone
    tau = size of time step
    t_f = breaking condition in the while-loop = last time step
    d = diameter of the initial cyliner
    '''
    starttime = timer()

    v_l = 0

    v_u = length +v_l



    HC = Complex(3)
    HC.load_complex(filename)


    # Define bottom boundary condition
    boundary_bottom = []
    for v in HC.V:
        if v.x[2] == v_l:
            boundary_bottom.append(v)

    # Define top boundary condition
    boundary_top = []
    for v in HC.V:
        if v.x[2] == v_u:
            boundary_top.append(v)

    # HC, boundary_top, boundary_bottom = close_boundary(HC, boundary_top, boundary_bottom)


    '''
    ----------------------------------------------------
    Dummy parameter for the residual
    '''

    t = 0  #  initial time (=constant)




    # runvariable for the iteration tracking
    residual_iteration = 0
    iteration_list = []

    # list to track the progress from df
    df_list_max = []
    df_list_min = []

    # list to track the progress of HNdA
    dHNdA_list_min = []
    dHNdA_list_max = []

    res_list = []

    volume_list = []



    '''
    -----------------------------------------------------------
    From here, the optimization is started
    '''

    i = 0


    while t < t_f:
        # Loop through all interior vertices
        # calculating the volume of the entire bridge

        V_current = 0

        for v in HC.V:
            V_ijk = volume(v)
            #print(V_ijk)
            V_current += np.sum(V_ijk)

        V_current = V_current/12
        volume_list.append(V_current)



        cHCV = copy.copy(HC.V)

        # dummy list to find the maximum df
        df_list_dummy       = []
        dHNdA_list_dummy    = []
        for v in cHCV:
            # exclude boundaries
            if v in set(boundary_bottom):
                continue
            # Define top boundary condition
            if v in set(boundary_top):
                continue
            # Compute curvature for each interior vertex
            HNdA_i = curvature(v)
            # HNdA_i, V_i, N_i = ????(v)

            # split loop, compute V_total
            # start new    for v in cHCV:
            #
            #print(HNdA_i)
            # Step in time
            df = -gamma * HNdA_i  # Add compressive force F_c = N_i * (V_initial - V_total)
            v_new = v.x_a - tau/2 * df
            HC.V.move(v, tuple(v_new))

            '''
            Try to implement a Residual with the mean curvature
            '''
            #print(HNdA_i)
            #print(np.linalg.norm(df))
            df_list_dummy.append(np.linalg.norm(df))
            dHNdA_list_dummy.append(np.linalg.norm(HNdA_i))


        # save the maximum and minimum datas for the Residuals
        df_list_max.append(max(df_list_dummy))
        df_list_min.append(min(df_list_dummy))

        # save the numbers of iteration for lately plotting
        residual_iteration = residual_iteration +1
        iteration_list.append(residual_iteration)

        dHNdA_list_max.append(max(dHNdA_list_dummy))
        dHNdA_list_min.append(min(dHNdA_list_dummy))


        res = abs((df_list_max[i-1]-df_list_max[i])/df_list_max[i])
        res_list.append(res)


        t = tau + t
        i = i+1

        if i % 100 == 0:
            print(i)

    endtime = timer()

    return(HC, iteration_list, df_list_min, df_list_max,res_list, starttime, endtime, volume_list)

