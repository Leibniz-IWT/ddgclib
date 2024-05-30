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

#from ddgclib._catenoid import *
from ._truncated_cone import *
from ddgclib._plotting import *

from timeit import default_timer as timer
from ._curvatures import *

#Function for calculating the curvature at vertex v
def curvature(v):
    F, nn = vectorise_vnn(v)
    curvature_dict = b_curvatures_hn_ij_c_ij(F, nn)
    #curvature_dict = b_curvatures_hn_ij_c_ij_play(F, nn)
    #N_f0 = v.x_a - np.array([0.0, 0.0, v.x_a[2]]) # Version from Stefan
    #N_f0 = normalized(N_f0)[0]
    #curvature_dict = b_curvatures_hn_ij_c_ij(F, nn, n_i = N_f0)
    HNdA_i = curvature_dict['HNdA_i']
    return HNdA_i

#Function for saving the complex
def save_complex(HC,filename):
    '''
    To save the complex, please enter the following line
    #save_complex(HC, 'test_vom_20240209_refinement4.json')
    '''
    for v in HC.V:
       # print('---')
        v_new = []
        for i, vi in enumerate(v.x):
            #print(type(vi))
            v_new.append(float(vi))
            #v.x[i] = float(vi)
        HC.V.move(v, tuple(v_new))
        v.x_a = np.array(v.x_a, dtype='float')
        #v.x_a = v.x_a.astype(float)
        #for vi in v.x:
         #   print(type(vi))
    #print(type(v.x_a))
    HC.save_complex(fn = filename)


#Function for calculating the volume of the mesh

def volume(v):
    '''
    require a closed mesh
    '''
    F, nn = vectorise_vnn(v)
    curvature_dict = b_curvatures_hn_ij_c_ij_play(F, nn)
    #print(curvature_dict)
    V_ijk = curvature_dict['V_ijk']
    N_i = curvature_dict['N_i']
    #print(N_i)
    return V_ijk


#General funtion to calculate average-values
def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t
    avg = sum_num / len(num)
    return avg

# Function for closing the mesh (Necessary for calculating the volume
#TODO: Activate close_boundary condition
if 0:
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

        boundary_top.append(vat)
        print(f'vat = {vat}')
        boundary_bottom.append(vab)
        return HC, boundary_top, boundary_bottom

# Plotting of the normvecs (dir_vecs)
def plot_polyscope_plus_normvec(HC, normveccts):
    '''
    Function to execute: plot_polyscope_plus_normvec(HC, N_f0_vectors_pairs)
    '''
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

    # Add normal vectors
    vecs_vert = []
    for v in verts:
        N_f0 = v - np.array([0.0, 0.0, v[2]])
        N_f0 = normalized(N_f0)[0]

        #'''
        F, nn = vectorise_vnn(v)
        c_outd = b_curvatures_hn_ij_c_ij(F, nn, n_i = N_f0)
        vij = []

        for vn in v.nn:
            if vn in bV:
                continue
            else:
                vij.append(vn)

        if 0:
            for vn in v.nn:
                if vn in bV:
                    vij.append(vn)

        E_ij = vij[0].x_a - v.x_a
        E_ik = vij[1].x_a - v.x_a
        E_ij_ik = np.cross(E_ij, E_ik)
        E_ij_ik_norm = np.linalg.norm(E_ij_ik) # || E_ij ^ E_ik ||
        dir_vec = E_ij_ik / E_ij_ik_norm

        if np.dot(N_f0, dir_vec) < 0:
            dir_vec = - dir_vec
        #'''
        #N_f0_vectors_pairs.append([v.x_a, dir_vec])
        #vecs_vert.append(N_f0)
        vecs_vert.append(dir_vec)
    vecs_vert = np.array(vecs_vert) * 1e-4
    #surface.add_vector_quantity("N_f0 vectors", vecs_vert, radius=0.001,
    #                            length=0.005, color=(0.2, 0.5, 0.5))
    surface.add_vector_quantity("N_f0 vectors", vecs_vert, vectortype='ambient')


    ps.show()
