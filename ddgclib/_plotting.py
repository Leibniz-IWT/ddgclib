import numpy as np
import scipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from ipywidgets import *
from matplotlib.widgets import Slider
from matplotlib.widgets import Slider

import polyscope as ps
from ddgclib._misc import *  # coldict is neeeded

import numpy as np

# Polyscope
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
    ps.show()


def pplot_surface(HC):

    HC.vertex_face_mesh()
    #print(f'verts = {HC.vertices_fm}')
    #print(f'faces = {HC.simplices_fm}')
    #print(f'faces i = {HC.simplices_fm_i}')
    # Initialize polyscope
   # print(f'verts = {np.array(HC.vertices_fm)}')
   # print(f'faces = {np.array(HC.simplices_fm)}')
   # print(f'faces i = {np.array(HC.simplices_fm_i)}')
    ps.init()

    ### Register a point cloud
    # `my_points` is a Nx3 numpy array
    if 0:
        my_points = np.array([[0, 0, 0],
                              [1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]
                              )
        ps.register_point_cloud("my points", my_points)

    #if 0:


    verts = np.array(HC.vertices_fm)
    faces = np.array(np.array(HC.simplices_fm_i))
    if 0:
        faces = np.array([[0, 1, 2],
                          [0, 2, 3],
                          [1, 2, 3],
                          #[0, 1, 3],
                          ]
                              )
    ### Register a mesh
    # `verts` is a Nx3 numpy array of vertex positions
    # `faces` is a Fx3 array of indices, or a nested list
    ps.register_surface_mesh("my mesh", verts, faces, smooth_shade=True)
    ps.set_up_dir('z_up')
    if 0:
        # Replot error data
        Nmax = 21
        lp_error = np.zeros(Nmax)
        N = list((range(Nmax)))

        plt.figure()
        plt.plot(N, lp_error)
        plt.xlabel(r'N (number of boundary vertices)')
        plt.ylabel(r'%')
        plt.show()

    ps.screenshot("mesh.png")
    return ps