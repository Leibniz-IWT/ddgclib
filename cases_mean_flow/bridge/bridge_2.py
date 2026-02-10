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
refinements =  [2, 3, 4, 5]   # NOTE: 2 is the minimum refinement needed for the complex to be manifold
#refinements =  [2, 3]   # NOTE: 2 is the minimum refinement needed for the complex to be manifold

# data containers:
errors_total = []
lp_error = []
lp_error_2 = []
geo_error = []
Nlist = []

for refinement in refinements:
    # Construct the new Catenoid
    a, b, c = 1, 0.0, 1  # Geometric parameters of the catenoid
    abc = (a, b, c)
    HC, bV, K_f, H_f, neck_verts, neck_sols = acatenoid_N(r, theta_p, gamma, abc, refinement=refinement,
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


    #
    # Total integral estimates:
    if 1:
        sum_HNdA_i = 0.0
        for hndA_i, c_ij in zip(HNdA_i_list, C_ij_i_list):
           # print(f' hndA_i = { hndA_i}')
            hndA_i_c_ij = hndA_i #/ np.sum(c_ij)
            #sum_HNdA_i += np.dot(hndA_i_c_ij, [0, 0, 1])
            sum_HNdA_i += hndA_i_c_ij
        # sum_HNdA_i += np.dot(hndA_i_c_ij, [0, 0, -1])

        print('-')
        print(f'HC.V.size() = {HC.V.size()}')
        print(f'len(HNdA_i_list) = {len(HNdA_i_list)}')
        print(f'len(bV) = {len(bV)}')
        print(f'sum_HNdA_i (total) = {sum_HNdA_i}')
        errors_total.append(sum_HNdA_i)

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
if 1:
    if 1:
        errors_point_wise_1 = np.abs(errors_point_wise_1)
        errors_point_wise_2 = np.abs(errors_point_wise_2)

        Nlist = verts
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # ax2 = ax.twinx()
        ln1 = ax.loglog(Nlist, errors_point_wise_1 * 100, 'x', color='tab:blue',
                        label=r'Point-wise errors in $\mathbf{x_{+}}=(0, 0, 1)$ direction  ($\frac{\widehat{H\mathbf{N}} d A_{i} \cdot \mathbf{x_{+}}}{C_i}$)')
        ln2 = ax.loglog(Nlist, errors_point_wise_2 * 100, 'X', color='tab:red',
                        label=r'Point-wise errors in $\mathbf{x_{-}}=(0, 0, -1)$ direction  ($\frac{\widehat{H\mathbf{N}} d A_{i} \cdot \mathbf{x_{-}}}{C_i}$)')

        # ax2.set_ylabel(r'Young-Laplace error (%)')
        ax.set_xlabel(r'$n$ (number of vertices)')

        ln3 = ax.loglog(Nlist, np.abs(np.sum(errors_total, axis=1)) * 100, 'o', color='tab:orange',
                        label=r'Integrated errors (sum of vector components in $\widehat{H\mathbf{N}} d A_{i}$ )')
        ax.set_ylabel(f'Integration error (%)')

        lns = ln1 + ln2 + ln3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)

        plt.tick_params(axis='y', which='minor')
        ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
        # ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())

        # ax2.set_ylim([1e-15, 1e-13])
        ax.set_ylim([1e-20, 1e4])
        #    ax2.set_ylim([1e-20, 1e4])

        plt.tick_params(axis='y', which='minor')

        plt.savefig('./fig/errors.png', bbox_inches='tight', dpi=600)

if 0:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()
    ln1 = ax2.loglog(Nlist, np.abs(lp_error), 'x', color='tab:blue',
                 label='Capillary force error: $(F_{cap} - \hat{F_{cap}}) / F_{cap} $')

    ax2.set_ylabel(r'Capillary force error (%)')
    ax.set_xlabel(r'$n$ (number of boundary vertices)')

    ln2 = ax.loglog(Nlist, geo_error, 'X', color='tab:orange',
              label='Integration error (p-normed trapezoidal rule $O(h^2)$)')

    ax.set_ylabel(r'Integration error (%)')

    #ax2.set_ylim([1e-15, 1e-18])
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

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    plt.tick_params(axis='y', which='minor')
    from matplotlib import pyplot as plt, ticker as mticker

    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())

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

    if 0:
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
    if 0:
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


if 1:
    plot_polyscope(HC)



# visualize!
#ps_cloud = ps.register_point_cloud("my points", points)
#ps.show()
