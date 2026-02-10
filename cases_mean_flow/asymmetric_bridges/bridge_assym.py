# Imports and physical parameters
import numpy as np
import scipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from ipywidgets import *
from matplotlib.widgets import Slider
from matplotlib import pyplot as plt, ticker as mticker


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

# Define local polyscope including particles
def plot_polyscope(HC, vector_field=None, scalar_field=None, fn='', up="x_up",):
    # Initialize polyscope
    ps.init()
    ps.set_up_dir(up)

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
    print(verts)
    print(faces)
    ### Register a mesh
    # `verts` is a Nx3 numpy array of vertex positions
    # `faces` is a Fx3 array of indices, or a nested list
    surface = ps.register_surface_mesh("my mesh", verts, faces,
                             color=do,
                             edge_width=1.0,
                             edge_color=(0.0, 0.0, 0.0),
                             smooth_shade=False)

    surface.set_transparency(0.7)
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

    # Plot particles
    if 0:
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

    # Ground plane options
    ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
    ps.set_ground_plane_height_factor(0.1)  # adjust the plane height
    ps.set_shadow_darkness(0.2)  # lighter shadows
    ps.set_shadow_blur_iters(2)  # lighter shadows
    ps.set_transparency_mode('pretty')
    #ps.look_at((0., -10., 0.), (0., 0., 0.))
    ps.look_at((1., -8., -8.), (0., 0., 0.))
    #ps.set_ground_plane_height_factor(x, is_relative=True)
    ps.set_screenshot_extension(".png")
    # Take a screenshot
    # It will be written to your current directory as screenshot_000000.jpg, etc
    ps.screenshot(fn)
    return ps

# Define local Catenoid
def catenoid_N(r, theta_p, gamma, abc, sref, refinement=1, cdist=1e-10, equilibrium=True):
    v_l, v_u = -1.5, 1.5
    #a, b, c = 1, 0.0, 1
    a, b, c = abc  # Test for unit sphere


    def field(x):
        return 0

    def sech(x):
        return 1 / np.cosh(x)

    def catenoid(u, v):
        x = a * np.cos(u) * np.cosh(v / a)
        y = a * np.sin(u) * np.cosh(v / a)
        z = v
        return x, y, z

    # Equation: x**2/a**2 + y**2/b**2 + z**2/c**2 = 1
    # TODO: Assume that a > b > c?
    # Exact values:
    R = a
    domain = [#(-2.0, 2.0),  # u
              (0.0, 2 * np.pi),  # u
              (v_l, v_u)  # v
              ]
    HC_plane = Complex(2, domain, sfield=field)
    HC_plane.triangulate()
    for i in range(refinement):
        HC_plane.refine_all()

    # Introduce assymetry:
    if 1:
        #HC_plane.refine(1)
        v = HC_plane.V[(0.0, -1.5)]
        v = HC_plane.V[(0.0, 0.0)]
        v = HC_plane.V[(4.712388980384689674, 1.5)]
        for i in range(sref):
            HC_plane.refine_star(v)
        HC_plane.V.process_pools()


    #HC_plane.refine_star(HC.V[10])
    HC = Complex(3, domain)
    bV = set()
    cdist = 1e-8

    u_list = []
    v_list = []
    for v in HC_plane.V:
        #print(f'-')
        #print(f'v.x = {v.x}')
        x, y, z = catenoid(*v.x_a)
        #print(f'tuple(x, y, z) = {tuple([x, y, z])}')
        v2 = HC.V[tuple([x, y, z])]

        u_list.append(v.x_a[0])
        v_list.append(v.x_a[1])

        #TODO: Does not work at all:
        boundary_bool = (
                        v.x[1] == domain[1][0] or v.x[1] == domain[1][1]
                        #v.x[2] == domain[0][0] or v.x[2] == domain[0][1]
                       # or v.x[1] == domain[1][0] or v.x[1] == domain[1][1]
                         )
        if boundary_bool:
            bV.add(v2)

    # Connect neighbours
    for v in HC_plane.V:
        for vn in v.nn:
            v1 = list(HC.V)[v.index]
            v2 = list(HC.V)[vn.index]
            v1.connect(v2)

    # Connect broken skinny edges (by hand, by clicking on the vertices in Polyscope)
    if sref > 0:
        vo = HC.V[(-1.2946832846768446878, 1.5855297404890792918e-16, 0.75)]
        v1 = HC.V[(-1.8406438468393905085, -0.7624196448594611932, 1.3125)]
        v2 = HC.V[(-0.56353868200649509035, -1.3605027290219386633, 0.9375)]
        v3 = HC.V[(-1.2038030912965164839, -1.2038030912965161154, 1.125)]
        vo.connect(v1)
        vo.connect(v2)
        vo.connect(v3)

        vo = HC.V[(1.2946832846768446878, 0.0, 0.75)]
        v1 = HC.V[(0.56353868200649459054, -1.3605027290219388704, 0.9375)]
        v2 = HC.V[(1.2038030912965160417, -1.2038030912965165576, 1.125)]
        v3 = HC.V[(1.8406438468393902285, -0.7624196448594618695, 1.3125)]
        vo.connect(v1)
        vo.connect(v2)
        vo.connect(v3)

    # Connect broken edges of secondary refinement:
    if sref > 1:
        v1 = HC.V[(-0.56353868200649509035, -1.3605027290219386633, 0.9375)]
        v2 = HC.V[(-0.35882532066278876272, -1.8039367053427573885, 1.21875)]
        v1.connect(v2)

        v1 = HC.V[(-1.8406438468393905085, -0.7624196448594611932, 1.3125)]
        v2 = HC.V[(-1.2016108767843284344, -1.7983377626769561825, 1.40625)]
        v1.connect(v2)

        v1 = HC.V[(1.2016108767843277737, -1.7983377626769566239, 1.40625)]
        v2 = HC.V[(1.8406438468393902285, -0.7624196448594618695, 1.3125)]
        v1.connect(v2)

        v1 = HC.V[(0.35882532066278809995, -1.8039367053427575203, 1.21875)]
        v2 = HC.V[(0.56353868200649459054, -1.3605027290219388704, 0.9375)]
        #v2 = HC.V[]
        v1.connect(v2)

    # Connect broken edges of third refinement:
    if sref > 2:
        v1 = HC.V[(-0.65463490728465849945, -2.1580420774668159366, 1.453125)]
        v2 = HC.V[(-1.2016108767843284344, -1.7983377626769561825, 1.40625)]
        v1.connect(v2)

        v1 = HC.V[(-0.2034140000680654245, -2.0652970019161447863, 1.359375)]
        v2 = HC.V[(-0.35882532066278876272, -1.8039367053427573885, 1.21875)]
        v1.connect(v2)

        v1 = HC.V[(0.20341400006806466572, -2.0652970019161448608, 1.359375)]
        v2 = HC.V[(0.35882532066278809995, -1.8039367053427575203, 1.21875)]
        v1.connect(v2)

        v1 = HC.V[(0.65463490728465770663, -2.158042077466816177, 1.453125)]
        v2 = HC.V[(1.2016108767843277737, -1.7983377626769566239, 1.40625)]
        v1.connect(v2)

    # Connect broken edges of fourth refinement:
    if sref > 3:
        v1 = HC.V[(-0.3379417342443479197, -2.2782181172116692076, 1.4765625) ]
        v2 = HC.V[(-0.65463490728465849945, -2.1580420774668159366, 1.453125)]
        v1.connect(v2)

        v1 = HC.V[(-0.108360494340628721166, -2.2057285343782671789, 1.4296875)]
        v2 = HC.V[(-0.2034140000680654245, -2.0652970019161447863, 1.359375)]
        v1.connect(v2)

        v1 = HC.V[(0.10836049434062791079, -2.2057285343782672188, 1.4296875)]
        v2 = HC.V[(0.20341400006806466572, -2.0652970019161448608, 1.359375)]
        v1.connect(v2)

        v1 = HC.V[(0.33794173424434708272, -2.2782181172116693318, 1.4765625)]
        v2 = HC.V[(0.65463490728465770663, -2.158042077466816177, 1.453125)]
        v1.connect(v2)

    if sref > 4:
        v1 = HC.V[(0.17123020824553256848, -2.3213112974838180104, 1.48828125)]
        v2 = HC.V[(0.33794173424434708272, -2.2782181172116693318, 1.4765625)]
        v1.connect(v2)

        v1 = HC.V[(0.0559292292968852944, -2.278304218001336963, 1.46484375)]
        v2 = HC.V[(0.10836049434062791079, -2.2057285343782672188, 1.4296875)]
        v1.connect(v2)

        v1 = HC.V[(-0.05592922929688613144, -2.2783042180013369426, 1.46484375)]
        v2 = HC.V[(-0.108360494340628721166, -2.2057285343782671789, 1.4296875)]
        v1.connect(v2)

        v1 = HC.V[(-0.3379417342443479197, -2.2782181172116692076, 1.4765625)]
        v2 = HC.V[(-0.17123020824553342131, -2.3213112974838179476, 1.48828125)]
        v1.connect(v2)


    # Remerge:
    HC.V.merge_all(cdist=cdist)
    bVc = copy.copy(bV)
    for v in bVc:
        #print(f'bv = {v.x}')
        if not (v in HC.V):
            bV.remove(v)

    H_f = []
    bV_H_f = []
    K_f = []
    bV_K_f = []

    neck_verts = []
    neck_sols = []
    for vert in HC.V:
        if not (vert in bV):
            for u_i, v_i in zip(u_list, v_list):
                x, y, z = catenoid(u_i, v_i)
                va = np.array([x, y, z])
                #print(f'va == vert.x_a = {va == vert.x_a}')
                ba = va == vert.x_a
                if ba.all():
                    #print(f'ba.all() = {ba.all()}')
                    u = u_i
                    v = v_i
                    #nom = c * (a ** 2 * (u ** 2 - 1) + c ** 2 * (u ** 2 + 1))
                    #denom = 2 * a * (u ** 2 * (a ** 2 + c ** 2) + c ** 2) ** (3 / 2.0)
                    H_f_i = 0.0
                    K_f_i = -(sech(v/a))**4 / (a**2)
                    H_f.append(H_f_i)
                    K_f.append(K_f_i)


                    if z == 0.0:
                        neck_verts.append(vert.index)
                        neck_sols.append((H_f_i, K_f_i))

    return HC, bV, K_f, H_f, neck_verts, neck_sols


srefinements = list(range(6))
sref = srefinements[4]
errors_point_wise_1 = []
errors_point_wise_2 = []
errors_total = []
verts = []

for sref in srefinements:
    # Parameters for a water droplet in air at standard laboratory conditions
    gamma = 0.0728  # N/m, surface tension of water at 20 deg C
    rho = 1000  # kg/m3, density
    g = 9.81  # m/s2

    # Numerical parameters #Stated this is what to plaay
    r = 1
    theta_p = 20 * np.pi/180.0  # rad, three phase contact angle
    refinements = []  # [2, 3, 4, 5, 6]   # NOTE: 2 is the minimum refinement needed for the complex to be manifold

    # data containers:
    lp_error = []
    lp_error_2 = []
    geo_error = []  # Imports and physical parameters

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
    HC, bV, K_f, H_f, neck_verts, neck_sols = catenoid_N(r, theta_p, gamma, abc,
                                    sref=sref,
                                    refinement=2,
                                    cdist=1e-5, equilibrium=True)

    bV = set()

    # Define and compute interior (+boundary) curvatures
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
                        n_hnda_i = normalized(hnda_i)[0]
                        hndA_ij_dot_hnda_i = 0.5 * np.sum(np.dot(hnda_ij, n_hnda_i)) / c_ijk

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

    # Point wise estimates:
    if 1:
        sum_HNdA_i = 0.0
        for hndA_i, c_ij in zip(HNdA_i_list, C_ij_i_list):
            #print(f'hndA_i = {hndA_i}')
            hndA_i_c_ij = hndA_i / np.sum(c_ij)
            sum_HNdA_i += np.dot(hndA_i_c_ij, [0, 0, 1])
           # sum_HNdA_i += np.dot(hndA_i_c_ij, [0, 0, -1])

        print('-')
        print(f'HC.V.size() = {HC.V.size()}')
        print(f'len(bV) = {len(bV)}')
        print(f'sum_HNdA_i (point-wise) = {sum_HNdA_i}')
        errors_point_wise_1.append(sum_HNdA_i)

        for hndA_i, c_ij in zip(HNdA_i_list, C_ij_i_list):
            hndA_i_c_ij = hndA_i / np.sum(c_ij)
            sum_HNdA_i += np.dot(hndA_i_c_ij, [0, 0, -1])

        print(f'sum_HNdA_i _ 2  (point-wise) = {sum_HNdA_i}')

        errors_point_wise_2.append(sum_HNdA_i)
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

        for hndA_i, c_ij in zip(HNdA_i_list, C_ij_i_list):
            hndA_i_c_ij = hndA_i #/ np.sum(c_ij)
            sum_HNdA_i += np.dot(hndA_i_c_ij, [0, 0, -1])

        print(f'sum_HNdA_i _ 2  (total) = {sum_HNdA_i}')

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

    verts.append(HC.V.size())

    # Plot in polyscope and save .png image
    if 0:
        fn = f'./fig/bridge_assym_{sref}.png'
        ps = plot_polyscope(HC, fn=fn)

# Plot errors
print(f'errors_point_wise_1 = {errors_point_wise_1}')
print(f'errors_point_wise_2 = {errors_point_wise_2}')
print(f'errors_total = {errors_total}')
print(f'errors_total = {np.sum(errors_total, axis=1)}')
print(f'errors_total = {np.abs(np.sum(errors_total, axis=1))*100}')
print(f'verts = {verts}')

if 1:
    errors_point_wise_1 = np.abs(errors_point_wise_1)
    errors_point_wise_2 = np.abs(errors_point_wise_2)

    Nlist = verts
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax2 = ax.twinx()
    ln1 = ax.loglog(Nlist, errors_point_wise_1*100, 'x', color='tab:blue',
                     label=r'Point-wise errors in $\mathbf{x_{+}}=(0, 0, 1)$ direction  ($\frac{\widehat{H\mathbf{N}} d A_{i} \cdot \mathbf{x_{+}}}{C_i}$)')
    ln2 = ax.loglog(Nlist, errors_point_wise_2*100, 'X', color='tab:red',
                     label=r'Point-wise errors in $\mathbf{x_{-}}=(0, 0, -1)$ direction  ($\frac{\widehat{H\mathbf{N}} d A_{i} \cdot \mathbf{x_{-}}}{C_i}$)')

    #ax2.set_ylabel(r'Young-Laplace error (%)')
    ax.set_xlabel(r'$n$ (number of vertices)')

    ln3 = ax.loglog(Nlist,  np.abs(np.sum(errors_total, axis=1))*100, 'o', color='tab:orange',
                    label=r'Integrated errors (sum of vector components in $\widehat{H\mathbf{N}} d A_{i}$ )')
    ax.set_ylabel(f'Integration error (%)')
    
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    plt.tick_params(axis='y', which='minor')
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    #ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())

    # ax2.set_ylim([1e-15, 1e-13])
    ax.set_ylim([1e-20, 1e4])
#    ax2.set_ylim([1e-20, 1e4])

    plt.tick_params(axis='y', which='minor')

    plt.savefig('./fig/errors.png', bbox_inches='tight', dpi=600)

if 1:
    plt.show()

if 0:
    sref=0
    HC, bV, K_f, H_f, neck_verts, neck_sols = catenoid_N(r, theta_p, gamma, abc,
                                    sref=sref,
                                    refinement=4,
                                    cdist=1e-5, equilibrium=True)

    # Plot in polyscope and save .png image
    if 1:
        fn = f'./fig/assym_bridge_smooth.png'
        ps = plot_polyscope(HC, fn=fn)
if 1:
    HC.V.print_out()
    ps.show()
