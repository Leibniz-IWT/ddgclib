# scikit-image is used for image processing:
from skimage import data, io
import skimage.feature
import skimage.viewer
import skimage

# NOTE: CP is an API for the NIST database used for the water Equation of State:
#from CoolProp.CoolProp import PropsSI

# Imports and physical parameters
import copy
import sys
import numpy as np
import scipy
#from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
#%matplotlib notebook


# Local library
from ddgclib import *
from ddgclib._complex import *
from ddgclib._sphere import *
from ddgclib._curvatures import * #plot_surface#, curvature
from ddgclib._sessile import *
from ddgclib._capillary_rise_flow import * #plot_surface#, curvature
from ddgclib._eos import *
from ddgclib._plotting import *
#from ddgclib.curvatures import plot_surface, curvature

# Equation of state for water droplet:
def eos(P=101.325, T=298.15):
    # P in kPa T in K
    return PropsSI('D','T|liquid',298.15,'P',101.325,'Water') # density kg /m3

# Surface tension of water gamma(T):
def IAPWS(T=298.15):
    T_C = 647.096  # K, critical temperature
    return 235.8e-3 * (1 -(T/T_C))**1.256 * (1 - 0.625*(1 - (T/T_C)))  # N/m


# Parameters
if 1:
    T_0 = 273.15 + 25  # K, initial tmeperature
    P_0 = 101.325  # kPa, Ambient pressure
    gamma = IAPWS(T_0)  # N/m, surface tension of water at 20 deg C
    #rho_0 = eos(P=P_0, T=T_0)  # kg/m3, density
    rho_0 = 998.2071  # kg/m3, density, STP
    g = 9.81  # m/s2

    theta_p = (63 / 75) * 20 + 30
    theta_p = theta_p * np.pi / 180.0
    r = ((44 / 58) * 0.25 + 1.0) * 1e-3  # height mm --> m  # Radius
    h = ((3 / 58) * 0.25 + 0.5) * 1e-3  # height mm --> m
    v = np.pi * (
            3 * r ** 2 * h + h ** 3) / 6.0  # Volume in m3 (Segment of a sphere, see note above)
    Volume, V = v, v

    #TEMP:
   # r, h, v, V, Volume = r*1e3, h*1e3, v*1e3, V*1e3, Volume*1e3
    m_0 = rho_0 * Volume  # kg, initial mass (kg/m3 * m3)
    print(f'theta_p = {theta_p * 180.0 / np.pi}')
    print(f'r = {r * 1e3} mm')
    print(f'h = {h * 1e3} mm')
    print(f'v = {v * 1e9} mm^3')
    print(f'V = {V * 1e9} mm^3')
    print(f'Volume = {Volume * 1e9} mm^3')

#from ddgclib.curvatures import plot_surface, curvature

# Colour scheme for surfaces
db = np.array([129, 160, 189]) / 255  # Dark blue
lb = np.array([176, 206, 234]) / 255  # Light blue

# droplet data:
if 1:
    # Find the Abscissa using scale interpolations
    voxels = 126  # Voxels between 100 and 200 in 4.c
    voxels_m =  126 * (33/100.0)

    # Fig. 5 is the droplet at 133 s, we read off the contact angle, radius and heigh from Fig. 4.c
    # The ratios below are the voxel fractions of where we read off the scales:

    # process cropped image
    img_str = '../data/hydrophillic_cropped_murray.png'
    I_png = plt.imread(img_str)
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


    gray = rgb2gray(I_png)
    if 0:
        plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)


    scale = 189  # voxels / 0.5 mm  (from the bar given in the paper)
    scale = 189 / 0.5  # voxels /  mm
    np.max(gray), np.min(gray)
    # Edgte detection
    if 1:
        # read command-line arguments
        # filename = sys.argv[1]
        filename = img_str
        # sigma = 0.2
        # low_threshold = float(sys.argv[3])
        # high_threshold = float(sys.argv[4])
        sigma = 5.0  # 2.0
        sigma = 10.0  # 2.0
        low_threshold = 0.1
        high_threshold = 0.3

        image = skimage.io.imread(fname=filename, as_gray=True)
        # viewer = skimage.viewer(image=image)
        viewer = skimage.viewer.ImageViewer(image=image)
        if 0:
            viewer.show()
        image_cropped = image  # [210:400,180:440]
        sigma = 2.0
        # sigma = 5.0
        # sigma = 5.5
        # low_threshold = 0.2#0.66#0.05
        low_threshold = 0.001  # 0.05
        # low_threshold = 0.2#0.05
        high_threshold = 0.50

        # max, min pair was = (0.4862258831709623, 0.0)
        edges = skimage.feature.canny(
            image=image_cropped,
            sigma=sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        viewer = skimage.viewer.ImageViewer(edges)

        # Show skimage plots
        if 0:
            viewer.show()
            plt.show()

        # Convert coordinates from edge data:
        ypos, xpos = np.argwhere(edges)[:, 0], np.argwhere(edges)[:, 1]
        xpos, ypos

        #np.savetxt('./data/output_murray.txt', np.argwhere(edges),
        #           delimiter=';')

        x0 = 2 + int(
            919 / 2.0)  # TODO: Detect maximum of edge contour to find symmetry automatically
        y0 = 2 + image_cropped.shape[0]
        x = (xpos - x0) / scale  # voxels --> mm
        y = (-ypos + y0) / scale  # voxels --> mm

        # Matplotlib scatter plot:
        if 0:
            plt.figure()
            plt.scatter(x, y)
            plt.xlabel('mm')
            plt.ylabel('mm')
            plt.show()

# Mean flow simulation

h_cylinder =  V / ( np.pi * r**2 )  # V =  np.pi * r**2 * h
h_cylinder, h, r

print(f'h_cylinder = {h_cylinder}')

# Cube test
if 0:
    HC = Complex(3, domain=[(0, 1), ] * 3)
    #HC = Complex(3, domain=[(-1, 1), ] * 3)
    HC.triangulate()
    for i in range(1):
        HC.refine_all()

    #v = HC.V[(0.0, 0.0, 0.0)]
    #HC.V.remove(v)

    #bV = HC.boundary()
    bV = HC.boundary()
    trimlist = []
    for v in HC.V:
        if v not in bV:
            trimlist.append(v)

    for v in trimlist:
        HC.V.remove(v)

    if 1:
        HC.dim = 3
        HC.vertex_face_mesh()
        HC.simplices_fm

        fig, axes, fig_s, axes_s = HC.plot_complex(point_color=db,
                                                   line_color=db,
                                                   complex_color_f=lb,
                                                   complex_color_e=db
                                                   )

        #plot.show()
    bV = set()
    (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
     HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
     Theta_i_cache) = HC_curvatures(HC, bV, r, theta_p, printout=0)

    int_C = 0
    for cij in C_ij_cache:
        # print(f'cij = {cij}')
        # print(f'C_ij_cache[cij] = {C_ij_cache[cij] }')
        int_C += np.sum(C_ij_cache[cij])
        pass
    # int_C = np.sum(HN_i)

    print(f'int_C = {int_C}')
    print(f'int_C/6.0 = {int_C / 6.0}')
    print(f'int_C/6.0 = {int_C / 6.0}')
    print(f'V = {1} m^3')

    fig, axes, fig_s, axes_s = HC.plot_complex(point_color=db,
                                               line_color=db,
                                               complex_color_f=lb,
                                               complex_color_e=db
                                               )


    # Compute simplices:
    HC.dim = 2
    HC.vertex_face_mesh()
    HC.simplices_fm

# Sphere test
if 0:
    R = 1
    N = 50
    Theta = np.linspace(0.0, np.pi, N)  # range of theta
    Phi = np.linspace(0.0, 2 * np.pi, N)  # range of phi
    #Phi = np.linspace(0.0, 0.5*np.pi, N)  # range of phi
    HC, bV = sphere_N(R, Phi, Theta, N=N,
                       refinement=1, cdist=1e-10
                       , equilibrium=True
                       )
    HC.plot_complex()
    plot.show()
# Old sphere
if 0:
    pass

    def sphere(R, theta, phi):
        return R * np.cos(theta) * np.sin(phi), R * np.sin(theta) * np.sin(phi), R * np.cos(phi)


    x, y, z = sphere(R, theta, phi)

    if 0:
        # Theta = [0.0, 0.0, (1/2.0)*np.pi, np.pi, 1.5*np.pi, 2.0*np.pi]
        Theta = [0.0, 0.0, (1.0 / 3.0) * 2 * np.pi, (2.0 / 3.0) * 2 * np.pi,
                 2 * np.pi]
        Thata = np.linspace(0, 2 * np.pi, 10)
        # Phi = [0.0, np.pi, (1/2.0)*np.pi, (1/2.0)*np.pi, (1/2.0)*np.pi, (1/2.0)*np.pi]
       # Phi = [0.0, np.pi, (1 / 2.0) * np.pi, (1 / 2.0) * np.pi, (1 / 2.0) * np.pi]
        Phi = np.linspace(0, 2 * np.pi, 10)
    F = []
    for theta, phi in zip(Theta, Phi):
        x, y, z = sphere(R, theta, phi)
        F.append(np.array([x, y, z]))

    #F = np.array(F)

# Spherical cap test
if 1:
    # Parameters for a water droplet in air at standard laboratory conditions
    gamma = 0.0728  # N/m, surface tension of water at 20 deg C
    rho = 1000  # kg/m3, density
    g = 9.81  # m/s2

    N = 7
    refinement = 0#1
    #theta_p = 45 * np.pi / 180.0  # Three phase contact angle
    theta_p = 0.0 * np.pi / 180.0  # Three phase contact angle
    r = 1e-3  # Radius of the tube (1 mm)
    #NOTE: MERGING DOES NOT WORK CORRECTLY WITH 1e-3, FIX FOR MICRODROPLETS
    r = 1  # Radius of the tube (1 mm)
    r = np.array(r, dtype=np.longdouble)
    R = r / np.cos(theta_p)
    equilibrium = 1  #
    cdist = 1e-8

    theta_p = np.array(theta_p, dtype=np.longdouble)

    H_f, K_f, dA, k_g_f, dC = analytical_cap(r, theta_p)
    print(f'H_f, K_f, dA, k_g_f, dC = {H_f, K_f, dA, k_g_f, dC}')
    k_K = k_g_f / K_f  #TODO: We no longer need this?
    h_jurin = 2 * gamma * np.cos(theta_p) / (rho * g * r)

    # Prepare film and move it to 0.0
    F, nn, HC, bV, K_f, H_f = cap_rise_init_N(r, theta_p, gamma, N=N,
                                              refinement=refinement,
                                              equilibrium=equilibrium
                                              )

    #h = 0.0
    params = (gamma, rho, g, r, theta_p, K_f, h)
    # HC.V.print_out()
    HC.V.merge_all(cdist=cdist)

    # Add boundary centroid:
    if 0:
        int_x, int_y, int_z = 0.0, 0.0, 0.0
        for v in bV:
            int_x += v.x_a[0]
            int_y += v.x_a[1]
            int_z += v.x_a[2]
        int_x = int_x/len(bV)
        int_y = int_y/len(bV)
        int_z = int_z/len(bV)

        vc = HC.V[(int_x, int_y, int_z)]
        #print(f'bV = {bV}')
        for v in bV:
            #print(f'v.x = {v.x}')
            v.connect(vc)

        for v in HC.V:
            #print(f'v.nn = {v.nn}')
            #print(f'v.x = {v.x}')
            #print(f'len(v.nn) = {len(v.nn)}')
            pass

        HC.V.merge_all(cdist=cdist)
        bV = set([])  # We closed the boundary
        #print(f'bV = {bV}')
        if 0:
            HC.plot_complex()
            plot.show()

    (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
     HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
     Theta_i_cache) = HC_curvatures(HC, bV, r, theta_p, printout=0)

    int_C = 0
    for cij in C_ij_cache:
        # print(f'cij = {cij}')
        # print(f'C_ij_cache[cij] = {C_ij_cache[cij] }')
        int_C += np.sum(C_ij_cache[cij])
        pass
    # int_C = np.sum(HN_i)

    print(f'len(bV) = {len(bV)}')
    print(f'int_C = {int_C}')
    print(f'int_C/6.0 = {int_C / 6.0}')
    print(f'int_C/6.0 = {int_C / 6.0}')
    theta = np.arcsin(r/R)  # sin (theta) = a /R
    print(f'theta = {theta * 180 /np.pi}')
    V = (np.pi/3.0) * R**3 * (2 + np.cos(theta)
                              * (1 - np.cos(theta))**2)
    A = 2* np.pi * R**2 * (1 - np.cos(theta))
    print(f'V = {V} m^3')
    V_sphere = (4/3.0) * np.pi * R**3
    A_sphere = 4 * np.pi * R**2
    print(f'V (half sphere) = {V_sphere/ 2.0} m^3')
    #print(f'V (half sphere)/int_C = {(V_sphere/ 2.0)/int_C} m^3')
    print(f'A = {A} m^2')
    print(f'A_sphere (half sphere) = {A_sphere/2.0} m^2')
    print(f'intC - circ = {int_C - np.pi * R**2} m^2')

    #
    print('=====================')
    print('Mean normal approach:')
    print('=====================')
    print(f'HN_i = {HN_i}')
    print(f'K_H_i = {K_H_i}')
    print(f'C_ij = {C_ij}')

    if 0:
        ps = pplot_surface(HC)
        # View the point cloud and mesh we just registered in the 3D UI
        ps.show()

    HC.plot_complex()
    plot.show()

# Cylinder init attempt:
if 0:
    # Initiate a cubical complex
    HC = Complex(3, domain=[(-r, r), ] * 3)
    HC.triangulate()

    for i in range(1):
        HC.refine_all()

    # for v in HC2V:
    #    HC.refine_star(v)
    del_list = []
    for v in HC.V:
        if np.any(v.x_a == -r) or np.any(v.x_a == r):
            if np.any(v.x_a[2] == -r):
                if np.any(v.x_a[0:2] == -r) or np.any(v.x_a[0:2] == r):
                    continue
                else:
                    del_list.append(v)
            else:
                continue
        else:
            del_list.append(v)

    for v in del_list:
        HC.V.remove(v)

    # Shrink to circle and move to zero
    for v in HC.V:
        x = v.x_a[0]
        y = v.x_a[1]
        f_k = [x * np.sqrt(r ** 2 - y ** 2 / 2.0) / r,
               y * np.sqrt(r ** 2 - x ** 2 / 2.0) / r, v.x_a[2]]
        HC.V.move(v, tuple(f_k))

    # Move up to zero (NOTE: DO NOT DO THIS IN THE CIRCLE LOOP
    # BECUASE SAME VERTEX INDEX BREAKS CONNECTIONS IN LOOP DURING MOVE:
    # TODO: FIX THIS IN THE hyperct LIBRARY CODE)
    for v in HC.V:
        if (v.x_a[2] == -r) or (v.x_a[2] == 0.0):
            continue
        f_k = [v.x_a[0], v.x_a[1], v.x_a[2] + r]
        HC.V.move(v, tuple(f_k))

    for v in HC.V:
        if (v.x_a[2] == 2 * r) or (v.x_a[2] == -r):
            continue
        f_k = [v.x_a[0], v.x_a[1], v.x_a[2] + r]
        HC.V.move(v, tuple(f_k))

    for v in HC.V:
        if (v.x_a[2] == 2 * r) or (v.x_a[2] == r):
            continue
        f_k = [v.x_a[0], v.x_a[1], v.x_a[2] + r]
        HC.V.move(v, tuple(f_k))

    ## Move to h
    for v in HC.V:
        if (v.x_a[2] == 0.0) or (v.x_a[2] == r):
            continue
        f_k = [v.x_a[0], v.x_a[1], v.x_a[2] + h_cylinder - 2 * r]
        HC.V.move(v, tuple(f_k))

    for v in HC.V:
        if (v.x_a[2] == r):
            f_k = [v.x_a[0], v.x_a[1], v.x_a[2] + 0.5 * h_cylinder - r]
            HC.V.move(v, tuple(f_k))

        # Find set of boundary vertices
    bV = set()
    for v in HC.V:
        # print('-')
        # print(f'v.x_a = {v.x_a}')
        # print(f'v.x_a[2] == 0.0 = {v.x_a[2] == 0.0}')
        if v.x_a[2] == 0.0:
            bV.add(v)
            # print(f'bV = {bV}')
        else:
            continue

    if 1:
        fig, axes, fig_s, axes_s = HC.plot_complex(point_color=db,
                                                   line_color=db,
                                                   complex_color_f=lb,
                                                   complex_color_e=db
                                                   )


    # Below can be used to confirm above bV
    if 0:
        for bv in bV:
            print(f'bv = {bv}')
        HC.dim = 2
        bV2 = HC.boundary()

        print('bV2')
        for bv in bV2:
            print(f'bv = {bv}')

    # axes.set_xlim3d(-(0.1*r + r) , 0.1*r + r)
    # axes.set_ylim3d(-(0.1*r + r) , 0.1*r + r)
    # axes.set_zlim3d(-(0.1*r + r) , 0.1*r + 2*r)
    # Test volumes:
    if 1:
        (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
         HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
         Theta_i_cache) = HC_curvatures(HC, bV, r, theta_p, printout=0)

        int_C = 0
        for cij in C_ij_cache:
            #print(f'cij = {cij}')
            #print(f'C_ij_cache[cij] = {C_ij_cache[cij] }')
            int_C += np.sum(C_ij_cache[cij])
            pass
        #int_C = np.sum(HN_i)

        print(f'int_C = {int_C}')
        print(f'int_C/6.0 = {int_C/6.0}')
        print(f'V = {V * 1e9} mm^3')
        print(f'int_C= {int_C * 1e9} mm^3')
        print(f'int_C/6.0 = {int_C/6.0 * 1e9} mm^3')
        print(f'int_C = {int_C * 1e6} mm^2')

        A_cylinder = 2 * np.pi * r * h_cylinder
        A_cylinder =  A_cylinder + (np.pi * r**2) # not including bottom face
        print(f'A_cylinder = {A_cylinder * 1e6} mm^2')

# Sanity checks for cylinder and volume, connect all to COM
if 0:
    ## Compute volume
    # Compute COM
    f = []
    for v in HC.V:
        f.append(v.x_a)

    f = np.array(f)
    # COM = numpy.average(nonZeroMasses[:,:3], axis=0, weights=nonZeroMasses[:,3])
    com = np.average(f, axis=0)
    com

    bf = []
    for v in bV:
        bf.append(v.x_a)

    bf = np.array(bf)
    # COM = numpy.average(nonZeroMasses[:,:3], axis=0, weights=nonZeroMasses[:,3])
    bcom = np.average(bf, axis=0)
    bcom

    HC2 = copy.copy(HC)
    for v in HC2.V:
        v.connect(HC2.V[tuple(com)])

    for v in bV:
        v.connect(HC2.V[tuple(bcom)])

    HC2.V[tuple(com)].connect(HC2.V[tuple(bcom)])
    # Compute simplices:
    HC2.vertex_face_mesh()
    HC2.simplices_fm

if 0:
    fig, axes, fig_s, axes_s = HC2.plot_complex(point_color=db,
                                                line_color=db,
                                                complex_color_f=lb,
                                                complex_color_e=db
                                                )

plt.show()