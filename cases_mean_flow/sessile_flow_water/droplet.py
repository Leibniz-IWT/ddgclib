# moved from ddgclib
# droplet.py

# scikit-image is used for image processing:
from skimage import data, io
import skimage.feature
import skimage.viewer
import skimage

# NOTE: CP is an API for the NIST database used for the water Equation of State:
# from CoolProp.CoolProp import PropsSI

# Imports and physical parameters
import copy
import sys
import numpy as np
import scipy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# ddg library imports
from ddgclib import *
from ddgclib._complex import *
# from ddgclib._sphere import *
from ddgclib._curvatures import HC_curvatures_sessile, normalized
# from ddgclib._sessile import *
# from ddgclib._capillary_rise_flow import *
from ddgclib._capillary_rise import analytical_cap, cap_rise_init_N
from ddgclib._sphere import sphere_N
from ddgclib._plotting import *



# Configuration dictionary

config = {
    "mode": "main_simulation",   # options: "main_simulation", "cube_test", "sphere_test", "spherical_cap_test",
                                 #          "cylinder_init", "data_processing"

    # main simulation settings
    "refinement": 0,
    "steps": 30,
    "tau": 0.001,

    # image processing settings
    "img_path": "../data/hydrophillic_cropped_murray.png",
    "canny_sigma": 2.0,
    "canny_low_threshold": 0.001,
    "canny_high_threshold": 0.5,

    # physical parameters
    "T_0": 273.15 + 25,
    "P_0": 101.325,
}



# Surface tension of water gamma(T):

def IAPWS(T=298.15):
    T_C = 647.096
    return 235.8e-3 * (1 - (T / T_C)) ** 1.256 * (1 - 0.625 * (1 - (T / T_C)))



# Local compression (approx. Hooke's law)

def local_compressive_forces(v, r, R, theta_p):
    gsc = -R * np.sin(theta_p)

    N_f0 = v.x_a - np.array([0.0, 0.0, gsc])

    d = np.linalg.norm(N_f0)
    if d < R:
        N_f0 = normalized(N_f0)[0]
        return N_f0 * (R - d)
    return np.array([0.0, 0.0, 0.0])



# Mean flow

def mean_flow(HC, bV, params, tau, print_out=False, fix_xy=False):
    gamma, rho, g, r, theta_p, K_f, h = params

    (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
     HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
     Theta_i_cache) = HC_curvatures_sessile(HC, bV, r, theta_p, printout=0)

    bV_new = set()
    for v in HC.V:

        if v in bV:
            rati = (np.pi - np.array(Theta_i) / 2 * np.pi)

            K_H_dA = K_H_i_cache[v.x] * np.sum(C_ij_cache[v.x])
            l_a = 2 * np.pi * r / len(bV)

            Xi = 1
            R_approx = 1 / np.sqrt(K_H_i_cache[v.x])
            theta_p_approx = np.arccos(np.min([r / R_approx, 1]))
            h = R_approx - r * np.tan(theta_p_approx)
            A_approx = 2 * np.pi * R_approx * h

            kg_ds = 2 * np.pi * Xi - K_H_i_cache[v.x] * A_approx
            ds = 2 * np.pi * r
            k_g = kg_ds / ds
            phi_est = np.arctan(R_approx * k_g)

            gamma_bt = gamma * (np.cos(phi_est) - np.cos(theta_p)) * np.array([0, 0, 1.0])
            F_bt = gamma_bt * l_a
            F_bt = np.zeros_like(F_bt)

            dc = local_compressive_forces(v, r, R, theta_p)
            dc *= 1e3

            new_vx = v.x + tau * (F_bt + dc)
            new_vx[2] = 0
            new_vx = tuple(new_vx)

            HC.V.move(v, new_vx)
            bV_new.add(HC.V[new_vx])

        else:
            H = HN_i_cache[v.x]
            height = np.max([v.x_a[2], 0.0])

            dg = np.array([0, 0, -rho * g * height])
            df = gamma * HNdA_i_cache[v.x]

            dc = local_compressive_forces(v, r, R, theta_p)
            dc *= 1e3

            f_k = v.x_a + tau * (df + dc + dg)
            f_k[2] = np.max([f_k[2], 0.0])

            if fix_xy:
                f_k[0] = v.x_a[0]
                f_k[1] = v.x_a[1]

            HC.V.move(v, tuple(f_k))

    return HC, bV_new



# Increment

def incr(HC, bV, params, tau=1e-5, plot=False, verbosity=1, fix_xy=False):
    HC.dim = 3
    print_out = (verbosity == 2)
    HC, bV = mean_flow(HC, bV, params, tau=tau, print_out=print_out, fix_xy=fix_xy)
    return HC, bV



# Main function

def main():
    # Physical parameters
    refinement = config["refinement"]
    steps = config["steps"]
    tau = config["tau"]

    T_0 = config["T_0"]
    P_0 = config["P_0"]

    gamma = IAPWS(T_0)
    rho_0 = 998.2071
    g = 9.81

    theta_p = (63 / 75) * 20 + 30
    theta_p = theta_p * np.pi / 180.0

    r = ((44 / 58) * 0.25 + 1.0) * 1e-3
    h = ((3 / 58) * 0.25 + 0.5) * 1e-3
    v = np.pi * (3 * r ** 2 * h + h ** 3) / 6.0
    Volume = v
    V = v

    m_0 = rho_0 * Volume

    R = r / np.cos(theta_p)
    K_f = (1 / R) ** 2
    params = (gamma, rho_0, g, r, theta_p, K_f, h)

    h_cylinder = V / (np.pi * r**2)

    db = np.array([129, 160, 189]) / 255
    lb = np.array([176, 206, 234]) / 255

    # Data processing (image)
    if config["mode"] in ["data_processing", "main_simulation"]:
        filename = config["img_path"]
        image = skimage.io.imread(fname=filename, as_gray=True)

        edges = skimage.feature.canny(
            image=image,
            sigma=config["canny_sigma"],
            low_threshold=config["canny_low_threshold"],
            high_threshold=config["canny_high_threshold"],
        )

        ypos, xpos = np.argwhere(edges)[:, 0], np.argwhere(edges)[:, 1]
        x0 = 2 + int(919 / 2.0)
        y0 = 2 + image.shape[0]

        scale = 189 / 0.5
        x = (xpos - x0) / scale
        y = (-ypos + y0) / scale

        data_xyz = np.zeros([len(x), 3])
        for ind, (x_i, y_i) in enumerate(zip(x, y)):
            data_xyz[ind, 0] = x_i
            data_xyz[ind, 2] = y_i

        data_xyz *= 1e-3
        data_points = data_xyz

        # Filter weird window thing
        a = data_points
        a = a[~np.any(a == np.max(a[:, 0]), axis=1)]
        a = a[~np.any(a == np.min(a[:, 0]), axis=1)]
        a = a[~np.any(a == np.max(a[:, 2]), axis=1)]
        a = a[~np.any(a == np.min(a[:, 2]), axis=1)]
        data_points = a

    
    # Run modes
    
    mode = config["mode"]

    
    # Cube test
    
    if mode == "cube_test":
        HC = Complex(3, domain=[(0, 1)] * 3)
        HC.triangulate()
        for _ in range(1):
            HC.refine_all()

        bV = HC.boundary()
        trimlist = [v for v in HC.V if v not in bV]
        for v in trimlist:
            HC.V.remove(v)

        HC.dim = 3
        HC.vertex_face_mesh()

        HC.plot_complex(point_color=db, line_color=db, complex_color_f=lb, complex_color_e=db)

        return

    
    # Sphere test
    
    if mode == "sphere_test":
        R = 1
        N = 50
        Theta = np.linspace(0.0, np.pi, N)
        Phi = np.linspace(0.0, 2 * np.pi, N)

        HC, bV = sphere_N(R, Phi, Theta, N=N, refinement=1, cdist=1e-10, equilibrium=True)
        HC.plot_complex()
        return

    
    # Spherical cap test
    
    if mode == "spherical_cap_test":
        gamma = 0.0728
        rho = 1000
        g = 9.81
        N = 7
        refinement = 0
        theta_p = 0.0
        r = 1e-3
        R = r / np.cos(theta_p)
        cdist = 1e-8

        H_f, K_f, dA, k_g_f, dC = analytical_cap(r, theta_p)
        h_jurin = 2 * gamma * np.cos(theta_p) / (rho * g * r)

        F, nn, HC, bV, K_f, H_f = cap_rise_init_N(
            r, theta_p, gamma, N=N, refinement=refinement, equilibrium=1
        )

        params = (gamma, rho, g, r, theta_p, K_f, h)
        HC.V.merge_all(cdist=cdist)

        HC.plot_complex()
        return




    # Cylinder init 
    
    if mode == "cylinder_init":
        cdist = 1e-7
        F, nn, HC, bV, K_f, H_f = cap_rise_init_N(
            r, theta_p, gamma, N=7, refinement=refinement, cdist=cdist, equilibrium=0
        )

        # lowers
        v_list, nn_list = [], []
        HC.dim = 2
        yl_b = HC.boundary(HC.V)
        HC.dim = 3

        for v in yl_b:
            v_list.append(v.x_a)
            nn_list.append([v2.x_a for v2 in v.nn if v2 in yl_b])

        for v in HC.V:
            HC.V.move(v, tuple(v.x_a + np.array([0, 0, h_cylinder])))

        for idx, va in enumerate(v_list):
            v = HC.V[tuple(va)]
            for va2 in nn_list[idx]:
                v2 = HC.V[tuple(va2)]
                v.connect(v2)

        # add half way layer
        for idx, va in enumerate(v_list):
            v = HC.V[tuple(va + 0.5 * np.array([0, 0, h_cylinder]))]
            v_l = HC.V[tuple(va)]
            v_u = HC.V[tuple(va + np.array([0, 0, h_cylinder]))]
            v.connect(v_l)
            v.connect(v_u)

            for va2 in nn_list[idx]:
                v2 = HC.V[tuple(va2 + 0.5 * np.array([0, 0, h_cylinder]))]
                v.connect(v2)
                v.connect(HC.V[tuple(va2)])
                v.connect(HC.V[tuple(va2 + np.array([0, 0, h_cylinder]))])

        return

    
    # Main simulation
    
    if mode == "main_simulation":
        # init droplet
        F, nn, HC, bV, K_f, H_f = cap_rise_init_N(
            r, theta_p, gamma, N=7, refinement=refinement, cdist=1e-7, equilibrium=0
        )

        for i in range(steps):
            HC, bV = incr(HC, bV, params, tau=tau, plot=0, verbosity=2)
            cdist = 2e-4
            if refinement == 1:
                cdist = 1e-4
            if refinement == 2:
                cdist = 0.5e-4
            HC.V.merge_all(cdist=cdist)

            HC.dim = 2
            bV = HC.boundary()
            HC.dim = 3

        (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
         HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
         Theta_i_cache) = HC_curvatures_sessile(HC, bV, r, theta_p, printout=0)

        HC.dim = 2
        bV = HC.boundary()
        HC.dim = 3

        phi_est_list = []
        for v in HC.V:
            if v in bV:
                R_approx = 1 / np.sqrt(K_H_i_cache[v.x])
                theta_p_approx = np.arccos(np.min([r / R_approx, 1]))
                h = R_approx - r * np.tan(theta_p_approx)
                A_approx = 2 * np.pi * R_approx * h
                Xi = 1
                kg_ds = 2 * np.pi * Xi - K_H_i_cache[v.x] * A_approx
                ds = 2 * np.pi * r
                k_g = kg_ds / ds
                phi_est = np.arctan(R_approx * k_g)

                if refinement == 2 and len(v.nn) == 4:
                    phi_est = np.arctan(R_approx / 2 * k_g)

                phi_est_list.append(phi_est * 180 / np.pi)

        minnorm_list = []
        for v in HC.V:
            if v not in bV and abs(v.x_a[1]) <= 1e-10:
                datadist = np.linalg.norm(data_points - v.x_a, axis=1)
                minnorm_list.append(np.min(datadist))

        theta_p_degrees = theta_p * 180.0 / np.pi

        print("=" * 10)
        print("RESULTS:")
        print("=" * 10)
        print(f"phi_est_list = {phi_est_list}")
        print(f"theta_p_degrees = {theta_p_degrees}")
        print(f"phi avg rel error = {np.sum(np.abs(np.array(phi_est_list) - theta_p_degrees) / theta_p_degrees) / len(phi_est_list)}")
        print(f"Avg data error = {np.sum(minnorm_list) / len(minnorm_list)}")
        print(f"HC.V.size() = {HC.V.size()}")
        print(f"len(bV) = {len(bV)}")
        print("=" * 10)


if __name__ == "__main__":
    main()
