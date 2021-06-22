import numpy as np
import copy
from ._curvatures import *
import matplotlib.pyplot as plt



def acap_rise_init_N(r, theta_p, gamma, N=4, refinement=0, cdist=1e-10, equilibrium=True):
    Theta = np.linspace(0.0, 2 * np.pi, N)  # range of theta
    u = Theta  # 0 to 2 pi
    #R = r / np.cos(theta_p)  # = R at theta = 0

    a, b, c = 1, 1, 1  # Test for unit sphere
    a, b, c = 0.5, 0.5, 0.5  # Test for unit sphere
  #  a, b, c = 2, 2, 2  # Test for unit sphere
    # Equation: x**2/a**2 + y**2/b**2 + z**2/c**2 = 1
    # TODO: Assume that a > b > c?

    # Exact values:
    R = a

    # iff a = b = c
    R = np.sqrt(1 * a**2)
    K_f = (1 / R) ** 2
    H_f = 1 / R + 1 / R  # 2 / R
    v = np.pi  #  0 to  pi
    v = theta_p

    u = np.pi  # 0 to pi
    nom = (a*b*c*(np.cos(2*v)*(a**2 + b**2 - 2*c**2)
           + 2*(b**2 - a**2) * np.cos(2*u) * (np.sin(v))**2
           + 3*(a**2 + b**2) + 2 * c**2)
           )

    denom = 8*(c**2 * (np.sin(v))**2 * (a**2 * (np.sin(u))**2 + b**2 * (np.cos(u)) ** 2)
               + a**2 * b**2 * (np.cos(v))**2) ** (3 / 2)

    H_f = nom/denom

    nom = a**2 * b**2 * c**2

    denom = (c**2 * (np.sin(v))**2
             * (a**2 * (np.sin(u))**2 + b**2 * (np.cos(u)) ** 2)
             + a**2 * b**2 * (np.cos(v))**2)**2

    K_f = nom / denom
    print(f'R = {R}')
    print(f'(1 / R) ** 2 = {(1 / R) ** 2}')
    print(f'K_f = {K_f}')
    print(f' 1 / R + 1 / R = { 1 / R + 1 / R}')
    print(f' 1 / R  = { 1 / R}')
    print(f'H_f = {H_f}')

    if 1:
        dp_exact = gamma * (2 / R)  # Pa      # Young-Laplace equation  dp = - gamma * H_f = - gamma * (1/R1 + 1/R2)
        F = []
        nn = []
        F.append(np.array([0.0, 0.0, R * np.sin(theta_p) - R]))
        nn.append([])
        ind = 0
        for theta in Theta:
            ind += 1
            # Define coordinates:
            # x, y, z = sphere(R, theta, phi)
            F.append(np.array([r * np.sin(theta), r * np.cos(theta), 0.0]))
            # Define connections:
            nn.append([])
            if ind > 0:
                nn[0].append(ind)
                nn[ind].append(0)
                nn[ind].append(ind - 1)
                nn[ind].append((ind + 1) % N)

        # clean F
        for f in F:
            for i, fx in enumerate(f):
                if abs(fx) < 1e-15:
                    f[i] = 0.0

        F = np.array(F)
        nn[1][1] = ind

        # Construct complex from the initial geometry:
        HC = construct_HC(F, nn)
        v0 = HC.V[tuple(F[0])]
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
            # New boundary vertices:
            for v in HC.V:
                if v.x[2] == 0.0:
                    bV.add(v)

        # Move to spherical cap
        for v in HC.V:
            z = v.x_a[2]
            z_sphere = z - R * np.sin(theta_p)  # move to origin
            # z_sphere = R * np.cos(phi)  # For a sphere centered at origin
            phi_v = np.arccos(z_sphere/R)
            plane_dist = R * np.sin(phi_v)
            # Push vertices on the z-slice the required distance
            z_axis = np.array([0.0, 0.0, z])  # axial centre
            vec = v.x_a - z_axis
            s = np.abs(np.linalg.norm(vec) - plane_dist)
            nvec = normalized(vec)[0]
            nvec = v.x_a + s * nvec
            HC.V.move(v, tuple(nvec))
            vec = nvec - z_axis
            np.linalg.norm(vec)

        # Rebuild set after moved vertices (appears to be needed)
        bV = set()
        for v in HC.V:
            if v.x[2] == 0.0:
                bV.add(v)

        if not equilibrium:
            # Move to zero, for mean flow simulations
            VA = []
            for v in HC.V:
                if v in bV:
                    continue
                else:
                    VA.append(v.x_a)

            VA = np.array(VA)
            for i, v_a in enumerate(VA):
                v = HC.V[tuple(v_a)]
                v_new = tuple(v.x_a - np.array([0.0, 0.0, v.x_a[2]]))
                HC.V.move(v, v_new)

        if 0:
            R = r / np.cos(theta_p)  # = R at theta = 0
            K_f = (1 / R) ** 2
            H_f = 1 / R + 1 / R  # 2 / R
            rho = 1000  # kg/m3, density
            g = 9.81  # m/s2
            h_jurin = 2 * gamma * np.cos(theta_p) / (rho * g * r)

            for v in HC.V:
                #v = HC.V[tuple(v_a)]
                v_new = tuple(v.x_a + np.array([0.0, 0.0, h_jurin]))
                HC.V.move(v, v_new)

        # TODO: Reconstruct F, nn
        F = []
        nn = []
    return #F, nn, HC, bV, K_f, H_f