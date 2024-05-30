import numpy as np
import copy
from ._curvatures import *
import matplotlib.pyplot as plt




def hyperboloid_N(r, theta_p, gamma, abc, N=4, refinement=2, cdist=1e-10, equilibrium=True):
    Theta = np.linspace(0.0, 2 * np.pi, N)  # range of theta
   # v = Theta  # 0 to 2 pi
  #  u = 0.0    # [-2, 2.0
    #R = r / np.cos(theta_p)  # = R at theta = 0

    #a, b, c = 1, 0.0, 1
    a, b, c = abc  # Test for unit sphere

    def hyperboloid(u, v):
        x = a * np.sqrt(u ** 2 + 1) * np.cos(v)
        y = a * np.sqrt(u ** 2 + 1) * np.sin(v)
        z = c * u
        return x, y, z

    #a, b, c = 0.5, 0.0, 0.5  # Test for unit sphere
  #  a, b, c = 2, 2, 2  # Test for unit sphere
    # Equation: x**2/a**2 + y**2/b**2 + z**2/c**2 = 1
    # TODO: Assume that a > b > c?
    # Exact values:
    R = a

    # iff a = b = c
   # R = np.sqrt(1 * a**2)
    #K_f = (1 / R) ** 2
   # H_f = 1 / R + 1 / R  # 2 / R
  #  v = np.pi  #  0 to  pi
   # v = theta_p

 #   u = np.pi  # 0 to pi


    print(f'R = {R}')
    print(f'(1 / R) ** 2 = {(1 / R) ** 2}')
    #print(f'K_f = {K_f}')
    print(f' 1 / R + 1 / R = { 1 / R + 1 / R}')
    print(f' 1 / R  = { 1 / R}')
    #print(f'H_f = {H_f}')




    if 0:
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



        # TODO: Reconstruct F, nn
        F = []
        nn = []
   #return F, nn, HC, bV, K_f, H_f

    domain = [(-2.0, 2.0),  # u
              #(0.0, np.pi)  # v
              (0.0, 2* np.pi)  # v
              ]
    HC_plane = Complex(2, domain)
    HC_plane.triangulate()
    for i in range(refinement):
        HC_plane.refine_all()

    # H
    HC = Complex(3, domain)
    bV = set()
    cdist = 1e-8

    u_list = []
    v_list = []
    for v in HC_plane.V:
        #print(f'-')
        #print(f'v.x = {v.x}')
        x, y, z = hyperboloid(*v.x_a)
        #print(f'tuple(x, y, z) = {tuple([x, y, z])}')
        v2 = HC.V[tuple([x, y, z])]

        u_list.append(v.x_a[0])
        v_list.append(v.x_a[1])

        #TODO: Does not work at all:
        boundary_bool = (
                        v.x[0] == domain[0][0] or v.x[0] == domain[0][1]
                        #v.x[2] == domain[0][0] or v.x[2] == domain[0][1]
                       # or v.x[1] == domain[1][0] or v.x[1] == domain[1][1]
                         )
        #print(f'boundary_bool = {boundary_bool}')
        if boundary_bool:
            bV.add(v2)
            #boundary_bool = False

    # Connect neighbours
    for v in HC_plane.V:
        for vn in v.nn:
            v1 = list(HC.V)[v.index]
            v2 = list(HC.V)[vn.index]
            v1.connect(v2)


   # for v1, v2 in HC_plane
    #print(f'list(HC.V) = {list(HC.V)}')

    # Remerge:
    HC.V.merge_all(cdist=cdist)
    bVc = copy.copy(bV)
    for v in bVc:
        #print(f'bv = {v.x}')
        if not (v in HC.V):
            bV.remove(v)

    for v in bV:
        print(f'bv = {v.x}')

    if 0:
        u_listc, v_listc = copy.copy(u_list), copy.copy(v_list)
        print(f'u_list = {u_list}')
        print(f'v_list = {v_list}')
        for u_i, v_i in zip(u_listc, v_listc):
            x, y, z = hyperboloid(u_i, v_i)
            print(f' -')
            print(f' tuple([x, y, z]) in HC.V.cache = {tuple([x, y, z]) in HC.V.cache}')

            # print(f'ind = {ind}')
            # if tuple(x, y, z)
            if not (tuple([x, y, z]) in HC.V.cache):
                # pass
                u_ind = u_list.index(u_i)
                v_ind = v_list.index(v_i)
                print(f'u_i = {u_i}')
                print(f'u_ind = {u_ind}')
                print(f'v_i = {v_i}')
                print(f'v_ind = {v_ind}')
                u_list.pop(u_ind)
                v_list.pop(v_ind)

        u = np.array(u_list)
        v = np.array(v_list)
        print(f'u = {u}')
        print(f'v = {v}')
        nom = c * (a**2*(u**2 - 1) + c**2 * (u**2 + 1))
        denom = 2 * a * (u**2 * (a**2 + c**2) + c**2)**(3/2.0)
        H_f = nom / denom
        K_f = - c**2 / ((u**2 * (a**2 + c**2) + c**2)**2)

        for ind, vert in enumerate(HC.V):
            print(f'-')
            print(f'ind = {ind}')
            print(f'v = {vert.x}')
            #x, y, z = hyperboloid(u[ind], v[ind])
            print(f'x, y, z =  {x, y, z}')

            nom = c * (a**2*(u**2 - 1) + c**2 * (u**2 + 1))
            denom = 2 * a * (u**2 * (a**2 + c**2) + c**2)**(3/2.0)


        for u_i, v_i in zip(u_list, v_list):
            x, y, z = hyperboloid(u_i, v_i)
            print(f' tuple([x, y, z]) in HC.V.cache = {tuple([x, y, z]) in HC.V.cache}')

            vert = HC.V[tuple([x, y, z])]
            print(f'vert.index = {vert.index}')
            #print(f'ind = {ind}')
            #if tuple(x, y, z)
            if not (tuple([x, y, z]) in HC.V.cache):
                print('WARRNIGN! not in cache')

    H_f = []
    bV_H_f = []
    K_f = []
    bV_K_f = []

    neck_verts = []
    neck_sols = []
    for vert in HC.V:
        if not (vert in bV):
            for u_i, v_i in zip(u_list, v_list):
                x, y, z = hyperboloid(u_i, v_i)
                va = np.array([x, y, z])
                print(f'va == vert.x_a = {va == vert.x_a}')
                ba = va == vert.x_a
                if ba.all():
                    print(f'ba.all() = {ba.all()}')
                    u = u_i
                    v = v_i
                    nom = c * (a ** 2 * (u ** 2 - 1) + c ** 2 * (u ** 2 + 1))
                    denom = 2 * a * (u ** 2 * (a ** 2 + c ** 2) + c ** 2) ** (3 / 2.0)
                    H_f_i = nom / denom
                    K_f_i = - c ** 2 / ((u ** 2 * (a ** 2 + c ** 2) + c ** 2) ** 2)
                    H_f.append(H_f_i)
                    K_f.append(K_f_i)
                    #TODO: MERGE NEAR VERTICES

                    if z == 0.0:
                        neck_verts.append(vert.index)
                        neck_sols.append((H_f_i, K_f_i))

    print(f'len(H_f) = {len(H_f)}')
    print(f'len(K_f) = {len(K_f)}')
    print(f'HC.V.size = {HC.V.size()}')
    print(f'len(bV) = {len(bV)}')
    return HC, bV, K_f, H_f, neck_verts, neck_sols