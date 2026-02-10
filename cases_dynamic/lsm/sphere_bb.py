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
from ddgclib import *
from hyperct import *
from ddgclib._curvatures import * #plot_surface#, curvature
from ddgclib._capillary_rise_flow import * #plot_surface#, curvature
from ddgclib._eos import *
from ddgclib._misc import *
from ddgclib._plotting import *



gamma = 0.0728  # N/m, surface tension of water at 20 deg C
rho = 1000  # kg/m3, density
g = 9.81  # m/s2

# Parameters from EoS:
#T_0 = 273.15 + 25  # K, initial tmeperature
#P_0 = 101.325  # kPa, Ambient pressure
#gamma = IAPWS(T_0)  # N/m, surface tension of water at 20 deg C
#rho_0 = eos(P=P_0, T=T_0)  # kg/m3, density
# Capillary rise parameters
r = 1 # Radius of the tube (20 mm)
bV = set([])


x_c1 = np.array([0.5, 0.5, 0.5])
x_c2 = np.array([1.0, 1.0, 1.0])
x_cell1 = x_c1 + np.random.rand(3, 3)
x_cell2 = np.array([[0, 0, 0],
                    [0, 1, 1],
                    [0, 1, 0],
                    [0, 0, 1],
                   ])

x_cell2 =  x_cell2 + -0.1* np.random.rand(4, 3)
x_cell3 = x_c2 + np.random.rand(4, 3)
#x_full = np.vstack([x_c1, x_cell1], [x_c2, x_cell2], )
cells = [
         [x_cell1],
         [x_cell2],
         ]

#cells =

def cell_con(cells, dim=3, cdist=1e-14):
    HC = Complex(dim)
    cell_centres = [] # needed to complete the trianbulatio
    for x_cell in cells:
        print(x_cell)
        # Case of only 3 points:
        if x_cell.shape[0] == 3:
            V = []
            for v in x_cell:
                V.append(HC.V[tuple(v)])

            print(f'V = {V}')
            #v_c =
            for i1, v in enumerate(V):
                # Optionally connect to centre vertex:
                if 0:
                    HC.V[tuple(x_c)].connect(V[i1])
                for i2, v in enumerate(V):
                    V[i1].connect(V[i2])

                #V[si1].connect(V[si2])

        # Case of 4 full points
        if x_cell.shape[0] == 4:
            V = []
            for v in x_cell:
                V.append(HC.V[tuple(v)])
            for i1, v in enumerate(V):
                # Find 2 nearest points (TODO: Optimise later)
                dx = x_cell - v.x_a
                dxn = np.linalg.norm(dx, axis=1)
                con = np.argsort(dxn)[1:3]
                for i2 in con:
                    V[i1].connect(V[i2])
    HC.V.merge_all(cdist=cdist)
    return HC.plot_complex()

HC = cell_con(cells, dim=3)
plt.show()
fn = ''
ps = plot_polyscope(HC, fn=fn, up="z_up", stl=True)
# ps = pplot_surface(HC)
# View the point cloud and mesh we just registered in the 3D UI
ps.show()

def tri_cell(x_c, x_cell):
    #Tri = scipy.spatial.Delaunay(x_full)
    Tri = scipy.spatial.Delaunay(x_cell)
    print(Tri.points)
    print(Tri.simplices)
    HC = Complex(x_full.shape[1])
    V = []
    for v in Tri.points:
        V.append(HC.V[tuple(v)])

    print(f'V = {V}')
    for s in Tri.simplices:
        for si1 in s:
            for si2 in s:
                V[si1].connect(V[si2])

    HC.plot_complex()
    plt.show()
    fn = ''
    ps = plot_polyscope(HC, fn=fn, up="z_up", stl=True)
    # ps = pplot_surface(HC)
    # View the point cloud and mesh we just registered in the 3D UI
    ps.show()
    return

#import matplotlib.pyplot as plt
#plt.triplot()
def HC_curvatures(HC, bV, r, theta_p, printout=False):
    R = r / np.cos(theta_p)
    K_f = (1 / R) ** 2
    H_f = 2 / R  # 2 / R
    HNdA_ij = []
    HN_i = []
    C_ij = []
    K_H_i = []
    HNdA_i_Cij = []
    Theta_i = []

    N_i = []  # Temp cap rise normal

    HNdA_i_cache = {}
    HN_i_cache = {}
    C_ij_cache = {}
    K_H_i_cache = {}
    HNdA_i_Cij_cache = {}
    Theta_i_cache = {}

    for v in HC.V:
        N_f0 = np.array(
            [0.0, 0.0, R * np.sin(theta_p)]) - v.x_a  # First approximation
        N_f0 = normalized(N_f0)[0]
        N_i.append(N_f0)
        F, nn = vectorise_vnn(v)
        # Compute discrete curvatures
        c_outd = b_curvatures_hn_ij_c_ij(F, nn, n_i=N_f0)
        # Append lists
        HNdA_ij.append(c_outd['HNdA_i'])
        #HNdA_ij_dot.append(np.sum(np.dot(c_outd['HNdA_ij'], c_outd['n_i'])))
        HN_i.append(c_outd['HN_i'])
        C_ij.append(c_outd['C_ij'])
        K_H_i.append(c_outd['K_H_i'])
        HNdA_i_Cij.append(c_outd['HNdA_ij_Cij'])
        Theta_i.append(c_outd['theta_i'])

        # Append chace
        HNdA_i_cache[v.x] = c_outd['HNdA_i']
        HN_i_cache[v.x] = c_outd['HN_i']
        C_ij_cache[v.x] = c_outd['C_ij']
        K_H_i_cache[v.x] = c_outd['K_H_i']
        HNdA_i_Cij_cache[v.x] = c_outd['HNdA_ij_Cij']
        Theta_i_cache[v.x] = c_outd['theta_i']

    if printout:
        print('.')
        print(f'HNdA_ij = {HNdA_ij}')
        print(f'HN_i = {HN_i}')
        print(f'C_ij = {C_ij}')
        print(f'K_H_i = {K_H_i}')
        print(f'HNdA_i_Cij = {HNdA_i_Cij}')
        print(f'Theta_i= {Theta_i}')
        print(f'np.array(Theta_i) in deg = {np.array(Theta_i) *180/np.pi}')
        print(f'np.array(Theta_i)/np.pi= {np.array(Theta_i) / np.pi}')
        # s = r * theta
        # circ = 2 pi r
        # circ / s = 2 pi / theta
        rati = 2 * np.pi /np.array(Theta_i)
        rati = 2 * np.pi / (2 * np.pi - np.array(Theta_i))
        rati =  2 * np.pi / (2 * np.pi - np.array(Theta_i))
        rati =  (np.pi - np.array(Theta_i)/ 2 * np.pi )
        #rati = np.array(Theta_i) / (2 * np.pi)
        #print(f' rati = 2 * np.pi /np.array(Theta_i)= { rati}')
        print(f' rati = { rati}')
       # rati =  (2 * np.pi  - np.array(Theta_i))/np.pi
        #print(f' rati = (2 * np.pi  - Theta_i)/np.pi = { rati}')
        print(f'HNdA_i[1] * rati[1]  = {HNdA_ij[1] * rati[1] }')
        print(f'C_ij   = {C_ij }')
        #print(f'np.sum(C_ij)   = {np.sum(C_ij) }')
       # print(f'HNdA_i / np.array(C_ijk)  = {HNdA_i  / np.array(C_ijk)}')

        print('.')
        print(f'HNdA_i_Cij = {HNdA_i_Cij}')

        #print(f'K_H = {K_H}')
        print('-')
        print('Errors:')
        print('-')
       # print(f'hnda_i_sum = {hnda_i_sum}')
       # print(f'K_H_2 = {K_H_2}')
       # print(f'C_ijk = {C_ijk}')
      #  print(f'np.sum(C_ijk) = {np.sum(C_ijk)}')
      #  print(f'A_ijk = {A_ijk}')
     #   print(f'np.sum(A_ijk) = {np.sum(A_ijk)}')
      #  print(f'K_H_2 dA = {np.sum(K_H_2) * np.sum(C_ijk)}')
      #  print(f'K_H_2 dA = {np.sum(K_H_2) * np.sum(A_ijk)}')
      #  print(f'np.sum(K_H_2) = {np.sum(K_H_2)}')
      #  print(f'HNdA_ij_dot_hnda_i = {np.array(HNdA_ij_dot_hnda_i)}')
     #   print(
    #        f'np.sum(HNdA_ij_dot_hnda_i) = {np.sum(np.array(HNdA_ij_dot_hnda_i))}')

        print(f'K_H_i - K_f = {np.array(K_H_i) - K_f}')
        #print(f'HNdA_ij_dot_hnda_i  - H_f = {HNdA_ij_dot_hnda_i - H_f}')
        #print(f'HHN_i  - H_f = {HNdA_ij_dot_hnda_i - H_f}')
        print(f'HN_i  - H_f = {HN_i - H_f}')
        print(f'HNdA_i_Cij  - H_f = {HNdA_i_Cij - H_f}')

        #print(f'np.sum(C_ij) = {np.sum(C_ij)}')

    return (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
            HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
            Theta_i_cache)

def b_curvatures_hn_ij_c_ij(F, nn, n_i=None):
    """
    F: Array of vectors forming the star domain of v_i = F[0]
    nn: Connections within each vertex in union with the star domain
    n_i: Normal vector at vertex v_i (approx.)

    :return: cout: A dictionary of local curvatures
    """
    # NOTE: We need a better solution to ensure signed quantities retain their structure.
    #      The mesh must be ordered to ensure we obtain the correct normal face directions
    # print(f'.')
    # print(f'.')
    # print(f'n_i = {n_i}')
    # print(f'F = {F}')
    # print(f'nn = {nn}')
    if n_i is None:
        n_i = normalized(F[0])[0]
    # print(f'n_i = {n_i}')
    # TODO: Later we can cache these containers to avoid extra computations
    # Edges from i to j:
    E_ij = F[1:] - F[0]
    E_ij = np.vstack([np.zeros(3), E_ij])
    E_jk = np.zeros_like(E_ij)
    E_ik = np.zeros_like(E_ij)

    E_jl = np.zeros_like(E_ij)
    E_il = np.zeros_like(E_ij)
    # print(f'E_ij = {E_ij}')
    hat_E_ij = normalized(E_ij)
    # E_ij = e_ij
    L_ij = np.linalg.norm(E_ij, axis=1)
    # print(f'L_ij = {L_ij}')
    Varphi_ij = np.zeros_like(L_ij)

    # Edge midpoints
    mdp_ij = 0.5 * E_ij + F[0]  # = 0.5*E_ik + v_i
    mdp_ik = np.zeros_like(E_ij)
    mdp_il = np.zeros_like(E_ij)

    j_k = np.zeros_like(L_ij)  # Temporary index container for tests (usually not needed outside loop)
    j_l = np.zeros_like(L_ij)  # Temporary index container for tests (usually not needed outside loop)

    # Simplices ijk:
    # Indexed by i=0, j and k is determined by "the neighbour of `i` and `j` that is not `l` "
    Theta_i_jk = np.zeros_like(L_ij)
    Wedge_ij_ik = np.zeros_like(E_ij)
    A_ijk = np.zeros_like(L_ij)
    N_ijk = np.zeros_like(E_ij)
    N_ijl = np.zeros_like(E_ij)

    # Define midpoints
    C_ijk = np.zeros_like(A_ijk)

    # Vector curvature containers
    HNdA_ij = np.zeros_like(E_ij)  # Vector mean curvature normal sums
    HNdA_ij_Cij = np.zeros([len(E_ij), 3])  # Vector mean curvature normal sums divided by dual ara
    NdA_ij = np.zeros_like(E_ij)  # Area curvature normal sums
    NdA_ij_Cij = np.zeros_like(E_ij)  # Area curvature normal sums, weighted
    C_ij = np.zeros_like(A_ijk)  # Dual area around an edge e_ij
    C_ijk = np.zeros_like(A_ijk)  # Dual area
    C_ijl = np.zeros_like(A_ijk)  # Dual area

    i = 0

    circle_fits = []
    circle_wedge = []
    # Note, every neighbour adds  precisely one simplex:
    for j in nn[0]:
        # print(f'-')
        # print(f'j = {j}')
        # Recover indices from nn (in code recover the vertex entry in the cache)

        # print(f'nn = {nn}')
        # print(f'len(nn[j]) = {len(nn[j])}')
        # The boundary edges have only one edge
        if len(nn[j]) == 1:
            # print(f'E_ij[j] = {E_ij[j]}')
            # print(f'np.linalg.norm(E_ij[j]) = {np.linalg.norm(E_ij[j])}')
            circle_fits.append(np.linalg.norm(E_ij[j]))
            circle_wedge.append(E_ij[j])

            # Compute dual area on edge
            k = nn[j][0]  # - 1
            wedge_ij_ik = np.cross(E_ij[j], E_ij[k])
            E_jk[j] = F[k] - F[j]
            E_ik[j] = F[k] - F[i]
            if np.dot(normalized(wedge_ij_ik)[0], n_i) < 0:
                # print(f'WARNING: Wrong direction in boundary curvature')
                # k, l = l, k
                wedge_ij_ik = np.cross(E_ij[k], E_ij[j])  # Maybe?

                E_jk[j] = F[j] - F[k]
                E_ik[j] = F[k] - F[i]
                if np.dot(normalized(wedge_ij_ik)[0], n_i) < 0:
                    print(f'WARNING: STILL THE WRONG DIRECTION')

            Wedge_ij_ik[j] = wedge_ij_ik
            # vector product of the parallelogram spanned by f_i and f_j is the triangle area
            a_ijk = np.linalg.norm(wedge_ij_ik) / 2.0
            A_ijk[j] = a_ijk
            n_ijk = wedge_ij_ik / np.linalg.norm(wedge_ij_ik)
            N_ijk[j] = n_ijk

            # E_jk[j] = F[k] - F[j]
            # E_ik[j] = F[k] - F[i]
            # Solve the plane (F[0] = F[i] =current vertex i)
            mdp_ik[j] = 0.5 * E_ik[j] + F[0]
            c = np.zeros(3)
            A = np.zeros([3, 3])
            A[0] = E_ij[j]
            A[1] = E_ik[j]
            A[2] = N_ijk[j]
            c[0] = np.dot(E_ij[j], mdp_ij[j])
            c[1] = np.dot(E_ik[j], mdp_ik[j])
            c[2] = np.dot(N_ijk[j], F[0])
            v_dual = np.linalg.solve(A, c)  # v_dual in the ijk triangle?
            h_ij = np.linalg.norm(0.5 * L_ij[j])
            # = 0.5*E_ik  + F[0] ?   = F[0] -(0.5*E_ik  + F[0]) = 0.5
            b_ij = np.linalg.norm(v_dual - mdp_ij[j])  # wrong?
            c_ijk = 0.5 * b_ij * h_ij
            # C_ij[k] = c_ijk
            C_ij[j] = c_ijk

            # Try adding "half-cotan"
            if 1:
                c = L_ij[j]  # l_ij
                a = np.linalg.norm(F[k] - F[i],
                                   axis=0)  # l_ik  # Symmetric to b
                # b = np.linalg.norm(F[k] - F[l], axis=0)  # l_lk
                b = np.linalg.norm(F[k] - F[j], axis=0)  # l_lk
                alpha_ij = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))
                HNdA_ij[j] = (cotan(alpha_ij)) * (F[j] - F[i])

            continue

        k = nn[j][0]  # - 1
        l = nn[j][1]  # - 1

        # Discrete vector area:
        # Simplex areas of ijk and normals
        wedge_ij_ik = np.cross(E_ij[j], E_ij[k])
        if np.dot(normalized(wedge_ij_ik)[0], n_i) < 0:
            k, l = l, k
            wedge_ij_ik = np.cross(E_ij[j], E_ij[k])

        # Save indexes (for testing)
        j_k[j] = k
        j_l[j] = l

        Wedge_ij_ik[j] = wedge_ij_ik
        # vector product of the parallelogram spanned by f_i and f_j is the triangle area
        a_ijk = np.linalg.norm(wedge_ij_ik) / 2.0
        A_ijk[j] = a_ijk
        n_ijk = wedge_ij_ik / np.linalg.norm(wedge_ij_ik)
        N_ijk[j] = n_ijk

        # Simplex areas of ijl and normals (TODO: RECoVER FROM A MINI IJK CACHE)
        wedge_ij_il = np.cross(E_ij[j], E_ij[l])
        a_ijl = np.linalg.norm(wedge_ij_il) / 2.0
        n_ijl = -wedge_ij_il / np.linalg.norm(wedge_ij_il)  # TODO: TEST THIS
        N_ijl[j] = n_ijl

        # Dihedral angle at oriented edge ij:
        arg1 = np.dot(hat_E_ij[j], np.cross(n_ijk, n_ijl))
        arg2 = np.dot(n_ijk, n_ijl)
        varphi_ij = np.arctan2(arg1, arg2)
        Varphi_ij[j] = varphi_ij  # NOTE: Signed value!

        # Interior angles: # Law of Cosines
        c = L_ij[j]  # l_ij
        a = np.linalg.norm(F[k] - F[i], axis=0)  # l_ik  # Symmetric to b
        # b = np.linalg.norm(F[k] - F[l], axis=0)  # l_lk
        b = np.linalg.norm(F[k] - F[j], axis=0)  # l_lk
        alpha_ij = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))
        # a = np.linalg.norm(F[k] - F[i], axis=0)  # l_il  # Symmetric to b
        a = np.linalg.norm(F[l] - F[i], axis=0)  # l_
        b = np.linalg.norm(F[l] - F[j], axis=0)  # l_lj
        beta_ij = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))

        ## Curvatures
        # Vector curvatures
        HNdA_ij[j] = (cotan(alpha_ij) + cotan(beta_ij)) * (F[j] - F[i])
        # print(f'(cotan(alpha_ij) + cotan(beta_ij)) * (F[i] - F[j]) = {(cotan(alpha_ij) + cotan(beta_ij)) * (F[i] - F[j])}')
        # NdA_ij[j] = np.cross(F[i], F[j])
        # (^ NOTE: The above vertices i, j MUST be consecutive for this formula to be valid (CHECK!))

        NdA_ij[j] = np.cross(F[j], F[k])
        # NdA_ij[j] = np.cross(F[i], F[j])  #TODO: Check again
        # (^ NOTE: The above vertices j, k MUST be consecutive for this formula
        # to be valid (CHECK!))
        # Scalar component ijk
        # Interior angle
        theta_i_jk = np.arctan2(np.linalg.norm(wedge_ij_ik),
                                np.dot(E_ij[j], E_ij[k]))
        Theta_i_jk[j] = theta_i_jk

        # Areas
        if 1:
            # ijk Areas
            E_jk[j] = F[k] - F[j]
            E_ik[j] = F[k] - F[i]
            # Solve the plane
            mdp_ik[j] = 0.5 * E_ik[j] + F[0]
            c = np.zeros(3)
            A = np.zeros([3, 3])
            A[0] = E_ij[j]
            A[1] = E_ik[j]
            A[2] = N_ijk[j]
            c[0] = np.dot(E_ij[j], mdp_ij[j])
            c[1] = np.dot(E_ik[j], mdp_ik[j])
            c[2] = np.dot(N_ijk[j], F[0])
            v_dual_ijk = np.linalg.solve(A, c)  # v_dual in the ijk triangle?
            h_ij = np.linalg.norm(0.5 * L_ij[j])
            # = 0.5*E_ik  + F[0] ?   = F[0] -(0.5*E_ik  + F[0]) = 0.5
            b_ij = np.linalg.norm(v_dual_ijk - mdp_ij[j])  # wrong?
            c_ijk = 0.5 * b_ij * h_ij
            # ijl Areas
            #  k --> l
            E_jl[j] = F[l] - F[j]
            E_il[j] = F[l] - F[i]
            # Solve the plane
            mdp_il[j] = 0.5 * E_il[j] + F[0]  # is j index ok here? Think so
            c = np.zeros(3)
            A = np.zeros([3, 3])
            A[0] = E_ij[j]
            A[1] = E_il[j]
            A[2] = N_ijl[j]  # Luckily N_ijl appears to exist as suspected
            c[0] = np.dot(E_ij[j], mdp_ij[j])
            c[1] = np.dot(E_il[j], mdp_il[j])
            c[2] = np.dot(N_ijl[j], F[0])
            v_dual_ijl = np.linalg.solve(A, c)  # v_dual in the ijl triangle?
            h_ij = np.linalg.norm(0.5 * L_ij[j])  # = 0.5*E_ik  + F[0] ?   = F[0] -(0.5*E_ik  + F[0]) = 0.5
            b_ij = np.linalg.norm(v_dual_ijl - mdp_ij[j])
            c_ijl = 0.5 * b_ij * h_ij

            # Add full areas
            C_ij[j] = c_ijk + c_ijl  # the area dual to A_ij

        # Compute the Mean normal curvature integral around e_ij
        # print(f'HNdA_ij[j] = {HNdA_ij[j]}')
        # print(f'C_ij[j] = {C_ij[j]}')
        HNdA_ij_Cij[j] = HNdA_ij[j] / C_ij[j]
        NdA_ij_Cij[j] = NdA_ij[j] / C_ij[j]

        # print(f'HNdA_ij_Cij[j] = {HNdA_ij_Cij[j]}')
        # HNdA_ij_Cij[j] = np.dot(HNdA_ij[j], n_i) / (C_ij[j])
        # HNdA_ij_Cij[j] = np.sum(np.dot(HNdA_ij[j], n_i) / (C_ij[j]), axis=0)
        # HNdA_ij_Cij[j] = np.sum(HNdA_ij_Cij[j])

    # Compute angles between boundary edges
    try:
        c_wedge = np.cross(circle_wedge[0], circle_wedge[1])
        # Interior angles: # Law of Cosines
        c = np.linalg.norm(circle_wedge[0] - circle_wedge[1], axis=0)
        a = np.linalg.norm(circle_wedge[0], axis=0)  # l_ik  # Symmetric to b
        b = np.linalg.norm(circle_wedge[1], axis=0)  # l_lk
        theta_i = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))
        # print(f'theta_i = {theta_i * 180 / np.pi}')

        # print(f'circle_fits = {circle_fits}')
        # TODO: THIS IS WRONG!
        # In 2D we have:
        # (x - x_c)**2 + (y - y_c)**2 = r**2
        # (x_1 - x_c)**2 + (y_1 - y_c)**2 = r**2
        # (x_2 - x_c)**2 + (y_2 - y_c)**2 = r**2
        # (x_3 - x_c)**2 + (y_3 - y_c)**2 = r**2
        # Subtract the first from the second, and the first from the third to
        # create two linear equations
        # (x_1 - x_c)**2 - (x_2 - x_c)**2  + (y_1 - y_c)**2 - (y_2 - y_c)**2  = 0

        # (x_3 - x_c) ** 2 + (y_3 - y_c) ** 2 = r ** 2
        r_est = np.linalg.norm(circle_fits)
        r_est = np.sqrt(circle_fits[0] ** 2 + circle_fits[1] ** 2)
        # print(f'r_est = {r_est}')

        # Arc length
        ds = theta_i * r_est
    except IndexError:  # Not a boundary
        theta_i = 0.0
        ds = 0.0

    # Normals
    N_ijk = np.array(N_ijk)
    N_ijl = np.array(N_ijl)
    Wedge_ij_ik = np.array(Wedge_ij_ik)
    # (^ The integrated area of the unit sphere)

    # Integrated curvatures
    HNdA_i = 0.5 * np.sum(HNdA_ij, axis=0)  # Vector mean curvature normal sums (multiplied by N?)
    # HN_i = 0.5 * np.sum(HN_ij, axis=0)
    NdA_i = (1 / 6.0) * np.sum(NdA_ij, axis=0)  # Vector normal  are sums
    # NdA_ij_Cij = np.sum(NdA_ij_Cij, axis=0)
    # (^ The integrated area of the original smooth surface (a Dual discrete differential 2-form))

    # Point-wise estimates
    HN_i = np.sum(HNdA_i) / np.sum(C_ij)
    # TODO: Need to replace with dot n_i

    HN_i = np.sum(np.dot(HNdA_i, n_i)) / np.sum(C_ij)
    # TURN THIS OFF IN NORMAL RUNS:
    if 0:
        HN_i = np.sum(np.dot(HNdA_i, normalized(np.sum(NdA_ij_Cij, axis=0))[0])) / np.sum(C_ij)
    if 0:
        HNdA_i = np.sum(HNdA_ij_Cij, axis=0)
        HN_i = np.sum(
            np.dot(HNdA_i, normalized(np.sum(NdA_ij_Cij, axis=0))[0])) / np.sum(
            C_ij)
    if 1:
        # print(f'C_ij = {C_ij}')
        C_i = np.sum(C_ij, axis=0)

    K_H_i = (HN_i / 2.0) ** 2

    # nt development:
    if 0:
        print('-')
        # print(f'HNdA_i= {HNdA_i}')

        print(f'n_i = {n_i}')
        print(f'C_i = {C_i}')
        #   print(f'normalized(C_i) = {normalized(C_i)[0]}')
        nt = []
        for nda, a_ijk in zip(NdA_ij, A_ijk):
            # print(f'nda = {nda}')
            # print(f'a_ijk = {a_ijk}')
            nt.append(nda / a_ijk)

        nt = np.array(nt)
        nt = np.nan_to_num(nt)
        print(f'nt = {nt}')
        nt = np.sum(nt, axis=0)
        print(f'nt = {nt}')
        print(f'normalized(np.sum(NdA_ij/A_ijk, axis=0)) = {normalized(nt)[0]}')
        print(f'normalized(NdA_i ) = {normalized(NdA_i)[0]}')
        print(f'normalized(np.sum(NdA_ij_Cij, axis=0)) '
              f'= {normalized(np.sum(NdA_ij_Cij, axis=0))[0]}')
        print(' ')
        print(f'NdA_i = {NdA_i}')
        print(f'NdA_ij_Cij = {NdA_ij_Cij}')
        # print(f'normalized(HNdA_i) = {normalized(HNdA_i)[0]}')
        # print(f'normalized(HNdA_ij_Cij) = {normalized(np.sum(HNdA_ij_Cij, axis=0))[0]}')
        # print(f'normalized(NdA_i ) = {normalized(NdA_i)[0]}')
        # print(f'HNdA_ij_Cij = {HNdA_ij_Cij}')
    HNdA_ij_Cij = np.sum(np.sum(HNdA_ij_Cij, axis=0))

    return dict(**locals())


