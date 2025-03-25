import numpy as np
#import scipy
from ._complex import Complex
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from ._gauss_bonnet import *

import decimal


# Colour scheme for surfaces
db = np.array([129, 160, 189]) / 255  # Dark blue
lb = np.array([176, 206, 234]) / 255  # Light blue

do = np.array([235, 129, 27]) / 255  # Dark orange
lo = np.array([242, 189, 138]) / 255  # Light orange


def geo_mean(iterable):
    a = np.log(iterable)
    return np.exp(a.mean())

def cotan(theta):
    return 1 / np.tan(theta)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

# Plot the surface compolex
def plot_surface(F, nn):

    # Colour scheme for surfaces
    db = np.array([129, 160, 189]) / 255  # Dark blue
    lb = np.array([176, 206, 234]) / 255  # Light blue

    HC = Complex(3)
    V = []
    for f in F:
        V.append(HC.V[tuple(f)])
    for i, i_nn in enumerate(nn):
        for i_v2 in i_nn:
            V[i].connect(V[i_v2])

    # Plot complex plot_surface

    fig, axes, fig_s, axes_s = HC.plot_complex(point_color=db,
                                               line_color=db,
                                               complex_color_f=lb,
                                               complex_color_e=db
                                               )
    return fig, axes, HC


# Plot a list of normal vectors
def plot_n(N_ijk, axes, v0=None, color="tab:red", mutation_scale=20):
    if v0 is None:
        v0 = [0, 0, 0]
    for n_ijk in N_ijk:
        a = Arrow3D([v0[0], v0[0] + n_ijk[0]], [v0[1], v0[1] + n_ijk[1]],
                    [v0[2], v0[2] + n_ijk[2]], mutation_scale=mutation_scale,
                    lw=3, arrowstyle="-|>", color=color, alpha=0.5)
        axes.add_artist(a)
    return axes


# Plot a plane
def plot_plane(n, p0, axes, dim=0.5, color='tab:red', alpha=.5):
    # n is the normal vector
    # p0 is a point on the plane
    # dim scales the plan area over the X, Y axis
    X, Y = np.meshgrid([-dim, dim], [-dim, dim])
    Z = X + Y  # np.zeros((2, 2))
    Z = (X * n[0] + Y * n[1]) / n[2]  # np.zeros((2, 2))
    Z = -(n[0] * (X - p0[0]) + n[1] * (Y - p0[1])) / n[2] + p0[2]  # np.zeros((2, 2))
    axes.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, zorder=1)
    return axes


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


#def complex_curvatures

def vectorise_vnn(v):
    F = [v.x_a]
    nn = []
    nn.append([])
    Complex_nn = []
    Complex_nn.append([])
    ind = 0
    F_set = set()  # Track unexplained overflow
    for v2 in v.nn:
        ind += 1
        # print(v2.x_a)
        nn.append([])
        Complex_nn.append([])
       # print(f'v2.x in F = {v2.x in F_set}')
       # print(f'F_set = {F_set}')
        F.append(v2.x_a)
        F_set.add(v2.x)
        nn[0].append(ind)
        Complex_nn[0].append(v2.index)
        for v3 in v2.nn:
            if v3 in v.nn:
                Complex_nn[ind].append(v3.index)

    mapping = {}
    for ind, cind in enumerate(Complex_nn[0]):
        mapping[cind] = ind + 1

    for nn_ind, cnn_ind in zip(nn[1:], Complex_nn[1:]):
        for cind in cnn_ind:
            nn_ind.append(mapping[cind])

    F = np.array(F)
    return F, nn

def b_vectorise_vnn(v):
    F = [v.x_a]
    nn = []
    nn.append([])
    Complex_nn = []
    Complex_nn.append([])
    ind = 0
    for v2 in v.nn:
        ind += 1
        # print(v2.x_a)
        nn.append([])
        Complex_nn.append([])
        F.append(v2.x_a)
        nn[0].append(ind)
        Complex_nn[0].append(v2.index)
        for v3 in v2.nn:
            if v3 in v.nn:
                Complex_nn[ind].append(v3.index)

    mapping = {}
    for ind, cind in enumerate(Complex_nn[0]):
        mapping[cind] = ind + 1

    for nn_ind, cnn_ind in zip(nn[1:], Complex_nn[1:]):
        for cind in cnn_ind:
            nn_ind.append(mapping[cind])
    print(f'nn = {nn}')
    b_F = []
    b_nn = []
    b_ind = []
    for ind, nn_i in enumerate(nn):
        print(f'nn_i = {nn_i}')
        print(f'len(nn_i) = {len(nn_i)}')
        if len(nn_i) > 1:
            continue
        else:
            b_ind.append(ind)
            b_F.append(F[ind])
            b_nn.append(nn[ind])

    # b_ind = sorted(b_ind)
    for ind in sorted(b_ind, reverse=True):
        del nn[ind]
        del F[ind]

    F = np.array(F)
    b_F = np.array(b_F)
    return F, nn, b_F, b_nn

# Curvature computations
# Interior curvatures
def HC_curvatures_sessile(HC, bV, r, theta_p, printout=False):
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
        #TODO: REMOVE UNDER NORMAL CONDITIONS:
        if 0:
            if v in bV:
                continue
        N_f0 = v.x_a - np.array([0.0, 0.0, -R * np.sin(theta_p)])  # First approximation
        #N_f0 = v.x_a - np.array([0.0, 0.0, R-R * np.sin(theta_p)])  # First approximation
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
        #TODO: REMOVE UNDER NORMAL CONDITIONS:
        if 0:
            if v in bV:
                continue
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


def int_curvatures(HC, bV, r, theta_p, printout=False):
    """
    TODO: This is from _capillary_rise.py and the normals need to be updated!

    :param HC:
    :param bV:
    :param r:
    :param theta_p:
    :param printout:
    :return:
    """
    R = r / np.cos(theta_p)
    K_f = (1 / R) ** 2
    H_f = 2 / R  # 2 / R
    HNdA_ij = []
    HNdA_i = []
    HNdA_ij_sum = []
    HNdA_ij_dot = []
    C_ijk = []
    A_ijk = []
    N_i = []
    int_V = []
    HNda_v_cache = {}
    C_ijk_v_cache = {}
    K_H_cache = {}
    HNdA_i_Cij = []
    for v in HC.V:
        if v in bV:
            continue
        else:
            N_f0 = np.array([0.0, 0.0, R * np.sin(theta_p)]) - v.x_a # First approximation
            N_f0 = normalized(N_f0)[0]
            N_i.append(N_f0)
            F, nn = vectorise_vnn(v)
            # Compute discrete curvatures
            # c_outd = curvatures(F, nn, n_i=N_f0)
            c_outd = curvatures_hn_i(F, nn, n_i=N_f0)
            #c_outd = curvatures_hn_ij_c_ij(F, nn, n_i=N_f0)
            HNda_v_cache[v.x] = c_outd['HNdA_ij']
            HNdA_i.append(c_outd['HNdA_i'])
            HNdA_ij.append(c_outd['HNdA_ij'])
            HNdA_ij_sum.append(np.sum(c_outd['HNdA_ij']))
            HNdA_ij_dot.append(np.sum(np.dot(c_outd['HNdA_ij'], c_outd['n_i'])))

            HNdA_i_Cij.append(c_outd['HNdA_ij_Cij'])
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
            h_disc = (1 / 2.0) * np.sum(np.dot(c_outd['HNdA_ij'], c_outd['n_i'])) / np.sum(c_outd['C_ijk'])
            #h_disc = (1 / 2.0) * np.sum(np.dot(c_outd['HNdA_ij'], N_f0)) / np.sum(c_outd['C_ijk'])
            K_H_cache[v.x] = (h_disc / 2.0) ** 2

    H_disc = (1 / 2.0) * np.array(HNdA_ij_dot) / C_ijk
    K_H = (H_disc / 2.0)**2

    # Adjust HNdA_ij_sum and HNdA_ij_dot
    HNdA_ij_sum = 0.5 * np.array(HNdA_ij_sum) / C_ijk
    HNdA_ij_dot = 0.5 * np.array(HNdA_ij_dot) / C_ijk

    # New normalized dot product odeas
    HNdA_ij_dot_hnda_i = []
    K_H_2 = []
    HN_i = []
    for hnda_ij, c_ijk, n_i in zip(HNdA_ij, C_ijk, N_i):
        if 0:
            hnda_i = np.sum(hnda_ij, axis=0)
            # print(f'hnda_i = {hnda_i}')
            n_hnda_i = normalized(hnda_i)[0]
            hndA_ij_dot_hnda_i = 0.5 * np.sum(np.dot(hnda_ij, n_hnda_i)) / c_ijk
            HNdA_ij_dot_hnda_i.append(hndA_ij_dot_hnda_i)
        else: # Appears to be more accurate, sadly
            #print(f'hnda_ij, n_i = {hnda_ij, n_i}')
            hndA_ij_dot_hnda_i = 0.5 * np.sum(np.dot(hnda_ij, n_i)) / c_ijk
            HNdA_ij_dot_hnda_i.append(hndA_ij_dot_hnda_i)

        k_H_2 = (hndA_ij_dot_hnda_i/2.0) ** 2
        K_H_2.append(k_H_2)

        HN_i.append(0.5 * np.sum(np.dot(hnda_ij, n_i)) / c_ijk)
        hnda_i_sum = 0.0
        for hnda_ij_sum in HNdA_ij_sum:
            hnda_i_sum += hnda_ij_sum

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

    return HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i, HNdA_ij_dot_hnda_i, K_H_2, HNdA_i_Cij

def curvatures_hn_ij_c_ij(F, nn, n_i=None):
    """
    F: Array of vectors forming the star domain of v_i = F[0]
    nn: Connections within each vertex in union with the star domain
    n_i: Normal vector at vertex v_i (approx.)
    """
    # NOTE: We need a better solution to ensure signed quantities retain their structure.
    #      The mesh must be ordered to ensure we obtain the correct normal face directions
    if n_i is None:
        n_i = normalized(F[0])[0]
    #print(f'n_i = {n_i}')
    # TODO: Later we can cache these containers to avoid extra computations
    # Edges from i to j:
    E_ij = F[1:] - F[0]
    E_ij = np.vstack([np.zeros(3), E_ij])
    E_jk = np.zeros_like(E_ij)
    E_ik = np.zeros_like(E_ij)

    E_jl = np.zeros_like(E_ij)
    E_il = np.zeros_like(E_ij)
    #print(f'E_ij = {E_ij}')
    hat_E_ij = normalized(E_ij)
    # E_ij = e_ij
    L_ij = np.linalg.norm(E_ij, axis=1)
    #print(f'L_ij = {L_ij}')
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
    C_ijk = np.zeros_like(A_ijk)  # Dual area
    C_ijl = np.zeros_like(A_ijk)  # Dual area

    i = 0
    # Note, every neighbour adds  precisely one simplex:
    for j in nn[0]:
        #print(f'-')
        #print(f'j = {j}')
        # Recover indices from nn (in code recover the vertex entry in the cache)
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
        #b = np.linalg.norm(F[k] - F[l], axis=0)  # l_lk
        b = np.linalg.norm(F[k] - F[j], axis=0)  # l_lk
        alpha_ij = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))
        #a = np.linalg.norm(F[k] - F[i], axis=0)  # l_il  # Symmetric to b
        a = np.linalg.norm(F[l] - F[i], axis=0)  # l_
        b = np.linalg.norm(F[l] - F[j], axis=0)  # l_lj
        beta_ij = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))

        ## Curvatures
        # Vector curvatures
        HNdA_ij[j] = (cotan(alpha_ij) + cotan(beta_ij)) * (F[j] - F[i])
        #print(f'(cotan(alpha_ij) + cotan(beta_ij)) * (F[i] - F[j]) = {(cotan(alpha_ij) + cotan(beta_ij)) * (F[i] - F[j])}')
        # NdA_ij[j] = np.cross(F[i], F[j])
        # (^ NOTE: The above vertices i, j MUST be consecutive for this formula to be valid (CHECK!))

        NdA_ij[j] = np.cross(F[j], F[k])
        # (^ NOTE: The above vertices j, k MUST be consecutive for this formula to be valid (CHECK!))
        # Scalar component ijk
        # Interior angle
        theta_i_jk = np.arctan2(np.linalg.norm(wedge_ij_ik), np.dot(E_ij[j], E_ij[k]))
        Theta_i_jk[j] = theta_i_jk

        # Areas
        # ijk Areas
        if 1:
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
            v_dual = np.linalg.solve(A, c)  # v_dual in the ijk triangle?
            h_ij = np.linalg.norm(0.5 * L_ij[j])  # = 0.5*E_ik  + F[0] ?   = F[0] -(0.5*E_ik  + F[0]) = 0.5
            b_ij = np.linalg.norm(v_dual - mdp_ij[j])  # wrong?
            C_ij = 0.5 * b_ij * h_ij
            C_ij_k = C_ij
            # h_ik = np.linalg.norm(0.5*L_ij[int(j_k[j])])  # = 0.5*E_ik  + F[0] ?   = F[0] -(0.5*E_ik  + F[0]) = 0.5
            h_ik = np.linalg.norm(0.5 * L_ij[k])
            b_ik = np.linalg.norm(v_dual - mdp_ik[j])
            C_ik = 0.5 * b_ik * h_ik
            C_ijk[j] = C_ij + C_ik  # the area dual to A_ijk (validated at 90deg/0 curvature)

            #print(f'C_ij = {C_ij}')
            #print(f'C_ik = {C_ik}')
        # ijl Areas
        if 1:
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
            v_dual = np.linalg.solve(A, c)  # v_dual in the ijl triangle?
            h_ij = np.linalg.norm(0.5 * L_ij[j])  # = 0.5*E_ik  + F[0] ?   = F[0] -(0.5*E_ik  + F[0]) = 0.5
            b_ij = np.linalg.norm(v_dual - mdp_ij[j])  # wrong?
            C_ij = 0.5 * b_ij * h_ij
            C_ij_l = C_ij
            # h_ik = np.linalg.norm(0.5*L_ij[int(j_k[j])])  # = 0.5*E_ik  + F[0] ?   = F[0] -(0.5*E_ik  + F[0]) = 0.5
            h_il = np.linalg.norm(0.5 * L_ij[l])
            b_il = np.linalg.norm(v_dual - mdp_il[j])
            C_il = 0.5 * b_il * h_il
            C_ijl[j] = C_ij + C_il  # the area dual to A_ijk (validated at 90deg/0 curvature)

            #print(f'-')
            #print(f'C_ij = {C_ij}')
            #print(f'C_il = {C_il}')

        if 0:
            HNdA_ij_Cij[j] = HNdA_ij[j] / (np.array(C_ij, dtype=np.longdouble)
                                           + np.array(C_ik, dtype=np.longdouble))

        if 0:
            HNdA_ij_Cij[j] = HNdA_ij[j] / (C_ij_l + C_ij_k)

        if 1:
            HNdA_ij_Cij[j] = HNdA_ij[j] / (C_ijk[j] + C_ijl[j])

        #NOTE: THIS WAS ON IN LAST FORMULATION:
        if 0:
            HNdA_ij_Cij[j] = HNdA_ij[j] * (C_ijk[j] + C_ijl[j])
        #print(f'HNdA_ij_Cij = {HNdA_ij_Cij}')
        #print(f'np.sum(HNdA_ij_Cij) = {np.sum(HNdA_ij_Cij)}')
    pass
    N_ijk = np.array(N_ijk)
    N_ijl = np.array(N_ijl)
    Wedge_ij_ik = np.array(Wedge_ij_ik)
    # (^ The integrated area of the unit sphere)
    HNdA_i = 0.5 * np.sum(HNdA_ij, axis=0)  # Vector mean curvature normal sums (multiplied by N?)
    # HN_i = 0.5 * np.sum(HN_ij, axis=0)
    NdA_i = (1 / 6.0) * np.sum(NdA_ij, axis=0)  # Vector normal  are sums
    # (^ The integrated area of the original smooth surface (a Dual discrete differential 2-form))
    # KN_i = 0.5 * np.sum(KN_ij, axis=0)  # Vector Gauss curvature normal sums ????

    return dict(**locals())

def curvatures_hn_i(F, nn, n_i=None):
    """
    F: Array of vectors forming the star domain of v_i = F[0]
    nn: Connections within each vertex in union with the star domain
    n_i: Normal vector at vertex v_i (approx.)
    """
    # NOTE: We need a better solution to ensure signed quantities retain their structure.
    #      The mesh must be ordered to ensure we obtain the correct normal face directions
    if n_i is None:
        n_i = normalized(F[0])[0]
    #print(f'n_i = {n_i}')
    # TODO: Later we can cache these containers to avoid extra computations
    # Edges from i to j:
    E_ij = F[1:] - F[0]
    E_ij = np.vstack([np.zeros(3), E_ij])
    E_jk = np.zeros_like(E_ij)
    E_ik = np.zeros_like(E_ij)
    #print(f'E_ij = {E_ij}')
    hat_E_ij = normalized(E_ij)
    # E_ij = e_ij
    L_ij = np.linalg.norm(E_ij, axis=1)
    #print(f'L_ij = {L_ij}')
    Varphi_ij = np.zeros_like(L_ij)

    # Edge midpoints
    mdp_ij = 0.5 * E_ij + F[0]  # = 0.5*E_ik + v_i
    mdp_ik = np.zeros_like(E_ij)

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
    C_ijk = np.zeros_like(A_ijk)  # Dual area

    i = 0
    # Note, every neighbour adds  precisely one simplex:
    for j in nn[0]:
        # Recover indices from nn (in code recover the vertex entry in the cache)
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
        #b = np.linalg.norm(F[k] - F[l], axis=0)  # l_lk
        b = np.linalg.norm(F[k] - F[j], axis=0)  # l_lk
        alpha_ij = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))
        #a = np.linalg.norm(F[k] - F[i], axis=0)  # l_il  # Symmetric to b
        a = np.linalg.norm(F[l] - F[i], axis=0)  # l_
        b = np.linalg.norm(F[l] - F[j], axis=0)  # l_lj
        beta_ij = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))

        ## Curvatures
        # Vector curvatures
        HNdA_ij[j] = (cotan(alpha_ij) + cotan(beta_ij)) * (F[j] - F[i])
        #print(f'(cotan(alpha_ij) + cotan(beta_ij)) * (F[i] - F[j]) = {(cotan(alpha_ij) + cotan(beta_ij)) * (F[i] - F[j])}')
        # NdA_ij[j] = np.cross(F[i], F[j])
        # (^ NOTE: The above vertices i, j MUST be consecutive for this formula to be valid (CHECK!))

        NdA_ij[j] = np.cross(F[j], F[k])
        # (^ NOTE: The above vertices j, k MUST be consecutive for this formula to be valid (CHECK!))
        # Scalar component ijk
        # Interior angle
        theta_i_jk = np.arctan2(np.linalg.norm(wedge_ij_ik), np.dot(E_ij[j], E_ij[k]))
        Theta_i_jk[j] = theta_i_jk

        # Areas
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
        v_dual = np.linalg.solve(A, c)
        h_ij = np.linalg.norm(0.5 * L_ij[j])  # = 0.5*E_ik  + F[0] ?   = F[0] -(0.5*E_ik  + F[0]) = 0.5
        b_ij = np.linalg.norm(v_dual - mdp_ij[j])  # wrong
        C_ij = 0.5 * b_ij * h_ij
        # h_ik = np.linalg.norm(0.5*L_ij[int(j_k[j])])  # = 0.5*E_ik  + F[0] ?   = F[0] -(0.5*E_ik  + F[0]) = 0.5
        h_ik = np.linalg.norm(0.5 * L_ij[k])
        b_ik = np.linalg.norm(v_dual - mdp_ik[j])
        C_ik = 0.5 * b_ik * h_ik
        C_ijk[j] = C_ij + C_ik  # the area dual to A_ijk (validated at 90deg/0 curvature)

       # for qind, hn in enumerate(HNdA_ij[j]):
        #    print(f'hn = {hn}')
         #   print(f' qind = { qind}')
            #HNdA_ij_Cij[j, qind] = decimal.Decimal(hn) / (decimal.Decimal(C_ij) + decimal.Decimal(C_ik))
       #     HNdA_ij_Cij[j, qind] = decimal.Decimal(hn) / (decimal.Decimal(C_ij) + decimal.Decimal(C_ik))
        HNdA_ij_Cij[j] = HNdA_ij[j] / (np.array(C_ij, dtype=np.longdouble)
                                       + np.array(C_ik, dtype=np.longdouble))
    pass
    N_ijk = np.array(N_ijk)
    N_ijl = np.array(N_ijl)
    Wedge_ij_ik = np.array(Wedge_ij_ik)
    # (^ The integrated area of the unit sphere)
    HNdA_i = 0.5 * np.sum(HNdA_ij, axis=0)  # Vector mean curvature normal sums (multiplied by N?)
    # HN_i = 0.5 * np.sum(HN_ij, axis=0)
    NdA_i = (1 / 6.0) * np.sum(NdA_ij, axis=0)  # Vector normal  are sums
    # (^ The integrated area of the original smooth surface (a Dual discrete differential 2-form))
    # KN_i = 0.5 * np.sum(KN_ij, axis=0)  # Vector Gauss curvature normal sums ????

    return dict(**locals())

def curvatures(F, nn, n_i=None):
    """
    F: Array of vectors forming the star domain of v_i = F[0]
    nn: Connections within each vertex in union with the star domain
    n_i: Normal vector at vertex v_i
    """
    # NOTE: We need a better solution to ensure signed quantities retain their structure.
    #      The mesh must be ordered to ensure we obtain the correct normal face directions
    if n_i is None:
        n_i = normalized(F[0])[0]
    #print(f'n_i = {n_i}')
    # TODO: Later we can cache these containers to avoid extra computations
    # Edges from i to j:
    E_ij = F[1:] - F[0]
    E_ij = np.vstack([np.zeros(3), E_ij])
    E_jk = np.zeros_like(E_ij)
    E_ik = np.zeros_like(E_ij)
    #print(f'E_ij = {E_ij}')
    hat_E_ij = normalized(E_ij)
    # E_ij = e_ij
    L_ij = np.linalg.norm(E_ij, axis=1)
    #print(f'L_ij = {L_ij}')
    Varphi_ij = np.zeros_like(L_ij)

    # Edge midpoints
    mdp_ij = 0.5 * E_ij + F[0]  # = 0.5*E_ik + v_i
    mdp_ik = np.zeros_like(E_ij)

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
    KNdA_ij = np.zeros_like(E_ij)  # Vector Gauss curvature normal sums
    HNdA_ij = np.zeros_like(E_ij)  # Vector mean curvature normal sums
    NdA_ij = np.zeros_like(E_ij)  # Area curvature normal sums

    # Scalar curvature containers
    H_ij = np.zeros_like(L_ij)  # Edge mean curvatures
    A_ijk = A_ijk  # Area
    C_ijk = np.zeros_like(A_ijk)  # Dual area
    V_ijk = np.zeros_like(A_ijk)  # Volume
    # A_ij = np.zeros_like(L_ij)   # Vector area ??

    i = 0
    # Note, every neighbour adds  precisely one simplex:
    for j in nn[0]:
        # Recover indices from nn (in code recover the vertex entry in the cache)
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
        #b = np.linalg.norm(F[k] - F[l], axis=0)  # l_lk
        b = np.linalg.norm(F[k] - F[j], axis=0)  # l_lk
        alpha_ij = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))
        #a = np.linalg.norm(F[k] - F[i], axis=0)  # l_il  # Symmetric to b
        a = np.linalg.norm(F[l] - F[i], axis=0)  # l_
        b = np.linalg.norm(F[l] - F[j], axis=0)  # l_lj
        beta_ij = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))

        ## Curvatures
        # Vector curvatures
        # Guassian normal curvature
        KNdA_ij[j] = (varphi_ij) / (L_ij[j]) * (F[j] - F[i])
        # Mean normal curvatre
        # Ratio of dual/primal length is given by cotan formula, yielding
        # HNdA_ij[j] = (cotan(alpha_ij) + cotan(beta_ij)) * (F[i] - F[j])
        HNdA_ij[j] = (cotan(alpha_ij) + cotan(beta_ij)) * (F[j] - F[i])
        #print(f'(cotan(alpha_ij) + cotan(beta_ij)) * (F[i] - F[j]) = {(cotan(alpha_ij) + cotan(beta_ij)) * (F[i] - F[j])}')
        # NdA_ij[j] = np.cross(F[i], F[j])
        # (^ NOTE: The above vertices i, j MUST be consecutive for this formula to be valid (CHECK!))
        NdA_ij[j] = np.cross(F[j], F[k])
        # (^ NOTE: The above vertices j, k MUST be consecutive for this formula to be valid (CHECK!))
        # Scalar component ijk
        # Interior angle
        theta_i_jk = np.arctan2(np.linalg.norm(wedge_ij_ik), np.dot(E_ij[j], E_ij[k]))
        Theta_i_jk[j] = theta_i_jk
        # Mean normal
        H_ij[j] = L_ij[j] * varphi_ij

        # Areas
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
        v_dual = np.linalg.solve(A, c)
        h_ij = np.linalg.norm(0.5 * L_ij[j])  # = 0.5*E_ik  + F[0] ?   = F[0] -(0.5*E_ik  + F[0]) = 0.5
        b_ij = np.linalg.norm(v_dual - mdp_ij[j])  # wrong
        C_ij = 0.5 * b_ij * h_ij
        # h_ik = np.linalg.norm(0.5*L_ij[int(j_k[j])])  # = 0.5*E_ik  + F[0] ?   = F[0] -(0.5*E_ik  + F[0]) = 0.5
        h_ik = np.linalg.norm(0.5 * L_ij[k])
        b_ik = np.linalg.norm(v_dual - mdp_ik[j])
        C_ik = 0.5 * b_ik * h_ik
        C_ijk[j] = C_ij + C_ik  # the area dual to A_ijk (validated at 90deg/0 curvature)

    pass
    N_ijk = np.array(N_ijk)
    N_ijl = np.array(N_ijl)
    Wedge_ij_ik = np.array(Wedge_ij_ik)
    KNdA_i = 0.5 * np.sum(KNdA_ij, axis=0)  # Vector Gauss curvature normal sums
    # (^ The integrated area of the unit sphere)
    HNdA_i = 0.5 * np.sum(HNdA_ij, axis=0)  # Vector mean curvature normal sums (multiplied by N?)
    # HN_i = 0.5 * np.sum(HN_ij, axis=0)
    NdA_i = (1 / 6.0) * np.sum(NdA_ij, axis=0)  # Vector normal  are sums
    # (^ The integrated area of the original smooth surface (a Dual discrete differential 2-form))
    # KN_i = 0.5 * np.sum(KN_ij, axis=0)  # Vector Gauss curvature normal sums ????

    ## Scalar curvatures of vertex i:
    # Gaussian curvature (angle defct)
    Omega_i = 2 * np.pi - np.sum(Theta_i_jk)
    K = Omega_i
    # Mean curvature
    H_i = (1 / 4.0) * np.sum(H_ij)
    H_ij_sum = (1 / 2.0) * np.sum(H_ij)

    # Area curvature:

    # Volume curvature

    ## Principle curvatures
    # Smooth approximated curvatures:
    # krms np.sqrt(H_i**2 - K)
    # kappa_1 = H - krms
    # kappa_2 = H - krms
    # A_i  # A_i := abs(C_i) the area of the dual cell
    return dict(**locals())

# Boundary curvatures
def b_curvatures(HC, bV, r, theta_p, printout=False):
    R = r / np.cos(theta_p)
    K_f = (1 / R) ** 2
    H_f = 2 / R  # 2 / R
    HNdA_ij = []
    HNdA_i = []
    HNdA_ij_sum = []
    HNdA_ij_dot = []
    C_ijk = []
    A_ijk = []
    N_i = []
    int_V = []
    HNda_v_cache = {}
    C_ijk_v_cache = {}
    K_H_cache = {}
    HNdA_i_Cij = []
    Theta_i = []
    for v in HC.V:
        if v in bV:
            N_f0 = np.array(
                [0.0, 0.0, R * np.sin(theta_p)]) - v.x_a  # First approximation
            N_f0 = normalized(N_f0)[0]
            N_i.append(N_f0)
            F, nn = vectorise_vnn(v)
            # Compute discrete curvatures
            # c_outd = curvatures(F, nn, n_i=N_f0)
            c_outd = b_curvatures_hn_ij_c_ij(F, nn, n_i=N_f0)

            # c_outd = curvatures_hn_ij_c_ij(F, nn, n_i=N_f0)
            HNda_v_cache[v.x] = c_outd['HNdA_ij']
            HNdA_i.append(c_outd['HNdA_i'])
            HNdA_ij.append(c_outd['HNdA_ij'])
            #HNdA_ij_sum.append(-np.sum(np.dot(c_outd['HNdA_i'], c_outd['n_i'])))
            HNdA_ij_sum.append(np.sum(c_outd['HNdA_ij']))
            HNdA_ij_dot.append(np.sum(np.dot(c_outd['HNdA_ij'], c_outd['n_i'])))

            HNdA_i_Cij.append(c_outd['HNdA_ij_Cij'])
            C_ijk.append(np.sum(c_outd['C_ijk']))
            C_ijk_v_cache[v.x] = np.sum(c_outd['C_ijk'])
            A_ijk.append(np.sum(c_outd['A_ijk']))
            int_V.append(v)
            Theta_i.append(c_outd['theta_i'])
            # New
            h_disc = (1 / 2.0) * np.sum(
                np.dot(c_outd['HNdA_ij'], c_outd['n_i'])) / np.sum(
                c_outd['C_ijk'])
            # h_disc = (1 / 2.0) * np.sum(np.dot(c_outd['HNdA_ij'], N_f0)) / np.sum(c_outd['C_ijk'])
            K_H_cache[v.x] = (h_disc / 2.0) ** 2
        else:
            continue

    H_disc = (1 / 2.0) * np.array(HNdA_ij_dot) / C_ijk
    K_H = (H_disc / 2.0)**2
    # Adjust HNdA_ij_sum and HNdA_ij_dot
    #HNdA_ij_sum = 0.5 * np.array(HNdA_ij_sum) / C_ijk
    HNdA_ij_sum = np.array(HNdA_ij_sum) / C_ijk
    HNdA_ij_dot = 0.5 * np.array(HNdA_ij_dot) / C_ijk

    # New normalized dot product odeas
    HNdA_ij_dot_hnda_i = []
    K_H_2 = []
    HN_i = []
    for hnda_ij, c_ijk, n_i in zip(HNdA_ij, C_ijk, N_i):
        if 0:
            hnda_i = np.sum(hnda_ij, axis=0)
            # print(f'hnda_i = {hnda_i}')
            n_hnda_i = normalized(hnda_i)[0]
            hndA_ij_dot_hnda_i = 0.5 * np.sum(np.dot(hnda_ij, n_hnda_i)) / c_ijk
            HNdA_ij_dot_hnda_i.append(hndA_ij_dot_hnda_i)
        
        # COMMENTED OUT 2021-06-22:
        elif 0: # Appears to be more accurate, sadly
            #hndA_ij_dot_hnda_i = 0.5 * np.sum(np.dot(hnda_ij, n_i)) / c_ijk
            #TODO: CHECK THIS:
            hndA_ij_dot_hnda_i = -HNdA_ij_sum
            HNdA_ij_dot_hnda_i.append(hndA_ij_dot_hnda_i)

        elif 0:  # Appears to be more accurate, sadly
            hndA_ij_dot_hnda_i = 0.5 * np.sum(np.dot(hnda_ij, n_i)) / c_ijk

        # Prev. converging working, changed on 2021-06-22:
        elif 0:
            hndA_ij_dot_hnda_i = 0.5 * np.sum(hnda_ij) / c_ijk
        elif 1:  # Appears to be more accurate, sadly
            hndA_ij_dot_hnda_i = 0.5 * np.sum(np.dot(hnda_ij, n_i)) / (c_ijk)
            HNdA_ij_dot_hnda_i.append(hndA_ij_dot_hnda_i)
        elif 0:
            hndA_ij_dot_hnda_i = -np.sum(np.dot(hnda_ij, n_i)) / c_ijk
        k_H_2 = (hndA_ij_dot_hnda_i/2.0) ** 2
        K_H_2.append(k_H_2)

        HN_i.append(0.5 * np.sum(np.dot(hnda_ij, n_i)) / c_ijk)
        hnda_i_sum = 0.0
        for hnda_ij_sum in HNdA_ij_sum:
            hnda_i_sum += hnda_ij_sum

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


    return (HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i,
            HNdA_ij_dot_hnda_i, K_H_2, HNdA_i_Cij, Theta_i)

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
    #print(f'.')
    #print(f'n_i = {n_i}')
    #print(f'F = {F}')
    #print(f'nn = {nn}')
    if n_i is None:
        n_i = normalized(F[0])[0]
    #print(f'n_i = {n_i}')
    # TODO: Later we can cache these containers to avoid extra computations
    # Edges from i to j:
    E_ij = F[1:] - F[0]
    E_ij = np.vstack([np.zeros(3), E_ij])
    E_jk = np.zeros_like(E_ij)
    E_ik = np.zeros_like(E_ij)

    E_jl = np.zeros_like(E_ij)
    E_il = np.zeros_like(E_ij)
    #print(f'E_ij = {E_ij}')
    hat_E_ij = normalized(E_ij)
    # E_ij = e_ij
    L_ij = np.linalg.norm(E_ij, axis=1)
    #print(f'L_ij = {L_ij}')
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
        #print(f'-')
        #print(f'j = {j}')
        # Recover indices from nn (in code recover the vertex entry in the cache)

        #print(f'nn = {nn}')
        #print(f'len(nn[j]) = {len(nn[j])}')
        # The boundary edges have only one edge
        if len(nn[j]) == 1:
            #print(f'E_ij[j] = {E_ij[j]}')
            #print(f'np.linalg.norm(E_ij[j]) = {np.linalg.norm(E_ij[j])}')
            circle_fits.append(np.linalg.norm(E_ij[j]))
            circle_wedge.append(E_ij[j])

            # Compute dual area on edge
            k = nn[j][0]  # - 1
            wedge_ij_ik = np.cross(E_ij[j], E_ij[k])
            E_jk[j] = F[k] - F[j]
            E_ik[j] = F[k] - F[i]
            if np.dot(normalized(wedge_ij_ik)[0], n_i) < 0:
                #print(f'WARNING: Wrong direction in boundary curvature')
                #k, l = l, k
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
            #E_ik[j] = F[k] - F[i]
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
            #C_ij[k] = c_ijk
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
        #if 1:
        try:
          k = nn[j][0]  # - 1
          l = nn[j][1]  # - 1
        except IndexError:
          print('l1326IndexError')
          print('j',j)
          print('F',F)
          print('nn',nn)

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
        #b = np.linalg.norm(F[k] - F[l], axis=0)  # l_lk
        b = np.linalg.norm(F[k] - F[j], axis=0)  # l_lk
        alpha_ij = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))
        #a = np.linalg.norm(F[k] - F[i], axis=0)  # l_il  # Symmetric to b
        a = np.linalg.norm(F[l] - F[i], axis=0)  # l_
        b = np.linalg.norm(F[l] - F[j], axis=0)  # l_lj
        beta_ij = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))

        ## Curvatures
        # Vector curvatures
        HNdA_ij[j] = (cotan(alpha_ij) + cotan(beta_ij)) * (F[j] - F[i])
        #print(f'(cotan(alpha_ij) + cotan(beta_ij)) * (F[i] - F[j]) = {(cotan(alpha_ij) + cotan(beta_ij)) * (F[i] - F[j])}')
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
        #print(f'HNdA_ij[j] = {HNdA_ij[j]}')
        #print(f'C_ij[j] = {C_ij[j]}')
        HNdA_ij_Cij[j] = HNdA_ij[j] / C_ij[j]
        NdA_ij_Cij[j] = NdA_ij[j] / C_ij[j]

        #print(f'HNdA_ij_Cij[j] = {HNdA_ij_Cij[j]}')
        #HNdA_ij_Cij[j] = np.dot(HNdA_ij[j], n_i) / (C_ij[j])
        #HNdA_ij_Cij[j] = np.sum(np.dot(HNdA_ij[j], n_i) / (C_ij[j]), axis=0)
        #HNdA_ij_Cij[j] = np.sum(HNdA_ij_Cij[j])

    # Compute angles between boundary edges
    try:
        c_wedge = np.cross(circle_wedge[0], circle_wedge[1])
        # Interior angles: # Law of Cosines
        c = np.linalg.norm(circle_wedge[0] - circle_wedge[1], axis=0)
        a = np.linalg.norm(circle_wedge[0], axis=0)  # l_ik  # Symmetric to b
        b = np.linalg.norm(circle_wedge[1], axis=0)  # l_lk
        theta_i = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))
        #print(f'theta_i = {theta_i * 180 / np.pi}')

        #print(f'circle_fits = {circle_fits}')
        #TODO: THIS IS WRONG!
        # In 2D we have:
        # (x - x_c)**2 + (y - y_c)**2 = r**2
        # (x_1 - x_c)**2 + (y_1 - y_c)**2 = r**2
        # (x_2 - x_c)**2 + (y_2 - y_c)**2 = r**2
        # (x_3 - x_c)**2 + (y_3 - y_c)**2 = r**2
        # Subtract the first from the second, and the first from the third to
        # create two linear equations
        # (x_1 - x_c)**2 - (x_2 - x_c)**2  + (y_1 - y_c)**2 - (y_2 - y_c)**2  = 0

        #(x_3 - x_c) ** 2 + (y_3 - y_c) ** 2 = r ** 2
        r_est = np.linalg.norm(circle_fits)
        r_est = np.sqrt(circle_fits[0]**2 +  circle_fits[1]**2)
        #print(f'r_est = {r_est}')

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
    #NdA_ij_Cij = np.sum(NdA_ij_Cij, axis=0)
    # (^ The integrated area of the original smooth surface (a Dual discrete differential 2-form))


    # Point-wise estimates
    HN_i = np.sum(HNdA_i) / np.sum(C_ij)
    # TODO: Need to replace with dot n_i

    HN_i = np.sum(np.dot(HNdA_i, n_i))/np.sum(C_ij)
    #TURN THIS OFF IN NORMAL RUNS:
    if 0:
        HN_i = np.sum(np.dot(HNdA_i, normalized(np.sum(NdA_ij_Cij, axis=0))[0])) / np.sum(C_ij)
    if 0:
        HNdA_i = np.sum(HNdA_ij_Cij, axis=0)
        HN_i = np.sum(
            np.dot(HNdA_i, normalized(np.sum(NdA_ij_Cij, axis=0))[0])) / np.sum(
            C_ij)
    if 1:
        #print(f'C_ij = {C_ij}')
        C_i = np.sum(C_ij, axis=0)

    K_H_i = (HN_i/ 2.0)**2

    # nt development:
    if 0:
        print('-')
        #print(f'HNdA_i= {HNdA_i}')

        print(f'n_i = {n_i}')
        print(f'C_i = {C_i}')
     #   print(f'normalized(C_i) = {normalized(C_i)[0]}')
        nt = []
        for nda, a_ijk in zip(NdA_ij, A_ijk):
            #print(f'nda = {nda}')
            #print(f'a_ijk = {a_ijk}')
            nt.append(nda/a_ijk)

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
        #print(f'normalized(HNdA_i) = {normalized(HNdA_i)[0]}')
        #print(f'normalized(HNdA_ij_Cij) = {normalized(np.sum(HNdA_ij_Cij, axis=0))[0]}')
        #print(f'normalized(NdA_i ) = {normalized(NdA_i)[0]}')
        #print(f'HNdA_ij_Cij = {HNdA_ij_Cij}')
    HNdA_ij_Cij = np.sum( np.sum(HNdA_ij_Cij, axis=0))
    
    return dict(**locals())

def construct_HC(F, nn):
    """
    Construct a simplicial complex from vectorised point cloud data
    :param F: Vector of function values in 3-dimensional space
    :param nn: List of neighbouring vertices (list of indices kn F)
    :return: HC, simplicial `Complex` object
    """
    HC = Complex(3)
    V = []
    for f in F:
        V.append(HC.V[tuple(f)])
    for i, i_nn in enumerate(nn):
        for i_v2 in i_nn:
            V[i].connect(V[i_v2])
    return HC

class Print_i(object):
    def __init__(self, c_outd):
        self.__dict__.update(c_outd)
        print('============================' * 5)
        print(f'Wedge_ij_ik =')
        print(f'{self.Wedge_ij_ik}')
        print(f'A_ijk = {self.A_ijk}')
        print(f'C_ijk = {self.C_ijk}')
        print(f'N_ijk = ')
        print(f'{self.N_ijk}')
        print(f'N_ijl =')
        print(f'{self.N_ijl}')
        # print(f'0.5*r*r = {0.5*r*r}')
        print(f'theta_i_jk = {self.theta_i_jk * 180 / np.pi}')
        print(f'Theta_i_jk = {self.Theta_i_jk}')
        # print(f'theta_i_jk*8 = {6*(2 * np.pi - 4*theta_i_jk)} should be 2 * pi * Xi = {2 * np.pi * 2 }' )
        print(f'varphi_ij = {self.varphi_ij * 180 / np.pi}')
        print(f'Varphi_ij = {self.Varphi_ij}')
        print(f'KNdA_ij = {self.KNdA_ij}')
        print(f'H_ij = {self.H_ij}')
        print(f'-')
        print(f'Analytical Gaussian curvature: K_f = 2/R = {self.K_f}')
        print(f'Analytical mean curvature: H_f = {self.H_f}')
        print(f'-')
        print(f'Vector curvatures (integrated):')
        print(f'==================')
        # print(f'A_ij = {A_ij}')
        # print(f'int_boundary NdA = {NdA}')
        print(f'KNdA_i = {self.KNdA_i}')
        print(f'HNdA_i = {self.HNdA_i}')
        print(f'HN_i = {self.HN_i}')
        # print(f'HN_ij = {HN_ij}')
        print(f'H_ij = {self.H_ij}')
        print(f'NdA_i = {self.NdA_i}')
        print(f'-')
        print(f'Scalar curvatures:')
        print(f'==================')
        print(f'K = Omega_i = {self.Omega_i}')
        print(f'H_ij_sum = {self.H_ij_sum}')
        print(f'H_i = {self.H_i}')
        print(f'H_i/A_ijk = {self.H_i / self.A_ijk}')
        print(f'H_i/NdA_i = {self.H_i / self.NdA_i}')
        print(f'H_ij_sum /np.sum(A_ijk) = {self.H_ij_sum / np.sum(self.A_ijk)}')
        print(f'H_ij_sum /np.sum(NdA_i) = {self.H_ij_sum / np.sum(self.NdA_i)}')
        print(f'H_ij_sum * (np.sum(A_ijk)/np.sum(NdA_i)) = {self.H_ij_sum * (np.sum(self.A_ijk) / np.sum(self.NdA_i))}')
        print(f'H_ij_sum * (np.sum(NdA_i)/np.sum(A_ijk)) = {self.H_ij_sum * (np.sum(self.NdA_i) / np.sum(self.A_ijk))}')
        print(
            f'H_ij_sum * (np.sum(A_ijk)/np.sum(NdA_i)) / 1/np.sum(NdA_i) = {self.H_ij_sum * (np.sum(self.A_ijk) / np.sum(self.NdA_i)) * np.sum(self.NdA_i)}')
        print(
            f'H_ij_sum * (np.sum(NdA_i)/np.sum(A_ijk)) / np.sum(NdA_i) = {self.H_ij_sum * (np.sum(self.NdA_i) / np.sum(self.A_ijk)) / np.sum(self.NdA_i)}')

        print(f'Principal scalar curvatures:')
        print(f'==================')
        print(f'H = (np.sqrt(K) + np.sqrt(K))/2 = Omega_i = {(np.sqrt(self.K) + np.sqrt(self.K)) / 2}')
        print(f'K = H_i**2 = Omega_i = {self.H_i ** 2}')

# Gauss-Bonnet
def old_b_curvatures_i(F, nn, b_F, b_nn, n_i=None):
    """
    Boundary curvatures
    F: Array of vectors forming the star domain of v_i = F[0]
    nn: Connections within each vertex in union with the star domain
    n_i: Normal vector at vertex v_i
    """

    bE_ij = b_F - F[0]
    # Discrete vector area (only two edges)
    # Simplex areas of ijk and normals
    wedge_ij_ik = np.cross(bE_ij[0], bE_ij[1])

    # Compute k_g
    theta_i_jk = np.arctan2(np.linalg.norm(wedge_ij_ik), np.dot(bE_ij[0], bE_ij[1]))
    k_g = np.pi - theta_i_jk #- othjers

    if 0:
        if n_i is None:
            n_i = normalized(F[0])[0]
        #print(f'n_i = {n_i}')
        # TODO: Later we can cache these containers to avoid extra computations
        # Edges from i to j:
        E_ij = F[1:] - F[0]
        E_ij = np.vstack([np.zeros(3), E_ij])
        E_jk = np.zeros_like(E_ij)
        E_ik = np.zeros_like(E_ij)
        #print(f'E_ij = {E_ij}')
        hat_E_ij = normalized(E_ij)
        # E_ij = e_ij
        L_ij = np.linalg.norm(E_ij, axis=1)
        #print(f'L_ij = {L_ij}')
        Varphi_ij = np.zeros_like(L_ij)

        # Edge midpoints
        mdp_ij = 0.5 * E_ij + F[0]  # = 0.5*E_ik + v_i
        mdp_ik = np.zeros_like(E_ij)

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
        KNdA_ij = np.zeros_like(E_ij)  # Vector Gauss curvature normal sums
        HNdA_ij = np.zeros_like(E_ij)  # Vector mean curvature normal sums
        NdA_ij = np.zeros_like(E_ij)  # Area curvature normal sums

        # Scalar curvature containers
        H_ij = np.zeros_like(L_ij)  # Edge mean curvatures
        A_ijk = A_ijk  # Area
        C_ijk = np.zeros_like(A_ijk)  # Dual area
        V_ijk = np.zeros_like(A_ijk)  # Volume
        # A_ij = np.zeros_like(L_ij)   # Vector area ??

        i = 0
        # Note, every neighbour adds  precisely one simplex:
        for j in nn[0]:
            # Recover indices from nn (in code recover the vertex entry in the cache)
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
            #b = np.linalg.norm(F[k] - F[l], axis=0)  # l_lk
            b = np.linalg.norm(F[k] - F[j], axis=0)  # l_lk
            alpha_ij = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))
            #a = np.linalg.norm(F[k] - F[i], axis=0)  # l_il  # Symmetric to b
            a = np.linalg.norm(F[l] - F[i], axis=0)  # l_
            b = np.linalg.norm(F[l] - F[j], axis=0)  # l_lj
            beta_ij = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))

            ## Curvatures
            # Vector curvatures
            # Guassian normal curvature
            KNdA_ij[j] = (varphi_ij) / (L_ij[j]) * (F[j] - F[i])
            # Mean normal curvatre
            # Ratio of dual/primal length is given by cotan formula, yielding
            # HNdA_ij[j] = (cotan(alpha_ij) + cotan(beta_ij)) * (F[i] - F[j])
            HNdA_ij[j] = (cotan(alpha_ij) + cotan(beta_ij)) * (F[j] - F[i])
            #print(f'(cotan(alpha_ij) + cotan(beta_ij)) * (F[i] - F[j]) = {(cotan(alpha_ij) + cotan(beta_ij)) * (F[i] - F[j])}')
            # NdA_ij[j] = np.cross(F[i], F[j])
            # (^ NOTE: The above vertices i, j MUST be consecutive for this formula to be valid (CHECK!))
            NdA_ij[j] = np.cross(F[j], F[k])
            # (^ NOTE: The above vertices j, k MUST be consecutive for this formula to be valid (CHECK!))
            # Scalar component ijk
            # Interior angle
            theta_i_jk = np.arctan2(np.linalg.norm(wedge_ij_ik), np.dot(E_ij[j], E_ij[k]))
            Theta_i_jk[j] = theta_i_jk
            # Mean normal
            H_ij[j] = L_ij[j] * varphi_ij

            # Areas
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
            v_dual = np.linalg.solve(A, c)
            h_ij = np.linalg.norm(0.5 * L_ij[j])  # = 0.5*E_ik  + F[0] ?   = F[0] -(0.5*E_ik  + F[0]) = 0.5
            b_ij = np.linalg.norm(v_dual - mdp_ij[j])  # wrong
            C_ij = 0.5 * b_ij * h_ij
            # h_ik = np.linalg.norm(0.5*L_ij[int(j_k[j])])  # = 0.5*E_ik  + F[0] ?   = F[0] -(0.5*E_ik  + F[0]) = 0.5
            h_ik = np.linalg.norm(0.5 * L_ij[k])
            b_ik = np.linalg.norm(v_dual - mdp_ik[j])
            C_ik = 0.5 * b_ik * h_ik
            C_ijk[j] = C_ij + C_ik  # the area dual to A_ijk (validated at 90deg/0 curvature)

        pass
        N_ijk = np.array(N_ijk)
        N_ijl = np.array(N_ijl)
        Wedge_ij_ik = np.array(Wedge_ij_ik)
        KNdA_i = 0.5 * np.sum(KNdA_ij, axis=0)  # Vector Gauss curvature normal sums
        # (^ The integrated area of the unit sphere)
        HNdA_i = 0.5 * np.sum(HNdA_ij, axis=0)  # Vector mean curvature normal sums (multiplied by N?)
        # HN_i = 0.5 * np.sum(HN_ij, axis=0)
        NdA_i = (1 / 6.0) * np.sum(NdA_ij, axis=0)  # Vector normal  are sums
        # (^ The integrated area of the original smooth surface (a Dual discrete differential 2-form))
        # KN_i = 0.5 * np.sum(KN_ij, axis=0)  # Vector Gauss curvature normal sums ????

        ## Scalar curvatures of vertex i:
        # Gaussian curvature (angle defct)
        Omega_i = 2 * np.pi - np.sum(Theta_i_jk)
        K = Omega_i
        # Mean curvature
        H_i = (1 / 4.0) * np.sum(H_ij)
        H_ij_sum = (1 / 2.0) * np.sum(H_ij)

        # Area curvature:

        # Volume curvature

        ## Principle curvatures
        # Smooth approximated curvatures:
        # krms np.sqrt(H_i**2 - K)
        # kappa_1 = H - krms
        # kappa_2 = H - krms
        # A_i  # A_i := abs(C_i) the area of the dual cell
    return dict(**locals())


def chi_H(HC, print_out=False):
    """
    Compute the 2D Euler Characterisitc
    """
    ## Computer Euler Characteristic:
    # Compute the number of vertices
    V = len(list(HC.V))
    # Compute the dges
    E = 0
    for v in HC.V:
        E += len(list(v.nn))

    E = E / 2.0  # We have added precisely twice the number of edges through adding each connection
    # Compute the faces
    HC.dim = 2  # We have taken the boundary of the sphere
    HC.vertex_face_mesh()
    Faces = HC.simplices_fm
    F = len(Faces)

    # Compute Euler
    chi = V - E + F
    if print_out:
        print(f'V = {V}')
        print(f'E = {E}')
        print(f'F = {F}')
        print(f'$\chi = V - E + F$ = {chi}')

    return chi

def K_t(HC, bV=set()):
    """
    Compute the integrated Gaussian curvature on the surface
    """
    KdA = 0.0
    for v in HC.V:
        if v in bV:
            continue
        else:
            N_f0 = v.x_a - np.zeros_like(v.x_a)  # First approximation # TODO: CHANGE FOR CAP RISE!
            F, nn = vectorise_vnn(v)
            # Compute discrete curvatures
            c_outd = curvatures(F, nn, n_i=N_f0)
            KdA += c_outd['Omega_i']  # == c_outd['K']

    return KdA

def k_g_t(HC, bV):
    k_g = 0
    for v in bV:

        Theta_i_jk = 0.0
        Simplices = set()
        Dual = set()
        for vn in v.nn:
            for vnn in vn.nn:
                if frozenset({vn, vnn}) in Dual:  # TODO:  think this should be uncommented
                    continue
                if vnn in v.nn:  # Add edges connected to v_i
                    E_ij = vn.x_a - v.x_a
                    E_ik = vnn.x_a - v.x_a
                    # Discrete vector area:
                    # Simplex areas of ijk and normals
                    wedge_ij_ik = np.cross(E_ij, E_ik)

                    # Wedge_ij_ik[j] = wedge_ij_ik
                    theta_i_jk = np.arctan2(np.linalg.norm(wedge_ij_ik), np.dot(E_ij, E_ik))
                    Theta_i_jk += theta_i_jk

                    Dual.add(frozenset({vn, vnn}))

        k_g += np.pi - Theta_i_jk

    return k_g


def Gauss_Bonnet(HC, bV=set(), print_out=False):
    """
    Compute a single iteration of mean curvature flow
    :param HC: Simplicial complex
    :return:
    """
    chi = chi_H(HC)
    KdA = K_t(HC, bV)
    k_g = k_g_t(HC, bV)
    if print_out:
        print(f' KdA = {KdA}')
        print(f' k_g = {k_g}')
        print(f' 2 pi chi$ = {2 * np.pi * chi}')
        print(f' chi$ = {chi}')
        print(f' LHS - RHS = {KdA + k_g - 2 * np.pi * chi}')
    return chi, KdA, k_g

#Gauss_Bonnet(HC, bV, print_out=1)

def Gauss_Bonnet_OLD(HC):
    """
    Compute a single iteration of mean curvature flow
    :param HC: Simplicial complex
    :return:
    """
    ## Computer Euler Characteristic:
    # Compute the number of vertices
    V = len(list(HC.V))
    # Compute the dges
    E = 0
    for v in HC.V:
        E += len(list(v.nn))

    E = E / 2.0  # We have added precisely twice the number of edges through adding each connection
    # Compute the faces
    HC.dim = 2  # We have taken the boundary of the sphere
    HC.vertex_face_mesh()
    Faces = HC.simplices_fm
    F = len(Faces)

    # Compute Euler
    chi = V - E + F
    print(f'V = {V}')
    print(f'E = {E}')
    print(f'F = {F}')
    chi = V - E + F
    print(f'$\chi = V - E + F$ = {chi}')

    return


def plot_variables(X, vdict, xlabel=r'Contact angle $\theta$', ylabel='-'):
    plot.figure()

    lstyles = ['-', '--', '-.', ':']
    mod = len(lstyles)
    ind = 0
    for key, value in vdict.items():
        plot.plot(X, value, linestyle=lstyles[ind], label=key, alpha=0.7)
        ind += 1
        ind = ind % mod

    plot.xlabel(xlabel)
    plot.ylabel(ylabel)
    plot.legend()


