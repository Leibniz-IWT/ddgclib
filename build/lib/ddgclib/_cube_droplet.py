import numpy as np
from ._curvatures import *

R = 1.0  # Radius of the sphere

theta = 0
phi = 0.0

Theta = np.linspace(0.0, 2*np.pi)
Phi = np.linspace(0.0, np.pi)
x, y, z = R * np.cos(theta) * np.sin(phi), R * np.sin(theta) * np.sin(phi), R * np.cos(phi)


def out_plot(N=5, R=0.5):
    Phi = np.linspace(0.1, 0.5 * np.pi, 20)
    # Phi = np.linspace((1/2.0)*np.pi, 0.9*np.pi, 50)
    H_f = 1 / R + 1 / R
    K_f = (1 / R) ** 2
    # dp_exact = gamma * H_f

    # Containers
    H_i = []
    H_ij_sum = []
    K = []
    KNdA_ij_sum = []
    KNdA_ij_dot = []
    HNdA_ij_sum = []
    HNdA_ij_dot = []
    # HNdA_i = []
    HdotNdA_ij_sum = []
    N_f0 = np.array([0.0, 0.0, 1.0])
    c_outd_list = []

    for phi in Phi:
        # Contruct the simplicial complex, plot the initial construction:
        F, nn = droplet_half_init(R, N, phi)
        HC = construct_HC(F, nn)
        v = HC.V[(0.0, 0.0, R)]
        F, nn = vectorise_vnn(v)

        # Compute discrete curvatures
        c_outd = curvatures(F, nn, n_i=N_f0)

        # Save results
        # c_outd = curvatures(F, nn)
        c_outd['K_f'] = K_f
        c_outd['H_f'] = H_f
        H_i.append(c_outd['H_i'])
        H_ij_sum.append(c_outd['H_ij_sum'])
        HNdA_ij_sum.append(np.sum(c_outd['HNdA_ij']))
        HNdA_ij_dot.append(np.sum(np.dot(c_outd['HNdA_ij'], c_outd['n_i'])))
        KNdA_ij_sum.append(np.sum(c_outd['KNdA_ij']))
        KNdA_ij_dot.append(np.sum(np.dot(c_outd['KNdA_ij'], c_outd['n_i'])))
        # HNdA_i.append()

        c_outd['HNdA_i']

        # print(f"HNdA_ij = {c_outd['HNdA_ij']}")
        # print(f" (np.dot(c_outd['HNdA_ij'], N_f0) = {np.dot(c_outd['HNdA_ij'], N_f0)}")
        HdotNdA_ij_sum.append(np.dot(c_outd['HNdA_ij'], N_f0))
        K.append(c_outd['K'])
        c_outd_list.append(c_outd)

    A_ijk = []
    C_ijk = []
    z = []

    for c_outd in c_outd_list:
        A_ijk.append(np.sum(c_outd['A_ijk']))
        C_ijk.append(np.sum(c_outd['C_ijk']))
        # Compute z:
        theta = 0.0
        r = R / np.cos(theta)  # = R at theta = 0
        y = r - r * np.sin(theta)
        theta_z = np.arctan(y / R)
        z_phi = y / np.sin(theta_z)
        z.append(z_phi)

    A_ijk = np.array(A_ijk)
    C_ijk = np.array(C_ijk)
    K_f = []
    H_f = []

    for c_outd in c_outd_list:
        K_f.append(np.sum(c_outd['K_f']))
        H_f.append(np.sum(c_outd['H_f']))
        # print(np.sum(c_outd['HNdA_ij'], axis=0))
        # HN_i.append(np.sum(c_outd['HN_i']))

    vdict = {'K_f': K_f,
             # 'K': K,
             'K/C_ijk': K / C_ijk,
             '- 0.5 * KNdA_ij_sum / C_ijk': -(1 / 2.0) * np.array(KNdA_ij_sum) / C_ijk,
             '- 0.5 * KNdA_ij_dot / C_ijk': -(1 / 2.0) * np.array(KNdA_ij_dot) / C_ijk,
             # '(K^0.5 + K^0.5) / C_ijk,': (np.sqrt(K) + np.sqrt(K))/2.0 / C_ijk,  # Way off
             '((K/C_ijk)^0.5 + (K/C_ijk)^0.5)': (np.sqrt(K / C_ijk) + np.sqrt(K / C_ijk)),  # /2.0,
             'H_f': H_f,
             # 'H_i': H_i,
             #   'H_i/C_ijk': H_i / C_ijk,  # Doesn't work as well as 2 * np.array(H_i) / C_ijk,
             #  'H_i/A_ijk': H_i / A_ijk,
             #  '2 * H_i/C_ijk': 2 * np.array(H_i) / C_ijk,  # Exactly equal to H_ij_sum/C_ijk': H_ij_sum/C_ijk,
             # '2*H_i*C_ijk/A_ijk': 2*(H_i*C_ijk/A_ijk),
             # 'H_ij_sum': H_ij_sum,
             '2 * H_i/C_ijk = H_ij_sum/C_ijk': H_ij_sum / C_ijk,
             ' -(1 / 2.0) * HNdA_ij_dot/C_ijk': -(1 / 2.0) * np.array(HNdA_ij_dot) / C_ijk,
             # 'HNdA_ij_sum/(N-1)': -(1 / (N - 1)) * np.array(HNdA_ij_sum) / C_ijk,
             # '2 * HNdA_ij_sum/(N-1)': -(2/(N-1))*np.array(HNdA_ij_sum)/C_ijk,
             #    '-(1/2)*HNdA_ij_sum/C_ijk': -(1 / 2.0) * np.array(HNdA_ij_sum) / C_ijk, # Exactly equal to  -(1 / 2.0) * HNdA_ij_dot/C_ijk
             # '-(1/2)*HNdA_ij_sum/C_ijk /R**2': (-(1 / 2.0) * np.array(HNdA_ij_sum)  / C_ijk) /R**2,
             # '-(1/2)*HNdA_ij_sum/C_ijk /A_ijk': (-(1 / 2.0) * np.array(HNdA_ij_sum)  / C_ijk) / A_ijk,
             # '-(1/4)*HNdA_ij_sum/C_ijk * Corr~': (
             #    -(1 / 4.0) * np.array(HNdA_ij_sum) / C_ijk) / (np.sqrt(np.array(z/R))) /R**2,
             # '(-(1 / 4.0) * np.array(HNdA_ij_sum) / C_ijk)/ H_f': (-(1 / 4.0) * np.array(HNdA_ij_sum) / C_ijk)/ H_f,
             # ' -(1 / 4.0) * (7 / (N - 1)) * np.array(HNdA_ij_sum) / C_ijk': -(1 / 4.0) * ((N - 1)  / 6) * np.array(HNdA_ij_sum) / C_ijk,
             # 'HN_i': HN_i
             }
    # print(-(1 / 2.0) * np.array(HNdA_ij_sum) / C_ijk)
    # print(-(1 / 2.0) *np.array(HNdA_ij_dot)/C_ijk)
    # print(-((1 / 2.0) * np.array(HNdA_ij_sum) / C_ijk) /R**2)
    # print(-((1 / 2.0) * np.array(HNdA_ij_sum) / C_ijk)  /  A_ijk)
    print(-(1 / 2.0) * np.array(KNdA_ij_sum) / C_ijk)
    print(-(1 / 2.0) * np.array(KNdA_ij_dot) / C_ijk)
    # print(f"H_f = {H_f}")
    # vdict = {'(-(1 / 4.0) * np.array(HNdA_ij_sum) / C_ijk)/ H_f': (-(1 / 4.0) * np.array(HNdA_ij_sum) / C_ijk)/ H_f,
    #         }

    X = Phi * 180 / np.pi

    return c_outd_list, c_outd, vdict, X

def out_plot_old(N=7, R=0.5):
    Phi = np.linspace(0.1, 0.5 * np.pi, 20)
    # Phi = np.linspace((1/2.0)*np.pi, 0.9*np.pi, 50)
    H_f = 1 / R + 1 / R
    K_f = (1 / R) ** 2
    #dp_exact = gamma * H_f

    # Containers
    H_i = []
    H_ij_sum = []
    K = []
    HNdA_ij_sum = []
    HdotNdA_ij_sum = []
    N_f0 = np.array([0.0, 0.0, 1.0])
    c_outd_list = []

    for phi in Phi:
        # Contruct the simplicial complex, plot the initial construction:
        F, nn = droplet_half_init(R, N, phi)
        HC = construct_HC(F, nn)
        v = HC.V[(0.0, 0.0, R)]
        F, nn = vectorise_vnn(v)

        # Compute discrete curvatures
        c_outd = curvatures(F, nn, n_i=N_f0)

        # Save results
        #c_outd = curvatures(F, nn)
        c_outd['K_f'] = K_f
        c_outd['H_f'] = H_f
        H_i.append(c_outd['H_i'])
        H_ij_sum.append(c_outd['H_ij_sum'])
        HNdA_ij_sum.append(np.sum(c_outd['HNdA_ij']))
        # print(f"HNdA_ij = {c_outd['HNdA_ij']}")
        # print(f" (np.dot(c_outd['HNdA_ij'], N_f0) = {np.dot(c_outd['HNdA_ij'], N_f0)}")
        HdotNdA_ij_sum.append(np.dot(c_outd['HNdA_ij'], N_f0))
        K.append(c_outd['K'])
        c_outd_list.append(c_outd)

    A_ijk = []
    C_ijk = []
    for c_outd in c_outd_list:
        A_ijk.append(np.sum(c_outd['A_ijk']))
        C_ijk.append(np.sum(c_outd['C_ijk']))

    A_ijk = np.array(A_ijk)
    C_ijk = np.array(C_ijk)
    K_f = []
    H_f = []
    for c_outd in c_outd_list:
        K_f.append(np.sum(c_outd['K_f']))
        H_f.append(np.sum(c_outd['H_f']))
        #HN_i.append(np.sum(c_outd['HN_i']))

    vdict = {'K_f': K_f,
             'K': K,
             'K/C_ijk': K / C_ijk,
             'H_f': H_f,
             'H_i': H_i,
             'H_i/C_ijk': H_i / C_ijk,
             'H_i/A_ijk': H_i / A_ijk,
             'H_i/C_ijk': H_i / C_ijk,
             '2 * H_i/C_ijk': 2 * np.array(H_i) / C_ijk,
             # '2*H_i*C_ijk/A_ijk': 2*(H_i*C_ijk/A_ijk),
             'H_ij_sum': H_ij_sum,
             'HNdA_ij_sum/(N-1)': -(1 / (N - 1)) * np.array(HNdA_ij_sum) / C_ijk,
             # '2 * HNdA_ij_sum/(N-1)': -(2/(N-1))*np.array(HNdA_ij_sum)/C_ijk,
             '(1/4)*HNdA_ij_sum/C_ijk': -(1 / 4.0) * np.array(HNdA_ij_sum) / C_ijk,
             #'HN_i': HN_i
             }
    X = Phi * 180 / np.pi


    return c_outd_list, c_outd, vdict, X

def droplet_init(r, theta, gamma, boundary_vertices=4, refinement=0):
    """
    DEPRECATED!
    :param r:
    :param theta:
    :param gamma:
    :param boundary_vertices:
    :param refinement:
    :return:
    """
    # Example calculation with the minimum discretisation
    # theta = 0  # NOTE: Not well defined?
    # theta = 5 * np.pi /180
    # theta = 10 * np.pi /180
    # theta = 20 * np.pi / 180
    # theta = 60 * np.pi /180
    # theta = 85 * np.pi /180  # Flat (No curvature!)
    #r = 1  # 1 mm radius of tube
    R = r / np.cos(theta)  # = R at theta = 0
    # Exact values:
    if theta < 90 * np.pi /180:
        K_f = 2 / R
        H_f = 1 / R + 1 / R  # 2 / R
    else:
        K_f = -2 / R
        H_f = -1 / R - 1 / R  # 2 / R

    dp_exact = gamma * (2/R)  # Pa      # Young-Laplace equation  dp = - gamma * H_f = - gamma * (1/R1 + 1/R2)

    ## Define vertex positions i and j = {1:4}
    # Let the 3 phase contact line form the plane where the function is zero,
    # we can define the heigh of the i vertex as follows using basic trig:
    fi = np.array([0.0, 0.0, R - R * np.sin(theta)])
    f1 = np.array([-r, 0.0, 0.0])
    f2 = np.array([+r, 0.0, 0.0])
    f3 = np.array([0.0, -r, 0.0])
    f4 = np.array([0.0, +r, 0.0])
    fi_nn = [f1, f2, f3, f4]
    F = [fi, f1, f2, f3, f4]  # Debugging

    # Connections (nearest neighbours)
    # Now let i=0 we find the other indices from nearest neighbours
    nn = np.array([[1, 2, 3, 4],  # f0_nn = f0_nn
                   [3, 4],  # f1_nn = interection (f0_nn, f1_nn) (always a 2 entries in R^3!)
                   [3, 4],  # f2_nn = interection (f0_nn, f2_nn) (always a 2 entries in R^3!)
                   [1, 2],  # f3_nn = interection (f0_nn, f3_nn) (always a 2 entries in R^3!)
                   [1, 2],  # f4_nn = interection (f0_nn, f4_nn) (always a 2 entries in R^3!)
                   ])
    return F, nn, K_f, H_f


def droplet_half_init(R, N, phi):
    # Theta = np.linspace(0.0, 2*np.pi)  # range of theta
    # Phi = np.linspace(0.0, np.pi)  # range of phi
    def sphere(R, theta, phi):
        return R * np.cos(theta) * np.sin(phi), R * np.sin(theta) * np.sin(phi), R * np.cos(phi)

    Theta = np.linspace(0.0, 2*np.pi, N)  # range of theta
    Phi = [0] + [phi,]*(N - 1)
    F = []
    nn = []
    ind = -1
    for theta, phi in zip(Theta, Phi):
        ind += 1
        # Define coordinates:
        x, y, z = sphere(R, theta, phi)
        F.append(np.array([x, y, z]))
        # Define connections:
        nn.append([])
        if ind > 0:
            nn[0].append(ind)
            nn[ind].append(0)
            nn[ind].append(ind - 1)
            nn[ind].append((ind + 1) % N)

    F = np.array(F)
    nn[1][1] = ind

    return F, nn

def mean_flow_old(HC, tau=0.5):
    """
    Compute a single iteration of mean curvature flow
    :param HC: Simplicial complex
    :return:
    """
    # Define the normals (NOTE: ONLY WORKS FOR CONVEX SURFACES)
    # Note that the cache HC.V is an OrderedDict:
    N_i = []

    for v in HC.V:
        N_i.append(normalized(v.x_a - np.array([0.5, 0.5, 0.5]))[0])

    f = []
    HNdA = []
    for i, v in enumerate(HC.V):
        F, nn = vectorise_vnn(v)
        c_outd = curvatures(F, nn, n_i=N_i[i])
       # HNdA.append(0.5*c_outd['HNdA_i'])
        #print(np.sum(c_outd['H_ij_sum']))
        #print(np.sum(c_outd['C_ijk']))
        #HNdA.append(np.sum(c_outd['H_ij_sum'])/np.sum(c_outd['C_ijk']))
        HNdA.append(N_i[i] * np.sum(c_outd['H_ij_sum'])/np.sum(c_outd['C_ijk']))
        f.append(v.x_a)

    pass
    #print(f'HNdA = { HNdA}')
    #print(f"(np.sum(c_outd['H_ij_sum'])/np.sum(c_outd['C_ijk']))  = {(np.sum(c_outd['H_ij_sum'])/np.sum(c_outd['C_ijk']))}")
    df = np.nan_to_num(np.array(HNdA))
    f = np.array(f)
    f_k = f - tau * df
    VA = []
    for v in HC.V:
        VA.append(v.x_a)

    VA = np.array(VA)
    for i, v_a in enumerate(VA):
        HC.V.move(HC.V[tuple(v_a)], tuple(f_k[i]))

    if 0:
        HCV2 = list(copy.copy(HC.V))
        for i, v in enumerate(HCV2):
            HC.V.move(v, tuple(f_k[i]))

    return HC


def cube_to_drop_init(refinements=2):
    # Initiate a cubical complex
    HC = Complex(3)
    HC.triangulate()

    for i in range(2):
        HC.refine_all()

    del_list = []
    for v in HC.V:
        if np.any(v.x_a == 0.0) or np.any(v.x_a == 1.0):
            continue
        else:
            del_list.append(v)

    for v in del_list:
        HC.V.remove(v)

    bV = set()

    return HC, bV