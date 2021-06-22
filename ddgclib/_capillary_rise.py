import numpy as np
import copy
from ._curvatures import *
import matplotlib.pyplot as plt

def sphere(R, theta, phi):
    return R * np.cos(theta) * np.sin(phi), R * np.sin(theta) * np.sin(phi), R * np.cos(phi)

def analytical_cap(r, theta_p):
    R = r / np.cos(theta_p)
    K_f = (1 / R) ** 2
    H_f = 2 / R  # 2 / R
    h = np.abs(R * np.sin(theta_p) - R)  # Height of the spherical cap
    # Area of spherical cap:
    dA = 2 * np.pi * R * h
    dC = 2 * np.pi * r
    a = R * np.cos(theta_p)
    # a = R * np.sin(theta_p)
    k_g_f = np.sqrt(R ** 2 - a ** 2) / (R * a)
    return H_f, K_f, dA, k_g_f, dC

def cap_rise_init_N(r, theta_p, gamma, N=4, refinement=0, cdist=1e-10, equilibrium=True):
    Theta = np.linspace(0.0, 2 * np.pi, N)  # range of theta
    R = r / np.cos(theta_p)  # = R at theta = 0
    # Exact values:
    K_f = (1 / R) ** 2
    H_f = 1 / R + 1 / R  # 2 / R
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
    return F, nn, HC, bV, K_f, H_f

def cap_rise_init_N_old(r, theta_p, gamma, N=4, refinement=0):
    Theta = np.linspace(0.0, 2 * np.pi, N)  # range of theta
    R = r / np.cos(theta_p)  # = R at theta = 0
    # Exact values:
    K_f = (1 / R) ** 2
    H_f = 1 / R + 1 / R  # 2 / R

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

    F = np.array(F)
    nn[1][1] = ind

    return F, nn, K_f, H_f

def out_plot_cap_rise(N=7, r=1, gamma=0.0728, refinement=0):
    #Theta_p = np.linspace(0.0, np.pi, 100)  # range of theta
    Theta_p = np.linspace(0.0, 0.5*np.pi, 100)  # range of theta

    # Containers
    H_i = []
    H_ij_sum = []
    K = []
    KNdA_ij_sum = []
    KNdA_ij_dot = []
    HNdA_ij_sum = []
    HNdA_ij_dot = []
    HdotNdA_ij_sum = []
    N_f0 = np.array([0.0, 0.0, -1])
    c_outd_list = []

    for theta_p in Theta_p:
        # Contruct the simplicial complex, plot the initial construction:
        # F, nn = droplet_half_init(R, N, phi)
        R = r / np.cos(theta_p)  # = R at theta = 0
        # Exact values:
        K_f = (1 / R) ** 2
        H_f = 1 / R + 1 / R  # 2 / R
        # dp_exact = gamma * H_f

        if 0:  # Old method
            F, nn, K_f, H_f = cap_rise_init_N_old(r, theta_p, gamma, N=N, refinement=0)
            HC = construct_HC(F, nn)
        else:  # New method
            F, nn, HC, bV, K_f, H_f = cap_rise_init_N(r, theta_p, gamma, N=N, refinement=refinement)

        R = r / np.cos(theta_p)  # = R at theta = 0
        v = HC.V[(0.0, 0.0, R * np.sin(theta_p) - R)]
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
        if 0:
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
        # HN_i.append(np.sum(c_outd['HN_i']))

    H_disc = (1 / 2.0) * np.array(HNdA_ij_sum) / C_ijk
    K_H = (H_disc / 2.0)
    vdict = {'K_f': K_f,
             # 'K': K,
             'K/C_ijk': K / C_ijk,
             ' 0.5 * KNdA_ij_sum / C_ijk': (1 / 2.0) * np.array(KNdA_ij_sum) / C_ijk,
             '- 0.5 * KNdA_ij_dot / C_ijk': -(1 / 2.0) * np.array(KNdA_ij_dot) / C_ijk,
             '((K/C_ijk)^0.5 + (K/C_ijk)^0.5)': (np.sqrt(K / C_ijk) + np.sqrt(K / C_ijk)),  # /2.0,
             'H_f': H_f,
             '2 * H_i/C_ijk = H_ij_sum/C_ijk': H_ij_sum / C_ijk,
             ' -(1 / 2.0) * HNdA_ij_dot/C_ijk': -(1 / 2.0) * np.array(HNdA_ij_dot) / C_ijk,
             '(1/2)*HNdA_ij_sum/C_ijk': (1 / 2.0) * np.array(HNdA_ij_sum) / C_ijk,
             # Exactly equal to  -(1 / 2.0) * HNdA_ij_dot/C_ijk
             'K_H': K_H ** 2,
             'K_H 2': K_H ** 2
             # ' -(1 / 2.0) * HNdA_ij_dot/A_ijk': -(1 / 2.0) * np.array(HNdA_ij_dot)/A_ijk,
             # ' -(1 / 2.0) * HNdA_ij_dot/C_ijk / H_f': (-(1 / 2.0) * np.array(HNdA_ij_dot)/C_ijk)/np.array(H_f),
             # ' (-(1 / 2.0) * HNdA_ij_dot/C_ijk)/r**2': r* ((-(1 / 2.0) * np.array(HNdA_ij_dot)/C_ijk)/r**2),
             # ' (-(1 / 2.0) * HNdA_ij_dot/C_ijk)/r**2/H_f': ((-(1 / 2.0) * np.array(HNdA_ij_dot)/C_ijk)/r**2)/np.array(H_f)
             }
    X = Theta_p * 180 / np.pi

    return c_outd_list, c_outd, vdict, X


def new_out_plot_cap_rise(N=7, r=1, gamma=0.0728, refinement=0):
    #Theta_p = np.linspace(0.0, np.pi, 100)  # range of theta
    Theta_p = np.linspace(0.0, 0.5*np.pi, 100)  # range of theta

    # Containers
    H_i = []
    H_ij_sum = []
    K = []
    KNdA_ij_sum = []
    KNdA_ij_dot = []
    HNdA_ij_sum = []
    HNdA_ij_dot = []
    HdotNdA_ij_sum = []
    N_f0 = np.array([0.0, 0.0, -1])
    c_outd_list = []

    for theta_p in Theta_p:
        # Contruct the simplicial complex, plot the initial construction:
        # F, nn = droplet_half_init(R, N, phi)
        R = r / np.cos(theta_p)  # = R at theta = 0
        # Exact values:
        K_f = (1 / R) ** 2
        H_f = 1 / R + 1 / R  # 2 / R
        # dp_exact = gamma * H_f

        F, nn, HC, bV, K_f, H_f = cap_rise_init_N(r, theta_p, gamma, N=N, refinement=refinement)

        R = r / np.cos(theta_p)  # = R at theta = 0
        v = HC.V[(0.0, 0.0, R * np.sin(theta_p) - R)]
        F, nn = vectorise_vnn(v)

        # Compute discrete curvatures (centre vertex)
        c_outd = curvatures_hn_ij_c_ij(F, nn, n_i=N_f0)

      #  HNdA_ij_Cij
        # Save results
        # c_outd = curvatures(F, nn)
        c_outd['K_f'] = K_f
        c_outd['H_f'] = H_f
        if 1:
         #   H_i.append(c_outd['H_i'])
          #  H_ij_sum.append(c_outd['H_ij_sum'])
            #HNdA_ij_sum.append(np.sum(c_outd['HNdA_ij_Cij']))
         #   HNdA_ij_sum.append(-np.sum(np.dot(c_outd['HNdA_ij'], c_outd['n_i'])))
            HNdA_ij_sum.append(-np.sum(np.dot(c_outd['HNdA_i'], c_outd['n_i'])))
           # HNdA_ij_dot.append(np.sum(np.dot(c_outd['HNdA_ij'], c_outd['n_i'])))
           # KNdA_ij_sum.append(np.sum(c_outd['KNdA_ij']))
           # KNdA_ij_dot.append(np.sum(np.dot(c_outd['KNdA_ij'], c_outd['n_i'])))
            # HNdA_i.append()

        # New particles
        if 0:
            sum_HNdA_ij_Cij = np.sum(c_outd['HNdA_ij_Cij'], axis=0)
            print(f"np.sum(HNdA_i_Cij, axis=0)) = {sum_HNdA_ij_Cij}")
            print(f"np.sum(HNdA_i_Cij) = {np.sum(c_outd['HNdA_ij_Cij'])}")
            HNdA_ij_Cij_dot_NdA_i = np.dot(c_outd['NdA_i'], sum_HNdA_ij_Cij)
            HNdA_ij_sum.append(-HNdA_ij_Cij_dot_NdA_i)

       # c_outd['HNdA_i']

        # print(f"HNdA_ij = {c_outd['HNdA_ij']}")
        # print(f" (np.dot(c_outd['HNdA_ij'], N_f0) = {np.dot(c_outd['HNdA_ij'], N_f0)}")
     #   HdotNdA_ij_sum.append(np.dot(c_outd['HNdA_ij'], N_f0))
        #K.append(c_outd['K'])
        K.append(K_f)
        c_outd_list.append(c_outd)

    A_ijk = []
    C_ijk = []
    z = []

    for c_outd in c_outd_list:
        A_ijk.append(np.sum(c_outd['A_ijk']))
        C_ijk.append(np.sum(c_outd['C_ijk']))
        # Compute z:
        if 0:
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
        # HN_i.append(np.sum(c_outd['HN_i']))

    #H_disc = (1 / 2.0) * np.array(HNdA_ij_sum) / C_ijk
    H_disc = (1 / 2.0) * np.array(HNdA_ij_sum) / C_ijk
    #K_H = (H_disc / 2.0)
    K_H = 2*(H_disc / 2.0)
    vdict = {'K_f': K_f,
             # 'K': K,
             'K/C_ijk': K / C_ijk,
           #  ' 0.5 * KNdA_ij_sum / C_ijk': (1 / 2.0) * np.array(KNdA_ij_sum) / C_ijk,
           #  '- 0.5 * KNdA_ij_dot / C_ijk': -(1 / 2.0) * np.array(KNdA_ij_dot) / C_ijk,
             '((K/C_ijk)^0.5 + (K/C_ijk)^0.5)': (np.sqrt(K / C_ijk) + np.sqrt(K / C_ijk)),  # /2.0,
             'H_f': H_f,
           #  '2 * H_i/C_ijk = H_ij_sum/C_ijk': H_ij_sum / C_ijk,
            # ' -(1 / 2.0) * HNdA_ij_dot/C_ijk': -(1 / 2.0) * np.array(HNdA_ij_dot) / C_ijk,
             '(1/2)*HNdA_ij_sum/C_ijk':  np.array(HNdA_ij_sum)/ C_ijk,
             '(1/2)*HNdA_ij_sum':  np.array(HNdA_ij_sum) / 2.0,
             # Exactly equal to  -(1 / 2.0) * HNdA_ij_dot/C_ijk
             'K_H': K_H ** 2,
             'K_H 2': K_H ** 2
             # ' -(1 / 2.0) * HNdA_ij_dot/A_ijk': -(1 / 2.0) * np.array(HNdA_ij_dot)/A_ijk,
             # ' -(1 / 2.0) * HNdA_ij_dot/C_ijk / H_f': (-(1 / 2.0) * np.array(HNdA_ij_dot)/C_ijk)/np.array(H_f),
             # ' (-(1 / 2.0) * HNdA_ij_dot/C_ijk)/r**2': r* ((-(1 / 2.0) * np.array(HNdA_ij_dot)/C_ijk)/r**2),
             # ' (-(1 / 2.0) * HNdA_ij_dot/C_ijk)/r**2/H_f': ((-(1 / 2.0) * np.array(HNdA_ij_dot)/C_ijk)/r**2)/np.array(H_f)
             }
    X = Theta_p * 180 / np.pi

    return c_outd_list, c_outd, vdict, X


def plot_c_outd(c_outd_list, c_outd, vdict, X, ylabel=r'$m$ or $m^{-1}$'):
    fig = plt.figure()
    # ax = fig.add_subplot(2, 1, 1)
    ax = fig.add_subplot(1, 1, 1)

    lstyles = ['-', '--', '-.', ':']
    mod = len(lstyles)
    ind = 0
    Lines = {}
    fig.legend()
    for key, value in vdict.items():
        line, = ax.plot(X, value, linestyle=lstyles[ind], label=key, alpha=0.7)
        Lines[key] = line
        # plot.plot(X, value, linestyle=lstyles[ind], label=key, alpha=0.7)
        ind += 1
        ind = ind % mod

    plot.xlabel(r'Contact angle $\Theta_{C}$')
    plot.ylabel(ylabel)
    # fig.legend(bbox_to_anchor=(1, 0.5), loc='right', ncol=2)
    fig.legend(ncol=2)
    # interact(update);
    return fig

def out_plot_cap_rise_OLD(N=7, r=1, theta_p=0.0, gamma=0.0728, refinement=0):
    Theta_p = np.linspace(0.0, 2 * np.pi, 50)  # range of theta
    # Phi = np.linspace((1/2.0)*np.pi, 0.9*np.pi, 50)
    R = r / np.cos(theta_p)  # = R at theta = 0
    # Exact values:
    K_f = (1 / R) ** 2
    H_f = 1 / R + 1 / R  # 2 / R
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
    N_f0 = np.array([0.0, 0.0, R * np.sin(theta_p) - R])
    c_outd_list = []

    for theta_p in Theta_p:
        # Contruct the simplicial complex, plot the initial construction:
        # F, nn = droplet_half_init(R, N, phi)
        F, nn, K_f, H_f = cap_rise_init_N(r, theta_p, gamma, N=N, refinement=0)
        HC = construct_HC(F, nn)
        R = r / np.cos(theta_p)  # = R at theta = 0
        v = HC.V[(0.0, 0.0, R * np.sin(theta_p) - R)]
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
        # HN_i.append(np.sum(c_outd['HN_i']))

    vdict = {'K_f': K_f,
             # 'K': K,
             'K/C_ijk': K / C_ijk,
             '- 0.5 * KNdA_ij_sum / C_ijk': -(1 / 2.0) * np.array(KNdA_ij_sum) / C_ijk,
             '- 0.5 * KNdA_ij_dot / C_ijk': -(1 / 2.0) * np.array(KNdA_ij_dot) / C_ijk,
             '((K/C_ijk)^0.5 + (K/C_ijk)^0.5)': (np.sqrt(K / C_ijk) + np.sqrt(K / C_ijk)),  # /2.0,
             'H_f': H_f,
             '2 * H_i/C_ijk = H_ij_sum/C_ijk': H_ij_sum / C_ijk,
             ' -(1 / 2.0) * HNdA_ij_dot/C_ijk': -(1 / 2.0) * np.array(HNdA_ij_dot) / C_ijk,
             }
    X = Theta_p * 180 / np.pi

    return c_outd_list, c_outd, vdict, X

def out_plot_cap_rise_boundary(N=7, r=1, gamma=0.0728, refinement=1):
    Theta_p = np.linspace(0.0, np.pi, 100)  # range of theta

    # Containers
    H_i = []
    H_ij_sum = []
    K = []
    KNdA_ij_sum = []
    KNdA_ij_dot = []
    HNdA_ij_sum = []
    HNdA_ij_dot = []
    HdotNdA_ij_sum = []
    N_f0 = np.array([0.0, 0.0, -1])  #TODO: ???
    c_outd_list = []

    for theta_p in Theta_p:
        # Contruct the simplicial complex, plot the initial construction:
        # F, nn = droplet_half_init(R, N, phi)
        R = r / np.cos(theta_p)  # = R at theta = 0
        # Exact values:
        K_f = (1 / R) ** 2
        H_f = 1 / R + 1 / R  # 2 / R
        # dp_exact = gamma * H_f

        if 0:  # Old method
            F, nn, HC, K_f, H_f = cap_rise_init_N(r, theta_p, gamma, N=N, refinement=refinement)
            HC = construct_HC(F, nn)
        else:
            F, nn, HC, bV, K_f, H_f = cap_rise_init_N(r, theta_p, gamma, N=N, refinement=refinement)
            HC = construct_HC(F, nn)
        R = r / np.cos(theta_p)  # = R at theta = 0
        v = HC.V[(0.0, 0.0, R * np.sin(theta_p) - R)]
        F, nn = vectorise_vnn(v)

        # boundary vertices are all on z=0:
        bV = set()
        for v in HC.V:
            if v.x[2] == 0.0:
                bV.add(v)

        # Compute discrete curvatures
        c_outd = curvatures(F, nn, n_i=N_f0)

        # Compute boundary curvatures and Gauss_Bonnet
        chi, KdA, k_g = Gauss_Bonnet(HC, bV, print_out=False)

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

        c_outd['chi'] = chi
        c_outd['KdA'] = KdA
        c_outd['k_g'] = k_g

        # print(f"HNdA_ij = {c_outd['HNdA_ij']}")
        # print(f" (np.dot(c_outd['HNdA_ij'], N_f0) = {np.dot(c_outd['HNdA_ij'], N_f0)}")
        HdotNdA_ij_sum.append(np.dot(c_outd['HNdA_ij'], N_f0))
        K.append(c_outd['K'])
        c_outd_list.append(c_outd)

    A_ijk = []
    C_ijk = []
    z = []
    chi = []
    KdA = []
    k_g = []

    for c_outd in c_outd_list:
        A_ijk.append(np.sum(c_outd['A_ijk']))
        C_ijk.append(np.sum(c_outd['C_ijk']))
        chi.append(c_outd['chi'])
        KdA.append(c_outd['KdA'])
        k_g.append(c_outd['k_g'])
        # Compute z:
        if 0:
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
        # HN_i.append(np.sum(c_outd['HN_i']))

    vdict = {'chi': chi,
             'KdA': KdA,
             'k_g': k_g,
             '2 pi chi - KdA - k_g': 2 * np.pi * np.array(chi) - KdA - k_g,
             'K_f': K_f,
             # 'K': K,
             'K/C_ijk': K / C_ijk,
             # ' 0.5 * KNdA_ij_sum / C_ijk': (1 / 2.0) * np.array(KNdA_ij_sum) / C_ijk,
             # '- 0.5 * KNdA_ij_dot / C_ijk': -(1 / 2.0) * np.array(KNdA_ij_dot) / C_ijk,
             # '((K/C_ijk)^0.5 + (K/C_ijk)^0.5)': (np.sqrt(K/C_ijk) + np.sqrt(K/C_ijk)) ,#/2.0,
             # 'H_f': H_f,
             # '2 * H_i/C_ijk = H_ij_sum/C_ijk': H_ij_sum/C_ijk,
             # ' -(1 / 2.0) * HNdA_ij_dot/C_ijk': -(1 / 2.0) * np.array(HNdA_ij_dot)/C_ijk,
             # '-(1/2)*HNdA_ij_sum/C_ijk': (1 / 2.0) * np.array(HNdA_ij_sum) / C_ijk, # Exactly equal to  -(1 / 2.0) * HNdA_ij_dot/C_ijk
             # ' -(1 / 2.0) * HNdA_ij_dot/A_ijk': -(1 / 2.0) * np.array(HNdA_ij_dot)/A_ijk,
             # ' -(1 / 2.0) * HNdA_ij_dot/C_ijk / H_f': (-(1 / 2.0) * np.array(HNdA_ij_dot)/C_ijk)/np.array(H_f),
             # ' (-(1 / 2.0) * HNdA_ij_dot/C_ijk)/r**2': r* ((-(1 / 2.0) * np.array(HNdA_ij_dot)/C_ijk)/r**2),
             # ' (-(1 / 2.0) * HNdA_ij_dot/C_ijk)/r**2/H_f': ((-(1 / 2.0) * np.array(HNdA_ij_dot)/C_ijk)/r**2)/np.array(H_f)
             }
    X = Theta_p * 180 / np.pi

    return c_outd_list, c_outd, vdict, X

def cap_rise_init(r, theta, gamma, boundary_vertices=4, refinement=0):
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
        K_f = (1 / R)**2
        H_f = 1 / R + 1 / R  # 2 / R
    else:
        K_f = (1 / R)**2
        H_f = -1 / R - 1 / R  # 2 / R

    dp_exact = gamma * (2/R)  # Pa      # Young-Laplace equation  dp = - gamma * H_f = - gamma * (1/R1 + 1/R2)

    ## Define vertex positions i and j = {1:4}
    # Let the 3 phase contact line form the plane where the function is zero,
    # we can define the heigh of the i vertex as follows using basic trig:
    fi = np.array([0.0, 0.0, R - R * np.sin(theta)])
    f1 = np.array([-r, 0.0, 0.0])
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

def cape_rise_plot(r, theta_p, gamma, N=5, refinement=0):
    # F, nn =F, nn = droplet_half_init(R, N, phi)
    R = r / np.cos(theta_p)  # = R at theta = 0
    F, nn, HC, bV, K_f, H_f = cap_rise_init_N(r, theta_p, gamma, N=N, refinement=refinement)
    #fig, axes, HC = plot_surface(F, nn)
    fig, axes, fig_s, axes_s = HC.plot_complex(point_color=db, line_color=lb)
    axes.set_xlim3d(-(0.1 * R + R), 0.1 * R + R)
    axes.set_ylim3d(-(0.1 * R + R), 0.1 * R + R)
    axes.set_zlim3d(-(0.1 * R + R), 0.1 * R + R)

    # Cylinder
    x = np.linspace(-r, r, 100)
    z = np.linspace(-r, r, 100)
    Xc, Zc = np.meshgrid(x, z)
    Yc = np.sqrt(r ** 2 - Xc ** 2)
    rstride = 20
    cstride = 10
    axes.plot_surface(Xc, Yc, Zc,
                      alpha=0.4, rstride=rstride, cstride=cstride, color=do)

    return fig, axes, HC

def droplet_half_init(R, N, phi):
    # Theta = np.linspace(0.0, 2*np.pi)  # range of theta
    # Phi = np.linspace(0.0, np.pi)  # range of phi
    def sphere(R, theta, phi):
        return R * np.cos(theta) * np.sin(phi), R * np.sin(theta) * np.sin(phi), R * np.cos(phi)

    Theta = np.linspace(0.0, 2 * np.pi, N)  # range of theta
    Phi = [np.pi] + [phi, ] * (N - 1)
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