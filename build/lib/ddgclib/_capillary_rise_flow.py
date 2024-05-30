import numpy as np
from ._curvatures import *
from ._capillary_rise import *





def kmean_flow(HC, bV, params, tau=0.5, print_out=True):
    """
    Compute a single iteration of mean curvature flow
    :param HC: Simplicial complex
    :return:
    """
    (gamma, rho, g, r, theta_p, K_f, h) = params
    print('.')
    # Define the normals (NOTE: ONLY WORKS FOR CONVEX SURFACES)
    # Note that the cache HC.V is an OrderedDict:
    N_i = []

    f = []
    for i, v in enumerate(HC.V):
        if v in bV:
            continue
        f.append(v.x_a)

    # Compute interior curvatures
    (HNda_v_cache, K_H_cache, C_ijk_v_cache, HN_i, HNdA_ij_dot_hnda_i,
     K_H_2, HNdA_i_Cij) = int_curvatures(HC, bV, r, theta_p, printout=False)

    # H = np.nan_to_num(-(1 / 2.0)*np.array(HN))
    #H = np.nan_to_num(-np.array(HN))
    H = np.nan_to_num(HNdA_ij_dot_hnda_i)  # TODO: Future use HN_i
    H = np.nan_to_num(HN_i)  # TODO: Future use HN_i
    #for i, h in enumerate(H):
    #    H[i] = np.max([H[i], 0.0])
    df = gamma * H  # Hydrostatic pressure

    # Add gravity force rho * g * h
    height = []
    for v in HC.V:
        if (v in bV):
            continue
        height.append(np.max([v.x_a[2], 0.0]))

    height = np.array(height)
    df = df -(rho * g * height) #* np.array([0.0, 0.0, -1.0])

#    print(f'height = {height}')
    f = np.array(f)
    dF = np.zeros_like(f)
    dF[:, 2] = df
    #f_k = f + tau * dF # Previously: f_k = f + tau * df
    dF_max = np.max(dF)
    t = tau
    t = 0.5* r /(dF_max + r) #tau
#    print(f't = {t}')
#    print(f't * dF = {t * dF}')
    tdF = t * dF
    for i, tdf in enumerate(tdF):
        if np.abs(tdf[2]) > r:  # Alleviate floating point errors
            #tdF[i, 2] = np.sign(tdf[2]) * 0.05 * r
            tdF[i, 2] = np.sign(tdf[2]) * 0.05 * h
    f_k = f + tdF # Previously: f_k = f + tau * df

    # Apply constraint z > 0  (not really needed)
   # for i, f_k_i in f_k:
    #    f_k[i] = np.max([f_k_i[i], 0.0])

    VA = []
    for v in HC.V:
        if v in bV:
            continue
        else:
            VA.append(v.x_a)

    # compute boundary curvatures:
    VB = []
    dK = []


    for v in bV:
        mean_K_H = []



        for vn in v.nn:
            if not (vn in bV):
            #    print(f'K_H_cache[vn.x] = {K_H_cache[vn.x]}')
                mean_K_H.append(K_H_cache[vn.x])

        #print(f'mean_K_H = {mean_K_H}')

        mean_K_H = geo_mean(mean_K_H)
        k_f_n = K_f / len(bV)
        #dk = K_f - mean_K_H
        dk = k_f_n - mean_K_H
        print(f'mean_K_H = {mean_K_H}')
        print(f'k_f_n = {k_f_n}')
        if print_out:
            print(f'mean_K_H = {mean_K_H}')
            print(f'k_f_n = {k_f_n}')
            print(f'K_f = {K_f}')
            print(f'dk = {dk}')
        #dK.append([v.x[0], v.x[1], 1e-6 * tau*dk])
        #if dk > h:
        #    t = tau * r / (dk + tau)
        #else:
       #     t = tau #* (r / dk)
        #print(f'b t = {t}')
        #dK.append([v.x[0], v.x[1], 1e-6 * tau*dk])
        #dK.append([v.x[0], v.x[1], t*dk])

        # add gravity force
        bheight = np.max([v.x_a[2], 0.0])
        dk = dk - (rho * g * bheight)
        t = tau
    #    print(f'1e-6* t*dk = {1e-6* t*dk}')
    #    print(f'1e-3*1e-6* t*dk = {1e-2*1e-6* t*dk}')
        #dK.append([v.x[0], v.x[1], v.x[2] + 1e-2* 1e-6* t*dk])
        #dK.append([v.x[0], v.x[1], v.x[2] + 1e-1*1e-2* 1e-6* t*dk])
        #dK.append([v.x[0], v.x[1], v.x[2] +  1e-6* t*dk])
        #dK.append([v.x[0], v.x[1], v.x[2] +  1e-3*  t*dk])
        #dK.append([v.x[0], v.x[1], v.x[2] + 1e1 * t*dk/K_f])
        #dK.append([v.x[0], v.x[1], v.x[2] + 1e-2 * t*dk])
        #dK.append([v.x[0], v.x[1], v.x[2] + t*dk])
        dK.append([v.x[0], v.x[1], v.x[2] + t*dk])
        VB.append(v.x)

    #f_k = vg + tau * dK # Previously: f_k = f + tau * df

    # Move interior complex
    VA = np.array(VA)
    if print_out:
        print(f'VA = {VA}')
    for i, v_a in enumerate(VA):
        HC.V.move(HC.V[tuple(v_a)], tuple(f_k[i]))
        if print_out:
            print(f'f_k[i] = {f_k[i]}')

    # Move boundary vertices:
    bV_new = set()
    for i, v_a in enumerate(VB):
        HC.V.move(HC.V[tuple(v_a)], tuple(dK[i]))
        bV_new.add(HC.V[tuple(dK[i])])
        if print_out:
            print(f'dK[i] = {dK[i]}')

    if print_out:
        print(f'bV_new = {bV_new}')
    return HC, bV_new



def film_init(r, h_jurin, refinement=1, test=True):
    """
    Initiate a thin film of radius r
    :param r:
    :return:
    """
    # Initiate a cubical complex
    HC = Complex(3, domain=[(-r, r), ] * 3)
    HC.triangulate()

    for i in range(refinement):
        HC.refine_all()

    # for v in HC2V:
    #    HC.refine_star(v)
    del_list = []
    for v in HC.V:
        if np.any(v.x_a == -r) or np.any(v.x_a == r):
            if (v.x_a[2] == -r):
                continue
            else:
                del_list.append(v)
        else:
            del_list.append(v)

    for v in del_list:
        HC.V.remove(v)

    # Shrink to circle and move to zero
    for v in HC.V:
        x = v.x_a[0]
        y = v.x_a[1]
        f_k = [x * np.sqrt(r ** 2 - y ** 2 / 2.0) / r, y * np.sqrt(r ** 2 - x ** 2 / 2.0) / r, v.x_a[2] + r]
        # f_k = r*(np.array(f_k) - v_o)
        HC.V.move(v, tuple(f_k))

    # Move up to twice the Jurin height
    for v in HC.V:
        f_k = [v.x_a[0], v.x_a[1], v.x_a[2] + 2 * h_jurin]
        HC.V.move(v, tuple(f_k))

    # Find origin and set of boundary vertices
    bV = set()
    for v in HC.V:
        if np.all(v.x_a[0:2] == 0.0):
            continue
        else:
            bV.add(v)

    if test:
        # Ensure the norm is correct
        print(f'Norm test (all should be equal to r = {r}):')
        for v in HC.V:
            print(f'v.x_a = {v.x_a}')
            # print(f'v in bHC_V: {v in bHC_V}')
  #          print(f'norm = {np.linalg.norm(v.x_a - HC.V[(0.0, 0.0, h_0)].x_a)}')
        # Ensure that the boundary doesn't contain the origin
        print(f'Boundary test:')
        for v in bV:
            print(f'v.x_a = {v.x_a}')
    #        if np.linalg.norm(v.x_a - HC.V[(0.0, 0.0, h_0)].x_a) in bV:
   #             print(f'FAILURE: ORIGIN in bHC_V')

    return HC, bV

def mean_flow(HC, bV, h_0, params, tau=0.5):
    """
    Compute a single iteration of mean curvature flow
    :param HC: Simplicial complex
    :return:
    """
    (gamma, rho, g) = params
    print('-')
    # Define the normals (NOTE: ONLY WORKS FOR CONVEX SURFACES)
    # Note that the cache HC.V is an OrderedDict:
    N_i = []

    for v in HC.V:
        # if v in bV:
        #    continue
        N_i.append(normalized(v.x_a - np.array([0.0, 0.0, h_0]))[0])

    f = []
    HNdA = []
    # HNdA_ij_dot = []
    HN = []  # sum(HNdA_ij, axes=0) / C_ijk
    for i, v in enumerate(HC.V):
        if v in bV:
            continue
        print(f'N_i[i] = {N_i[i]}')
        F, nn = vectorise_vnn(v)
        c_outd = curvatures(F, nn, n_i=N_i[i])
        # HNdA.append(0.5*c_outd['HNdA_i'])
        # print(np.sum(c_outd['H_ij_sum']))
        # print(np.sum(c_outd['C_ijk']))
        # HNdA.append(np.sum(c_outd['H_ij_sum'])/np.sum(c_outd['C_ijk']))
        # HNdA_ij_dot.append(np.sum(np.dot(c_outd['HNdA_ij'], c_outd['n_i'])))
        # HNdA.append(N_i[i] * np.sum(c_outd['H_ij_sum'])/np.sum(c_outd['C_ijk']))
        # HNdA.append(np.sum(c_outd['H_ij_sum'])/np.sum(c_outd['C_ijk']))
        HN.append(c_outd['HNdA_i'] / np.sum(c_outd['C_ijk']))
        f.append(v.x_a)

    pass
    # print(f'HNdA = { HNdA}')
    # print(f"(np.sum(c_outd['H_ij_sum'])/np.sum(c_outd['C_ijk']))  = {(np.sum(c_outd['H_ij_sum'])/np.sum(c_outd['C_ijk']))}")
    print(f'HN = {HN}')
    # H = np.nan_to_num(-(1 / 2.0)*np.array(HN))
    H = np.nan_to_num(-np.array(HN))
    df = -gamma * H  # Hydrostatic pressure
    # Add gravity force rho * g * h
    print(f'df(H) = {df}')
    for v in HC.V:
        if (v in bV) or (v.x_a[2] < 0.0):
            continue
        # elif (v.x_a[2] >= h_0):
        else:
            # h = h_0 - v.x_a[2]
            h = v.x_a[2]
            print(f'(rho * g * h) * np.array([0.0, 0.0, -1.0]) = {(rho * g * h) * np.array([0.0, 0.0, -1.0])}')
            df += (rho * g * h) * np.array([0.0, 0.0, -1.0])
            print(f'df += rho g h = {df}')
    # else:
    #     continue

    f = np.array(f)
    f_k = f + tau * df
    VA = []
    for v in HC.V:
        if v in bV:
            continue
        else:
            VA.append(v.x_a)

    VA = np.array(VA)
    print(f'VA = {VA}')
    for i, v_a in enumerate(VA):
        print(f'f_k[i] = {f_k[i]}')
        HC.V.move(HC.V[tuple(v_a)], tuple(f_k[i]))

    return HC

