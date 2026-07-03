import copy
from ._curvatures import *

def willmore_flow(HC, tau=0.5):
    """
    Compute a single iteration of Willmore curvature flow
    :param HC: Simplicial complex
    :return:
    """
    # Define the normals (NOTE: ONLY WORKS FOR CONVEX SURFACES)
    # Note that the cache HC.V is an OrderedDict:
    N_i = []

    for v in HC.V:
        N_i.append(normalized(v.x_a - np.array([0.5, 0.5, 0.5]))[0])

    f = []
    W = []
    for i, v in enumerate(HC.V):
        F, nn = vectorise_vnn(v)
        c_outd = curvatures(F, nn, n_i=N_i[i])
        W.append(np.linalg.norm(c_outd['HNdA_i'])/c_outd['C_ijk'])
        f.append(v.x_a)

    df = np.nan_to_num(np.array(W))
    f = np.array(f)
    tau = 0.5
    f_k = f + tau * df
    VA = []
    for v in HC.V:
        VA.append(v.x_a)

    VA = np.array(VA)
    for i, v_a in enumerate(VA):
        HC.V.move(HC.V[tuple(v_a)], tuple(f_k[i]))

    return HC

def willmore_flow_old(HC, tau=0.5):
    """
    Compute a single iteration of Willmore curvature flow
    :param HC: Simplicial complex
    :return:
    """
    # Define the normals (NOTE: ONLY WORKS FOR CONVEX SURFACES)
    # Note that the cache HC.V is an OrderedDict:
    N_i = []

    for v in HC.V:
        N_i.append(normalized(v.x_a - np.array([0.5, 0.5, 0.5]))[0])

    f = []
    W = []
    for i, v in enumerate(HC.V):
        F, nn = vectorise_vnn(v)
        c_outd = curvatures(F, nn, n_i=N_i[i])
        W.append(np.linalg.norm(c_outd['HNdA_i'])/c_outd['C_ijk'])
        f.append(v.x_a)

    df = np.nan_to_num(np.array(W))
    f = np.array(f)
    tau = 0.5
    f_k = f + tau * df
    VA = []
    for v in HC.V:
        VA.append(v.x_a)

    VA = np.array(VA)
    for i, v_a in enumerate(VA):
        HC.V.move(HC.V[tuple(v_a)], tuple(f_k[i]))

    return HC

def mean_flow(HC, tau=0.5):
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
        HNdA.append(c_outd['HNdA_i'])
        f.append(v.x_a)

    df = np.nan_to_num(np.array(HNdA))
    f = np.array(f)
    f_k = f + tau * df
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