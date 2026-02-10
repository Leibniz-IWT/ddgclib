# Imports and physical parameters

# ddg imports
import os, sys
module_path = os.path.abspath(os.path.join('../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from ddgclib._plotting import *
from ddgclib._sphere import *
#from ddgclib._case2 import *

# Numerical parameters #Stated this is what to pla
N = 7  # Determines mesh incidence
refinement = 2

r = 1  # analytical area = 2 pi r**2, so take 2 pi
HC = sphere_from_cap(r, N=N, refinement=refinement)

# Compute the curvature
HNdA_i = []  # total HNdA_i at vertex i
HNdA_ij = []  # total HNdA_i for edge ij
HN_i = []  # point-wise
C_ij = []
alpha_ij = []
for v in HC.V:
    #print(f'v.index = {v.index}')
    N_f0 = v.x_a - np.array([0.0, 0.0, 0.0])  # First approximation
    N_f0 = normalized(N_f0)[0]
    F, nn = vectorise_vnn(v)
    # Compute discrete curvatures
    c_outd = b_curvatures_hn_ij_c_ij(F, nn, n_i=N_f0)
    # Append lists
    c_outd['HNdA_ij']
    HNdA_ij.append(c_outd['HNdA_ij'])
    HNdA_i.append(c_outd['HNdA_i'])
    HN_i.append(c_outd['HN_i'])
    C_ij.append(c_outd['C_ij'])
    alpha_ij.append(c_outd['alpha_ij'])
    # Theta_i_jk.append(c_outd['Theta_i_jk'])
    # break

# [0]
# HNdA_i = 0.5 * np.sum(HNdA_ij, axis=0)
# HN_i = np.sum(HNdA_i) / np.sum(C_ij)
HNdA_ij, HN_i, HNdA_i, C_ij, alpha_ij
print(f'np.sum(HNdA_i, axis=0) = {np.sum(HNdA_i, axis=0)}')
print(f'HNdA_i[0] = {HNdA_i[0]}')
print(f'nn = {nn}')
print(f'F= {F}')

# Now we compute the surface areas
#for v in HC.V:
#    HNdC_ijk(e_ij, l_ij, l_jk, l_ik)


if 1:
    HC.plot_complex()
    plt.show()




# Plot conforimation
if 1:
    ps = plot_polyscope(HC, vector_field=None, scalar_field=None, fn='', up="x_up"
                        , stl=False)
    ps.show()