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
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from ddgclib._particle_liquid_bridge_initial_truncated_cone_flo import *
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt

#set parameters for plots
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.default"] = "it"
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

# refinement=3# NOTE: 2 is the minimum refinement needed for the complex to be manifold


#gamma = 0.0728  # N/m, surface tension of water at 20 deg C

gamma = 728e-5  # N/mm, surface tension of water at 20 deg C
refinement = 4


diameter_l  = 0.8     # diameter on the lower side of the cone in m
diameter_u  = 1.6        # diameter on the upper side of the cone in m

length      = 0.7     # length of the initial cylinder

t_f         = 25000     # End-Residual-Time // if t_f = 0, no optimization
tau         = 0.5   # step-size

save_name = 'Complex_tf25000_tau05_refinment4.json'

#%% ---------------------------------


complex = fun_liquid_bridge_truncated_cone_N(diameter_l, diameter_u, length, refinement, tau, t_f, gamma)

# return(HC, iteration_list, df_list_min, df_list_max,res_list, starttime, endtime)

HC              = complex[0]
iteration_list  = complex[1]
df_list_min     = complex[2]
df_list_max     = complex[3]
res_list        = complex[4]

print(f"Time elapsed: {complex[6]-complex[5]:.2f} s")

#%%

save_complex(HC,save_name)

time  = []
time.append(complex[6] - complex[5])

np.savetxt('time.txt',time)

np.savetxt('df_list_min.txt', df_list_min)
np.savetxt('res_list.txt', res_list)
np.savetxt('df_list_max.txt', df_list_max)
np.savetxt('iteration_list.txt', iteration_list)

#%% ---------------------------------


fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(12, 6))


ax[0].plot(iteration_list, df_list_max)
ax[0].plot(iteration_list, df_list_min)

ax[1].plot(iteration_list, res_list)
#ax[1].plot(iteration_list, dHNdA_list_min)

ax[0].set_yscale('log')


ax[0].set_title('absolute for df')
ax[1].set_title('Residuals for df')

fig.savefig('Residuals.png')