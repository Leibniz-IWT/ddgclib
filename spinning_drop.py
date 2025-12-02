#Ianto Cannon 2025 Dec 1. Plot the profile of a levitating spinning drop, photographed in Fauzia Wardani's gallery of fluid motion at APS DFD.
import numpy as np
from ddgclib._plotting import plot_drop_profile 
from ddgclib._bubble import spinning_drop_profile, spinning_drop_from_neck
rad_top = 1 #m radius of curvature at the tip
height = 1.9 #m distance of the tip from the axis of rotation
rot_len = 1 # (surf_tens / density / rot**2 ) **(1/3)
#rad_neck = 1 #m radius of the neck
#rad_c_neck = 1 #m radius of curvature the neck
vol, height, centroid, *_ = spinning_drop_profile(rot_len, rad_top, height, fname=f'data/spin.txt')
#vol, height, centroid, *_ = spinning_drop_from_neck(rot_len, rad_neck, rad_c_neck, fname=f'data/neck.txt')
plot_drop_profile('spin')
surf_tens = 72e-3 #N/m, surface tension
#rot = 1 #radians/sec rotation rate
density = 1e3 #kg*m**-3
rot = ( surf_tens / density / (rot_len*5e-4)**3 )**.5
print('rot',rot)
