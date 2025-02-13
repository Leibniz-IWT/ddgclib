#!/usr/bin/env python
# coding: utf-8

# Imports and physical parameters
#import copy
#import sys
import numpy as np
import polyscope as ps

# Local library
#from ddgclib import *
#from ddgclib._curvatures import HC_curvatures_sessile
#from ddgclib._complex import *
#from ddgclib._sphere import *
#from ddgclib._sessile import *
#from ddgclib._capillary_rise_flow import * #plot_surface#, curvature
#from ddgclib._eos import *
from ddgclib._plotting import plot_polyscope
from ddgclib._bubble import triangle_prism_volume, cone_init
from ddgclib._integrators import Euler


# ### Parameters
Bo=0 #Bond number
P_0 = 101.325e3  # Pa, Ambient pressure
gamma = 72.8e-3  # N/m, surface tension
g = 9.81  # m/s2 gravitational acceleration
RadTop = 1e-3 #(Bo*gamma/rho/g)**.5 # m Radius of curvature of bubble top
rho = Bo*gamma/g/RadTop**2 #998.2071 - 1.225 # kg/m3, density, STP
print('rho',rho)
theta_p = np.pi/2 #contact angle

#compute analytical interface shape according to eq 1 of Demirkir2024Langmuir
d=.0001
psi=0
r=0
z=0
Volume=0
fname='data/adams'+str(Bo)+'.txt'
with open(fname, "w") as adams_txt:
  print('saving',fname)
  for i in range(int(4/d)):
    r += d * np.cos(psi)
    dz = d * np.sin(psi)
    z += dz
    Volume += np.pi*r**2*dz
    if i*d*100%1 == 0: print(r*RadTop, -z*RadTop, file=adams_txt)
    psi += d * (2 - Bo*z - np.sin(psi)/r)
    #if 2 - Bo*z - np.sin(psi)/r < 0: break
    #if z < -.4: break
    if psi > np.pi/2: break
    if psi > np.pi: break
    if psi < np.pi/2 and 2 - Bo*z - np.sin(psi)/r < 0: break

# define params tuple used in solver:
Volume = Volume*RadTop**3
RadFoot = r*RadTop
minEdge = .1*RadFoot
maxEdge = 2*minEdge
maxMove = .5*minEdge**2/RadFoot
print(f'RadFoot = {RadFoot}')
tInit=0
t=tInit
if tInit==0: HC,bV = cone_init(RadFoot, Volume, NFoot=6)
else: HC, bV = load_complex(t)
print('vol area centroid', triangle_prism_volume(HC))
plot_polyscope(HC)
initial_volume = Volume
Euler(HC, tInit, 300, 0.1)
plot_polyscope(HC)
plt.show()
