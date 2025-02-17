#!/usr/bin/env python
# coding: utf-8

# Imports and physical parameters
#import copy
#import sys
import numpy as np
#import polyscope as ps

# Local library
#from ddgclib import *
#from ddgclib._curvatures import HC_curvatures_sessile
#from ddgclib._complex import *
#from ddgclib._sphere import *
#from ddgclib._sessile import *
#from ddgclib._capillary_rise_flow import * #plot_surface#, curvature
#from ddgclib._eos import *
from ddgclib._plotting import plot_polyscope
from ddgclib._bubble import triangle_prism_volume, cone_init, AdamsBashforthProfile, load_complex
from ddgclib._integrators import Euler, AdamsBashforth, NewtonRaphson, lineSearch

# ### Parameters
Bo=0 #Bond number
RadTop = 1e-3 #(Bo*gamma/rho/g)**.5 # m Radius of curvature of bubble top
params = {'P_0': 101.325e3,  # Pa, Ambient pressure
'gamma': 72.8e-3,  # N/m, surface tension
'g': 9.81}  # m/s2 gravitational acceleration
params['rho']= Bo*params['gamma']/params['g']/RadTop**2 #998.2071 - 1.225 # kg/m3, density, STP
print('rho',params['rho'])


# define params tuple used in solver:
params['initial_volume'], RadFoot = AdamsBashforthProfile(Bo, RadTop)
print(f'RadFoot = {RadFoot}')
minEdge = .1*RadFoot
maxEdge = 2*minEdge
maxMove = .5*minEdge**2/RadFoot

print('pressure length',3*RadFoot/params['P_0']/maxEdge**2)
print('tension length',1/params['gamma'])

t=0
if t==0: HC,bV = cone_init(RadFoot, params['initial_volume'], NFoot=6, maxEdge=maxEdge)
else: HC, bV = load_complex(t)
plot_polyscope(HC)
print('vol area centroid', triangle_prism_volume(HC))
t = Euler(HC, bV, params, t, 100, .5*minEdge, minEdge=minEdge, maxEdge=maxEdge, implicitVolume=True, constMoveLen=True)
#plot_polyscope(HC)
t = AdamsBashforth(HC, bV, params, t, 100, .2)
#t = NewtonRaphson(HC, bV, params, t, 20, .1, implicitVolume=True)
#t = lineSearch(HC, bV, params, t, 20, .5*minEdge)
plot_polyscope(HC)
