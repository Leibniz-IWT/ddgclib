#!/usr/bin/env python
# coding: utf-8

import numpy as np
from ddgclib._plotting import plot_polyscope
from ddgclib._bubble import triangle_prism_volume, cone_init, AdamsBashforthProfile, load_complex
from ddgclib._integrators import Euler, AdamsBashforth, NewtonRaphson, lineSearch

# ### Parameters
Bo=-0.4 #Bond number
RadTop = 1e-3 #(Bo*gamma/rho/g)**.5 # m Radius of curvature of bubble top
prm = {}  #dictionary of parameters
prm['gamma'] = 72.8e-3  # N/m, surface tension
prm['g'] = 9.81 # m/s2 gravitational acceleration
prm['rho'] = Bo*prm['gamma']/prm['g']/RadTop**2 #998.2071 - 1.225 # kg/m3, density, STP
print('rho',prm['rho'])

# define prm tuple used in solver:
prm['initial_volume'], RadFoot, height = AdamsBashforthProfile(Bo, RadTop)
print(f'RadFoot = {RadFoot}')
print('initial_volume',prm['initial_volume'])
print('height',height)
prm['P_0'] = 2*prm['gamma']/RadTop - prm['rho']*prm['g']*height #101.325e3 # Pa, Ambient pressure
print('P_0',prm['P_0'])
minEdge = .5*RadFoot
maxEdge = 2*minEdge
#maxMove = .5*minEdge**2/RadFoot

print('pressure length',3*RadFoot/prm['P_0']/maxEdge**2)
print('tension length',1/prm['gamma'])

t=1500
if t==0: HC,bV = cone_init(RadFoot, prm['initial_volume']/5, NFoot=6, maxEdge=maxEdge)
else: HC, bV = load_complex(t)
plot_polyscope(HC)
print('vol area centroid', triangle_prism_volume(HC))
#t = Euler(HC, bV, prm, t, 50, .5*minEdge, minEdge=minEdge, maxEdge=maxEdge, implicitVolume=True, constMoveLen=True)
#plot_polyscope(HC)
minEdge = .2*RadFoot
maxEdge = 2*minEdge
#t = Euler(HC, bV, prm, t, 50, .5*minEdge, minEdge=minEdge, maxEdge=maxEdge, implicitVolume=True, constMoveLen=True)
#plot_polyscope(HC)
t = AdamsBashforth(HC, bV, prm, t, 500, .1, maxMove=.5*minEdge)
#t = NewtonRaphson(HC, bV, prm, t, 100, .5*minEdge)
#t = lineSearch(HC, bV, prm, t, 20, .5*minEdge)
plot_polyscope(HC)
