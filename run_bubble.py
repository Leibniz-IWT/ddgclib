#!/usr/bin/env python
# coding: utf-8
import numpy as np
from ddgclib._plotting import plot_polyscope
from ddgclib._bubble import AdamsBashforthProfile, load_complex
from ddgclib._integrators import Euler, AdamsBashforth, NewtonRaphson, lineSearch
from ddgclib._volume import cone_init, spherical_cap_init

#Parameters
Bo=0 #Bond number
RadTop = 1e-3 #(Bo*gamma/rho/g)**.5 # m Radius of curvature of bubble top
prm = {}  #dictionary of parameters
prm['gamma'] = 72.8e-3  # N/m, surface tension
prm['g'] = 9.81 # m/s2 gravitational acceleration
prm['rho'] = Bo*prm['gamma']/prm['g']/RadTop**2 #998.2071 - 1.225 # kg/m3, density, STP
print('rho',prm['rho'])
prm['initial_volume'], RadFoot, height, centroid = AdamsBashforthProfile(Bo, RadTop)
print(f'RadFoot = {RadFoot}')
#prm['initial_volume'] = 2*np.pi/3*RadTop**3
prm['initial_volume'] = 2*prm['initial_volume']
RadFoot = RadTop
print('initial_volume',prm['initial_volume'])
print('height',height)
print('centroid',centroid)
prm['P_0'] = 2*prm['gamma']/RadTop #101.325e3 # Pa, Ambient pressure at base
print('P_0',prm['P_0'])
minEdge = .2*RadTop
maxEdge = 2*minEdge
print('pressure length',3*RadFoot/prm['P_0']/maxEdge**2)
print('tension length',1/prm['gamma'])

t=0
if t==0: 
  HC,bV = spherical_cap_init(2*RadFoot, prm['initial_volume'], maxEdge=maxEdge)
  plot_polyscope(HC)
  plot_polyscope(HC)
  t = Euler(HC, bV, prm, t, 100, .5*minEdge, minEdge=minEdge, maxEdge=maxEdge, implicitVolume=True, constMoveLen=True)
  #plot_polyscope(HC)
  #minEdge = .2*RadFoot
  #maxEdge = 2*minEdge
  #t = Euler(HC, bV, prm, t, 500, .5*minEdge, minEdge=minEdge, maxEdge=maxEdge, implicitVolume=True, constMoveLen=True)
else: 
  HC, bV = load_complex(t)
plot_polyscope(HC)
#t = Euler(HC, bV, prm, t, 100, .5*minEdge, minEdge=minEdge, maxEdge=maxEdge, implicitVolume=True, constMoveLen=True)
t = AdamsBashforth(HC, bV, prm, t, 50, 0, maxMove=.5*minEdge, minEdge=minEdge, maxEdge=maxEdge)
plot_polyscope(HC)
t = AdamsBashforth(HC, bV, prm, t, 100, .1, maxMove=.5*minEdge, minEdge=minEdge, maxEdge=maxEdge)
plot_polyscope(HC)
