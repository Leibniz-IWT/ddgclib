#Ianto Cannon 2025 Mar 26. Find the profile of a bubble with Bond number 0.4
import numpy as np
from ddgclib._plotting import plot_polyscope, plot_profile, plot_centroid_vs_iteration
from ddgclib._bubble import AdamsBashforthProfile, load_complex
from ddgclib._integrators import Euler, AdamsBashforth, NewtonRaphson, lineSearch
from ddgclib._volume import cone_init, spherical_cap_init, spherical_cap_contact_angle

#Parameters
Bo=0.4 #Bond number
RadTop = 1 # m, radius of curvature of bubble top
prm = {} # dictionary of parameters
prm['contactAng'] = -1 #radians, angle inside the spherical cap. Set negative for pinned contact line
prm['gamma'] = 1 # N/m, surface tension
prm['gravity'] = 1 # m/s^2 gravitational acceleration
prm['density'] = Bo*prm['gamma']/prm['gravity']/RadTop**2 # kg/m3, bubble density difference
print('density',prm['density'])
prm['targetVol'], RadFoot, height, centroid = AdamsBashforthProfile(Bo, RadTop, .5*np.pi) # m^3
print('targetVol',prm['targetVol'])
print(f'RadFoot = {RadFoot}')
print('height',height)
print('centroid',centroid)
prm['targetPressure'] = 2*prm['gamma']/RadTop #101.325e3 # Pa, Ambient pressure at base
print('targetPressure',prm['targetPressure'])
minEdge = RadTop/8
maxEdge = 2*minEdge

t=0
if t==0: 
  HC,bV = spherical_cap_init(*spherical_cap_contact_angle(RadFoot, prm['targetVol']), maxEdge=maxEdge)
else: 
  HC, bV = load_complex(t)
t = AdamsBashforth(HC, bV, prm, t, 500, .1, maxMove=.5*minEdge, minEdge=minEdge, maxEdge=maxEdge)
plot_profile(t)
plot_centroid_vs_iteration(centroid)
plot_polyscope(HC)
