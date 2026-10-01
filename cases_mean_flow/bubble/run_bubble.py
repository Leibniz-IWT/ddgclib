#Ianto Cannon 2025 Mar 26. Find the profile of a bubble with Bond number 0.4
import numpy as np
#IC 2026 Oct 1: ddgclib is two dirs up 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2])) 
from ddgclib._plotting import plot_polyscope, plot_profile, plot_centroid_vs_iteration
from ddgclib._bubble import AdamsBashforthProfile, load_complex
from ddgclib.mean_flow_integrators._integrators_mean_flow import AdamsBashforth
from ddgclib.geometry._volume import spherical_cap_init, spherical_cap_contact_angle

#Parameters
Bo=0.4 #Bond number
RadTop = 1 # m, radius of curvature of bubble top
prm = {} # dictionary of parameters
prm['contactAng'] = -1 #radians, angle inside the spherical cap. Set negative for pinned contact line
prm['gamma'] = 1 # N/m, surface tension
prm['gravity'] = 1 # m/s^2 gravitational acceleration
prm['density'] = Bo*prm['gamma']/prm['gravity']/RadTop**2 # kg/m3, bubble density difference
print('density',prm['density'])
prm['targetVol'], RadFoot, height, centroid, psi = AdamsBashforthProfile(Bo, RadTop, .5*np.pi) # m^3
print('targetVol',prm['targetVol'])
print(f'RadFoot = {RadFoot}')
print('height',height)
print('centroid',centroid)
prm['targetPressure'] = 2*prm['gamma']/RadTop #101.325e3 # Pa, Ambient pressure at base
print('targetPressure',prm['targetPressure'])
minEdge = RadTop/8
maxEdge = 2*minEdge


d = 0.0001
psi = 0
r = 0
z = 0
Volume = 0
fname = 'data/adams' + str(Bo) + '.txt'
with open(fname, "w") as adams_txt:
    print('saving', fname)
    for i in range(int(4 / d)):
        r += d * np.cos(psi)
        dz = d * np.sin(psi)
        z += dz
        Volume += np.pi * r**2 * dz
        if i * d * 100 % 1 == 0:
            print(r * RadTop, -z * RadTop, file=adams_txt)
        psi += d * (2 - Bo * z - np.sin(psi) / r)
        if psi > np.pi / 2:
            break
        if psi > np.pi:
            break
        if psi < np.pi / 2 and 2 - Bo * z - np.sin(psi) / r < 0:
            break


t=0
if t==0: 
  HC,bV = spherical_cap_init(*spherical_cap_contact_angle(RadFoot, prm['targetVol']), maxEdge=maxEdge)
else: 
  HC, bV = load_complex(t)
t = AdamsBashforth(HC, bV, prm, t, 500, .1, maxMove=.5*minEdge, minEdge=minEdge, maxEdge=maxEdge)
plot_profile(t)
plot_centroid_vs_iteration(centroid)
plot_polyscope(HC)
