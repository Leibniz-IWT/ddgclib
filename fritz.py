#Ianto Cannon 2025 Jun 2. Plot the detachment size of pinned and spreading bubbles
import numpy as np
from ddgclib._plotting import plot_polyscope, plot_profile, plot_centroid_vs_iteration_compare
from ddgclib._bubble import AdamsBashforthProfile, load_complex

#Parameters
RadTop = 1 # m, radius of curvature of bubble top
prm = {} # dictionary of parameters
prm['contactAng'] = -1 #radians, angle inside the spherical cap. Set negative for pinned contact line
prm['gamma'] = 1 # N/m, surface tension
prm['gravity'] = 1 # m/s^2 gravitational acceleration
fname='data/fritz.txt'
with open(fname, "w") as fritz_txt:
  print('saving',fname)
  for b in range(1,10):
    Bo=b/10
    prm['density'] = Bo*prm['gamma']/prm['gravity']/RadTop**2 # kg/m3, bubble density difference
    print('density',prm['density'])
    capiLen = ( prm['gamma'] / prm['density'] / prm['gravity'] )**.5
    VPin, RadFootPin, heightPin, centroidPin, anglePin = AdamsBashforthProfile(Bo, RadTop, .5*np.pi) # m^3
    RadPin = (3*VPin/4/np.pi)**(1/3)
    VSpr, RadFootSpr, heightSpr, centroidSpr, angleSpr = AdamsBashforthProfile(Bo, RadTop) # m^3
    RadSpread = (3*VSpr/4/np.pi)**(1/3)
    print(Bo, angleSpr, RadSpread/capiLen, RadFootPin/capiLen, RadPin/capiLen, file=fritz_txt)
#plot_centroid_vs_iteration_compare(centroid)
