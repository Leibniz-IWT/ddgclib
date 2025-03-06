#Ianto Cannon 2025 Feb 20
#Make complexes and measure their volume

import numpy as np
# Allow for relative imports from main library:
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from ddgclib._volume import spherical_cap_init, spherical_cap_contact_angle, cone_init, sector_volume, prism_volume, triangle_prism_volume

print('make conical complex')
HC,bV = cone_init(1, 1, maxEdge=.2)
print('sector_volume error ', sector_volume(HC)-1)
print('prism_volume error ', prism_volume(HC)-1)
print('triangle_prism_volume error ', triangle_prism_volume(HC)[0]-1)

print('')
print('make spherical cap complex')
RadSphere, contactAng = spherical_cap_contact_angle(.1 ,1)
HC,bV = spherical_cap_init(RadSphere, contactAng, maxEdge=.2)
print('sector_volume error ', sector_volume(HC)-1)
print('prism_volume error ', prism_volume(HC)-1)
print('triangle_prism_volume error ', triangle_prism_volume(HC)[0]-1)
