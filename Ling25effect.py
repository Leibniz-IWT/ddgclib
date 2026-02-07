#Ianto Cannon 2026 Feb 4. Find the maximum volume for each contact angle. RadTop starts at infinity and 
import numpy as np
from ddgclib._plotting import plot_drop_profile
from ddgclib._bubble import AdamsBashforthProfile
contAng = (180-159)*np.pi/180 #radians, angle inside the spherical cap. Set negative for pinned contact line
capLen = 2.7e-3
fname='data/fritz.txt'
maxVol=0
with open(fname, "w") as fritz_txt:
  print('saving',fname)
  for b in range(100):
    RadTop = b*capLen/40+.5*capLen
    output = AdamsBashforthProfile(capLen, RadTop, contAng, fname=f'data/spread{b:05}.txt')
    print(*output, RadTop, file=fritz_txt)
    if output[0]>maxVol:
      maxB=b
      maxVol=output[0]
print('maxVol',maxVol,maxB)
plot_drop_profile('spread')
plot_drop_profile(f'spread{maxB:05}')
