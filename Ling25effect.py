#Ianto Cannon 2026 Jan 29. Plot the detachment size of pinned and spreading bubbles
import numpy as np
from ddgclib._plotting import plot_polyscope, plot_drop_profile, plot_detach_radius_vs_cont_angle, plot_detach_radius_vs_cont_radius
from ddgclib._bubble import AdamsBashforthProfile, load_complex
contAng = (180-159)*np.pi/180 #radians, angle inside the spherical cap. Set negative for pinned contact line
capLen = 2.7e-3
fname='data/fritz.txt'
with open(fname, "w") as fritz_txt:
  print('saving',fname)
  for b in range(100):
    RadTop = b*capLen/40+.75*capLen
    print(*AdamsBashforthProfile(capLen, RadTop, contAng, fname=f'data/spread{b:05}.txt'), file=fritz_txt)
plot_drop_profile('spread')
