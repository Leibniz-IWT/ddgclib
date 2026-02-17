#Ianto Cannon 2026 Feb 4. Find the maximum volume for each contact angle.
import numpy as np
from ddgclib._plotting import plot_drop_size_vs_rad, plot_drop_size_vs_ang, plot_drop_profile, plot_drop_growing
from ddgclib._bubble import AdamsBashforthProfile, reorder_drop_height_vs_vol
for b in range(1200):#40#1200
  RadTop = 1.01**(b-400)#400
  #RadTop = 1.01**(b-20)
  AdamsBashforthProfile(1, RadTop, angleSave=5, radSave=0.1, fname=f'data/bub{b:05}.txt')
  #AdamsBashforthProfile(1, RadTop, angleSave=5)
reorder_drop_height_vs_vol(nam='ang')
reorder_drop_height_vs_vol(nam='rad')
plot_drop_size_vs_ang()
plot_drop_size_vs_rad()
plot_drop_profile(name='bub', angleSave=(5,20,90,170), radSave=())
plot_drop_growing(radInd=0, rads=(0.5,1,1.5,2))
angs=np.array((5,20,90,170))
plot_drop_growing(radInd=2, rads=angs*np.pi/180)
