#Ianto Cannon 2026 Feb 4. Find the maximum volume for each contact angle.
import numpy as np
from ddgclib._plotting import plot_drop_height_vs_rad, plot_drops_growing
from ddgclib._bubble import AdamsBashforthProfile, reorder_drop_height_vs_vol
for b in range(0):#7#1,11#
  RadTop = .1*b+.1
  print(b,'RadTop',RadTop)
  AdamsBashforthProfile(1, RadTop, fname=f'data/bub{b:05}.txt')
  #RadTop = 10**(.5*b)
  #print(b,'RadTop',RadTop)
  #AdamsBashforthProfile(1, RadTop, fname=f'data/bub{b+20:05}.txt')
rads=np.array((.01,.2,.5,1,2,3,4))
plot_drops_growing(radInd=0, rads=rads)
#angs=np.array((175,150,90,30,5))
#angs=np.array((135,90,45))
angs=np.array((165,150,120,90,60,30))
plot_drops_growing(radInd=2, rads=angs*np.pi/180)
#rads=np.array((1.005,1.01,1.05,1.1,1.5))
for r in rads:
  for b in range(0):#100):
    print(r,b)
    RadTop = r**(b-50)
    AdamsBashforthProfile(1, RadTop, angleSave=5, radSave=0.1)
#for b in range(500):#1200):
#  print(b)
#  RadTop = 1.05**(b-100)
#  AdamsBashforthProfile(1, RadTop, angleSave=5, radSave=0.1)
#reorder_drop_height_vs_vol(nam='ang')
#reorder_drop_height_vs_vol(nam='rad')
#plot_drop_height_vs_rad(nam='loop_rad loop_ang bub')
