#Ianto Cannon 2026 Feb 4. Find the maximum volume for each contact angle.
import numpy as np
from ddgclib._plotting import plot_drop_height_vs_rad, plot_drops_growing, plot_bubble_coords
from ddgclib._bubble import AdamsBashforthProfile, reorder_drop_height_vs_vol
#plot_bubble_coords()
for b in range(0):#100#7#1,11#20
  RadTop = .1*b+.1
  print(b,'RadTop',RadTop)
  AdamsBashforthProfile(1, RadTop, fname=f'data/bub{b:05}.txt')
#rads=np.array((.5,1,1.5,2,2.5,3,3.5))
#plot_drops_growing(radInd=0, rads=rads)
#angs=np.array((150,120,90,60,30,0))
#angs = np.arange(0.1, 1.1, 0.1)
#plot_drops_growing(radInd=2, rads=angs*np.pi)
#rads=np.array((1.005,1.01,1.05,1.1,1.5))
#for r in range(len(rads)):
#  for b in range(0):#100):
#    print(r,b)
#    RadTop = rads[r]**(b-50)
#    AdamsBashforthProfile(1, RadTop, angleSave=5, radSave=0.1, fname=f'data/bub{100*r+b:05}.txt')
#RadTops = np.concatenate([
#    np.logspace(-2,-.5, 100, endpoint=False),
#    np.logspace(-.5,.5, 200, endpoint=False),
#    np.logspace(.5,  2, 100)
#    ])
RadTops = np.concatenate([
    #np.logspace(-2,            np.log10(.3),  200, endpoint=False),
    #np.logspace(np.log10(.3),  np.log10(1.5), 400, endpoint=False),
    #np.logspace(np.log10(1.5), 2,             100),
    #np.logspace(-2, 2,             1000),
    np.logspace(-2, 2,             100),
    ])
#RadTops = np.logspace(np.log10(.5),np.log10(1.5), 500)
#RadTops = np.logspace(np.log10(.5),np.log10(1.5), 200)
#RadTops = np.linspace(0,4,1000)[1:]
for b in range(len(RadTops)):
#for b in range(0):
  #RadTop = (1+1e-4)**( (b-200) * abs(b-200) )
  RadTop = 1/RadTops[b]
  print(b,RadTop)
  #AdamsBashforthProfile(RadTop, 1, angleSave=.01, radSave=0.1*RadTop, fname=f'data/bub{b:05}.txt')
  AdamsBashforthProfile(1, RadTop, angleSave=.01, radSave=0.1)
reorder_drop_height_vs_vol(nam='ang')
reorder_drop_height_vs_vol(nam='rad')
#plot_drop_height_vs_rad(nam='bub loop_rad loop_ang')
plot_drop_height_vs_rad(nam='loop_rad loop_ang')
for b in range(0):#500
  #RadTop = (1+b)/100
  print(b,RadTop)
  AdamsBashforthProfile(1, RadTop, angleSave=5, radSave=0.1, fname=f'data/bub{b:05}.txt')
RadTops = np.logspace(np.log10(5), 2, 50)
for b in range(0):#len(RadTops)):
  RadTop = RadTops[b]
  print(b,RadTop)
  AdamsBashforthProfile(1, RadTop, fname=f'data/bub{500+b:05}.txt')
#plot_drop_height_vs_rad(nam='bub')
