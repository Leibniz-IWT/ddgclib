#Ianto Cannon 2026 Feb 4. Find the maximum volume for each contact angle.
import numpy as np
from ddgclib._plotting import plot_drop_vol_vs_rad_top, plot_drop_profile
from ddgclib._bubble import AdamsBashforthProfile
for b in range(400):
  RadTop = 1.01**(b-100)
  AdamsBashforthProfile(1, RadTop, fname='spread', angleSave=10)
plot_drop_vol_vs_rad_top('angle')
plot_drop_profile('spread')
