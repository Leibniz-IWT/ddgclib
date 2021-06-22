# NOTE: CP is used for the Equation of State:
from CoolProp.CoolProp import PropsSI
# Imports and physical parameters
import numpy as np
import scipy
#from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage import data, io
import sys
import skimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from ddgclib import *
from ddgclib._complex import Complex


# Colour scheme for surfaces
db = np.array([129, 160, 189]) / 255  # Dark blue
lb = np.array([176, 206, 234]) / 255  # Light blue

# Parameters for a water droplet in air at standard laboratory conditions
#gamma = 0.0728  # N/m, surface tension of water at 20 deg C
gamma =71.03e-3  # N/m kg/s^2 = N m-1 surface tension of water used in SE
Volmue = 24.8e-6  # Volume (0units unknown (mL?) 0.01 mL = 0.01 cm^3)
Volume = 24.8e-9  # Volume L --> m3 (30.7 uL)
rho = 998   # kg/m3, density
g = 9.81  # m/s2
# Capillary rise parameters
theta_p = 69.4  * np.pi /180.0

#from ddgclib.curvatures import plot_surface, curvature

# Colour scheme for surfaces
db = np.array([129, 160, 189]) / 255  # Dark blue
lb = np.array([176, 206, 234]) / 255  # Light blue


#!conda install --yes --prefix {sys.prefix} skimage


# Parameters for a water droplet in air at standard laboratory conditions
def eos(P=101.325, T=298.15):
    # P in kPa T in K
    return PropsSI('D','T|liquid',298.15,'P',101.325,'Water') # density kg /m3

# Surface tension of water gamma(T)
def IAPWS(T=298.15):
    T_C = 647.096  # K, critical temperature
    return 235.8e-3 * (1 -(T/T_C))**1.256 * (1 - 0.625*(1 - (T/T_C)))  # N/m

T_0 = 273.15 + 25  # K, initial tmeperature
P_0 = 101.325  # kPa, Ambient pressure
gamma = IAPWS(T_0)  # N/m, surface tension of water at 20 deg C
rho_0 = eos(P=P_0, T=T_0)  # kg/m3, density
g = 9.81  # m/s2

from ddgclib import *
from ddgclib._complex import Complex
#from ddgclib.curvatures import plot_surface, curvature

# Colour scheme for surfaces
db = np.array([129, 160, 189]) / 255  # Dark blue
lb = np.array([176, 206, 234]) / 255  # Light blue