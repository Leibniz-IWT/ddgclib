import matplotlib.pyplot as plt
plt.style.use('standard.mplstyle')   # or the full path to the file
 
import numpy as np

# Data
surface_meshes = np.array([18, 179, 72, 104, 124, 116, 124, 104])
v_tetra_patch = np.array([0.000004, 0.000004, 0.001000, 0.000100, 0.00590, 0.030000, 0.00590, 0.000100])
v_tetra = np.array([-39.45, -1.07, -0.52, -1.23, -0.91, -4.45, -0.91, -1.23])

plt.figure(figsize=(8,5))

# V_tetra + V_patch
plt.scatter(surface_meshes, v_tetra_patch, marker='o', label=r"$\sum_{i=1}^{N} V_{\text{tetra},i}+\sum_{i=1}^{n} V_{\text{patch},i}$")

# V_tetra (absolute, since log scale)
plt.scatter(surface_meshes, np.abs(v_tetra), marker='s', label=r"$\sum_{i=1}^{N} V_{\text{tetra},i}$ only")

plt.yscale("log")  # log y-axis
plt.xlabel("No. of surface meshes")
plt.ylabel("Relative Error (%)")
#plt.title("Relative Error vs. Surface Mesh Count")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.6)

plt.show()
