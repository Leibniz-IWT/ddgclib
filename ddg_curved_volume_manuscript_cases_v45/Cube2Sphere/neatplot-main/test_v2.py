import matplotlib.pyplot as plt
plt.style.use('standard.mplstyle')   # or the full path to the file

import numpy as np

# Data (from the table)
surface_meshes = np.array([18, 179, 72, 104, 124, 116, 124, 104])

# Relative error of total volume (V_tetra + V_patch) vs V_theory  [%]
v_tetra_patch = np.array([8.8e-15, 4.0e-6, 3.3e-5, 9.3e-5, 5.9e-3, 3.8e-3, 5.9e-3, 9.5e-5])

# Relative error of V_tetra only vs V_theory  [%] (signed)
v_tetra = np.array([39.45, 1.07, 0.52, 1.23, 0.91, 4.45, 0.91, 1.23])

plt.figure(figsize=(8, 5))

# V_tetra + V_patch (already positive; log scale OK)
plt.scatter(
    surface_meshes, v_tetra_patch,
    marker='o',
    label=r"$\sum_{i=1}^{N} V_{\mathrm{tetra},i}+\sum_{i=1}^{n} V_{\mathrm{patch},i}$"
)

# V_tetra only (use abs for log scale)
plt.scatter(
    surface_meshes, np.abs(v_tetra),
    marker='s',
    label=r"$\sum_{i=1}^{N} V_{\mathrm{tetra},i}$ only"
)

plt.yscale("log")
plt.xlabel("No. of surface meshes")
plt.ylabel("Relative Error (%)")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.6)

plt.show()
