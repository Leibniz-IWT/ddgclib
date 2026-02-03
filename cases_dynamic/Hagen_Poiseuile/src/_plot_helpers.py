
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def extract_radial_profiles(HC, z_positions, dr_tol=0.05):
    """Extract u_z vs r at specified z slices."""
    profiles = {}
    for z_target in z_positions:
        r_list = []
        uz_list = []
        for v in HC.V:
            if abs(v.x_a[2] - z_target) < dr_tol:
                r = np.linalg.norm(v.x_a[:2])
                uz = v.u[2]
                r_list.append(r)
                uz_list.append(uz)
        if r_list:
            sort_idx = np.argsort(r_list)
            profiles[z_target] = (np.array(r_list)[sort_idx], np.array(uz_list)[sort_idx])
    return profiles


def plot_radial_profiles(profiles, U_max, r_max, title="Radial velocity profiles"):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(profiles)))

    r_anal = np.linspace(0, r_max, 200)
    uz_anal = U_max * (1 - (r_anal / r_max) ** 2)
    plt.plot(r_anal, uz_anal, 'k--', lw=2, label='Analytical (fully developed)')

    for i, (z, (r, uz)) in enumerate(profiles.items()):
        plt.plot(r, uz, 'o-', color=colors[i], label=f'z = {z:.1f} m', markersize=4)

    plt.xlabel('Radial distance r [m]')
    plt.ylabel('Axial velocity u_z [m/s]')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, r_max)
    plt.ylim(0, U_max * 1.05)
    plt.tight_layout()
    plt.show()


def plot_centerline_velocity(HC, L, U_max, title="Centerline velocity development"):
    z_slices = np.linspace(-L / 2 + 0.1, L / 2 - 0.1, 30)
    z_centers = []
    u_center = []

    for z in z_slices:
        uz_max = 0.0
        count = 0
        for v in HC.V:
            if abs(v.x_a[2] - z) < 0.08:
                uz_max = max(uz_max, v.u[2])
                count += 1
        if count > 0:
            z_centers.append(z)
            u_center.append(uz_max)

    plt.figure(figsize=(10, 5))
    plt.plot(z_centers, u_center, 'b.-', label='Simulated centerline u_z')
    plt.axhline(U_max, color='k', linestyle='--', label=f'U_max = {U_max:.3f} m/s (analytical)')
    plt.xlabel('Axial position z [m]')
    plt.ylabel('Centerline velocity u_z [m/s]')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_3d_velocity(HC, cmap='viridis', s=15, alpha=0.8):
    """Safe 3D scatter with explicit float64 casting."""
    x = np.asarray([v.x_a[0] for v in HC.V], dtype=np.float64)
    y = np.asarray([v.x_a[1] for v in HC.V], dtype=np.float64)
    z = np.asarray([v.x_a[2] for v in HC.V], dtype=np.float64)
    uz = np.asarray([v.u[2] for v in HC.V], dtype=np.float64)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=uz, cmap=cmap, s=s, alpha=alpha)
    plt.colorbar(scatter, label='u_z [m/s]')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('Axial velocity distribution')
    plt.show()
    return fig, ax