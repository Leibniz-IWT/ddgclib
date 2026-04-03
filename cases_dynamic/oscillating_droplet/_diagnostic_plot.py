"""Diagnostic visualization of the droplet-in-box mesh construction."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ddgclib.geometry.domains import droplet_in_box_2d, droplet_in_box_3d

os.makedirs('fig', exist_ok=True)
R0 = 0.01


def plot_2d():
    result = droplet_in_box_2d(R=R0, L=0.05, refinement_outer=3,
                                refinement_droplet=3)
    HC = result.HC

    n_total = sum(1 for _ in HC.V)
    n_p0 = sum(1 for v in HC.V if v.phase == 0)
    n_p1 = sum(1 for v in HC.V if v.phase == 1)
    n_iface = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
    iface_rs = [np.linalg.norm(v.x_a[:2])
                for v in HC.V if getattr(v, 'is_interface', False)]
    print(f'2D: {n_total} verts ({n_p0} outer, {n_p1} droplet, '
          f'{n_iface} interface)')
    if iface_rs:
        print(f'  Interface r: min={min(iface_rs):.6f}, '
              f'max={max(iface_rs):.6f}, R0={R0}')

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Full mesh
    ax = axes[0]
    for v in HC.V:
        for nb in v.nn:
            if id(v) < id(nb):
                ax.plot([v.x_a[0], nb.x_a[0]], [v.x_a[1], nb.x_a[1]],
                        'gray', lw=0.3, alpha=0.4)
    for v in HC.V:
        is_if = getattr(v, 'is_interface', False)
        if is_if:
            c, s = 'red', 15
        elif v.phase == 1:
            c, s = 'dodgerblue', 6
        else:
            c, s = 'lightgreen', 6
        ax.scatter(v.x_a[0], v.x_a[1], c=c, s=s, zorder=5, edgecolors='none')
    ax.add_patch(plt.Circle((0, 0), R0, fill=False, color='k', ls='--', lw=1.5))
    ax.set_aspect('equal')
    ax.set_title(f'Full mesh ({n_total} verts)')

    # Panel 2: Zoom to interface
    ax = axes[1]
    zoom = 1.8 * R0
    for v in HC.V:
        if np.linalg.norm(v.x_a[:2]) > zoom:
            continue
        for nb in v.nn:
            if np.linalg.norm(nb.x_a[:2]) > zoom:
                continue
            if id(v) < id(nb):
                cross_phase = (v.phase != nb.phase)
                col = 'orange' if cross_phase else 'gray'
                lw = 1.0 if cross_phase else 0.4
                ax.plot([v.x_a[0], nb.x_a[0]], [v.x_a[1], nb.x_a[1]],
                        col, lw=lw, alpha=0.7)
    for v in HC.V:
        if np.linalg.norm(v.x_a[:2]) > zoom:
            continue
        is_if = getattr(v, 'is_interface', False)
        if is_if:
            c, s = 'red', 30
        elif v.phase == 1:
            c, s = 'dodgerblue', 15
        else:
            c, s = 'lightgreen', 20
        ax.scatter(v.x_a[0], v.x_a[1], c=c, s=s, zorder=5, edgecolors='none')
    ax.add_patch(plt.Circle((0, 0), R0, fill=False, color='k', ls='--', lw=1.5))
    ax.set_aspect('equal')
    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-zoom, zoom)
    ax.set_title('Interface zoom (orange = cross-phase edges)')

    # Panel 3: Radial distribution
    ax = axes[2]
    for v in HC.V:
        r = np.linalg.norm(v.x_a[:2])
        is_if = getattr(v, 'is_interface', False)
        if is_if:
            c = 'red'
        elif v.phase == 1:
            c = 'dodgerblue'
        else:
            c = 'lightgreen'
        ax.scatter(r, v.phase + np.random.uniform(-0.1, 0.1),
                   c=c, s=10, alpha=0.5)
    ax.axvline(R0, color='k', ls='--', lw=1.5, label=f'R0={R0}')
    ax.set_xlabel('r (distance from center)')
    ax.set_ylabel('Phase (jittered)')
    ax.set_title('Phase vs radius')
    ax.legend()

    fig.suptitle('2D Droplet-in-Box Mesh', fontsize=14)
    fig.tight_layout()
    fig.savefig('fig/droplet_mesh_2D_diagnostic.png', dpi=200)
    print('Saved fig/droplet_mesh_2D_diagnostic.png')
    plt.close()


def plot_3d():
    result = droplet_in_box_3d(R=R0, L=0.05, refinement_outer=2,
                                refinement_droplet=2)
    HC = result.HC

    n_total = sum(1 for _ in HC.V)
    n_p0 = sum(1 for v in HC.V if v.phase == 0)
    n_p1 = sum(1 for v in HC.V if v.phase == 1)
    n_iface = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
    print(f'3D: {n_total} verts ({n_p0} outer, {n_p1} droplet, '
          f'{n_iface} interface)')

    fig = plt.figure(figsize=(14, 6))

    # Panel 1: 3D scatter
    ax = fig.add_subplot(121, projection='3d')
    for v in HC.V:
        is_if = getattr(v, 'is_interface', False)
        if is_if:
            c, s = 'red', 20
        elif v.phase == 1:
            c, s = 'dodgerblue', 5
        else:
            c, s = 'lightgreen', 5
        ax.scatter(v.x_a[0], v.x_a[1], v.x_a[2], c=c, s=s, alpha=0.6)
    ax.set_title(f'3D mesh ({n_total} verts)')

    # Panel 2: Radial distribution
    ax2 = fig.add_subplot(122)
    for v in HC.V:
        r = np.linalg.norm(v.x_a[:3])
        is_if = getattr(v, 'is_interface', False)
        if is_if:
            c = 'red'
        elif v.phase == 1:
            c = 'dodgerblue'
        else:
            c = 'lightgreen'
        ax2.scatter(r, v.phase + np.random.uniform(-0.1, 0.1),
                    c=c, s=10, alpha=0.5)
    ax2.axvline(R0, color='k', ls='--', lw=1.5, label=f'R0={R0}')
    ax2.set_xlabel('r')
    ax2.set_ylabel('Phase (jittered)')
    ax2.set_title('Phase vs radius')
    ax2.legend()

    fig.suptitle('3D Droplet-in-Box Mesh', fontsize=14)
    fig.tight_layout()
    fig.savefig('fig/droplet_mesh_3D_diagnostic.png', dpi=200)
    print('Saved fig/droplet_mesh_3D_diagnostic.png')
    plt.close()


if __name__ == '__main__':
    plot_2d()
    plot_3d()
