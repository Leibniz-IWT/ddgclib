import numpy as np
from matplotlib import pyplot as plt


def plot_dual_mesh_2D(HC, tri, points):
    """
    Plot the dual mesh and show edge connectivity. Blue is the primary mesh. Orange is the dual mesh.

    Note: Original points need to be fed that was used in the order to build `tri`
    """
    import matplotlib.pyplot as plt

    # Find the dual points
    dual_points = []
    for vd in HC.Vd:
        dual_points.append(vd.x_a)
    dual_points = np.array(dual_points)
    # Primal points
    # points = []
    # for v in HC.V:
    #    points.append(v.x_a)

    # points = np.array(points)

    for v in HC.V:
        # "Connect duals":
        for v2 in v.nn:
            v1vdv2vd = v.vd.intersection(v2.vd)  # Cardinality always 1 or 2?
            if len(v1vdv2vd) == 1:
                continue
            v1vdv2vd = list(v1vdv2vd)
            x = [v1vdv2vd[0].x[0], v1vdv2vd[1].x[0]]
            y = [v1vdv2vd[0].x[1], v1vdv2vd[1].x[1]]
            plt.plot(x, y, color='orange')

        for vd in v.vd:
            x = [v.x[0], vd.x[0]]
            y = [v.x[1], vd.x[1]]
            plt.plot(x, y, '--', color='tab:green')
    plt.triplot(points[: ,0], points[: ,1], tri.simplices, color='tab:blue')
    plt.plot(points[: ,0], points[: ,1],  'o', color='tab:blue')
    plt.plot(dual_points[: ,0], dual_points[: ,1], 'o', color='tab:orange')
    plt.show()


def plot_dual_mesh_3D(HC, dual_points):
    """
    NOTE: The bug remains where it can only be plotted once, therefore rebuild the complex.
    """
    hcfig, hcaxes, _, _ = HC.plot_complex()
    hcaxes.scatter3D(dual_points[:, 0], dual_points[:, 1], dual_points[:, 2], color = "green")
    for v in HC.V:
        for vd in v.vd:
            x = [v.x[0], vd.x[0]]
            y = [v.x[1], vd.x[1]]
            z = [v.x[2], vd.x[2]]
            hcaxes.plot(x, y, zs=z, linestyle='--',  color='tab:green')
        # ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i],VecEnd_y[i]],zs=[VecStart_z[i],VecEnd_z[i]])

    plt.show()
    return
