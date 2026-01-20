import numpy as np
import matplotlib.pyplot as plt

def plot_refinements(shape, refinements, r=1.0, a=2.0, b=1.0):
    data = []  # List to return: [(vertices, integrated_curvature), ...] for each refinement
    integrated_curvature = np.pi / 2  # Fixed for quarter domain
    if shape == 'circle':
        t = np.linspace(0, np.pi/2, 1000)
        x = r * np.cos(t)
        y = r * np.sin(t)
        plt.plot(x, y, color='tab:blue', linewidth=1.5, label='Analytical Quarter Circle')
        for N in refinements:
            t_d = np.linspace(0, np.pi/2, N + 1)
            x_d = r * np.cos(t_d)
            y_d = r * np.sin(t_d)
            vertices = np.stack((x_d, y_d), axis=1)
            data.append((vertices, integrated_curvature))
            for i in range(N):
                # Draw triangle edges in black
                plt.plot([0, x_d[i]], [0, y_d[i]], color='black', label=f'Triangles for N={N}' if i == 0 else None)
                plt.plot([x_d[i], x_d[i+1]], [y_d[i], y_d[i+1]], color='black')
                plt.plot([x_d[i+1], 0], [y_d[i+1], 0], color='black')
    elif shape == 'ellipse':
        t = np.linspace(0, np.pi/2, 1000)
        x = a * np.cos(t)
        y = b * np.sin(t)
        plt.plot(x, y, color='tab:blue', linewidth=1.5, label='Analytical Quarter Ellipse')
        for N in refinements:
            t_d = np.linspace(0, np.pi/2, N + 1)
            x_d = a * np.cos(t_d)
            y_d = b * np.sin(t_d)
            vertices = np.stack((x_d, y_d), axis=1)
            data.append((vertices, integrated_curvature))
            for i in range(N):
                # Draw triangle edges in black
                plt.plot([0, x_d[i]], [0, y_d[i]], color='black', label=f'Triangles for N={N}' if i == 0 else None)
                plt.plot([x_d[i], x_d[i+1]], [y_d[i], y_d[i+1]], color='black')
                plt.plot([x_d[i+1], 0], [y_d[i+1], 0], color='black')
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'{shape.capitalize()} Refinements')
    plt.show()
    return data

# Example usage:
# data = plot_refinements('circle', [5, 10, 20])
# for vertices, ic in data:
#     L, A = circle_curve_length_area(vertices, ic)
#     print(L, A)
# Similarly for 'ellipse' with ellipse_curve_length_area