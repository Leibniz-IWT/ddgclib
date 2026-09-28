import numpy
import matplotlib.pyplot as plt

# Define colours:
lo = numpy.array([242, 189, 138])/255  # light orange
do = numpy.array([235, 129, 27])/255  # Dark alert orange

lw = 1  # linewidth

def f(x):
    #return 1.1*x[0] + x[1]
    return (x[0] - 0.51)**2 + (x[1] - 0.5)**2

points = numpy.array([[0.0, 0.0],
                      [1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0],

                      ]
                     )


class Plot_complex_normals(object):
    def __init__(self, c_outd, F, nn, xlim=1.5, ylim=1.5, zlim=1.0):
        self.__dict__.update(c_outd)
        color_l = [0, 'tab:blue', 'tab:green', "tab:red", 'tab:purple', 'tab:orange', 'tab:brown','tab:pink',
                   'tab:gray', 'tab:olive', 'tab:cyan']
        #print(f'F[ind = 1] = {F[1]}')
        # Plot the surface compolex
        fig, axes, HC = plot_surface(F, nn)
        # Plot a list of normal vectors
        axes = plot_n([self.n_i], axes, color="tab:gray")

        # Define midpoints
        # mdp_ij = E_ij*0.5  + F[0] # = 0.5*E_ik + v_i
        # mdp_ik = 0.5*E_ik  + F[0]
        # C_ijk = np.zeros_like(A_ijk)
        for ind in range(1, len(nn[0]) + 1):  # ind := j
            try:
                colour = color_l[ind]
            except IndexError:
                cindex = ind % len(color_l)
                colour = color_l[cindex]
            # Plot the e_ij dual planes
            # axes = plot_plane(hat_E_ij[ind], mdp_ij[ind], axes, dim=0.55, color='tab:blue', alpha=.3)
            axes.scatter(*self.mdp_ij[ind], color=colour, alpha=0.5)
            # Plot the e_ik dual planes

            axes.scatter(*self.mdp_ik[ind], color=colour, alpha=0.5)
            # axes = plot_plane(hat_E_ik[ind], mdp_ik[ind], axes, dim=0.55, color='tab:red', alpha=.3)
            # Plot the T_ijk plane
            # axes = plot_plane(N_ijk[ind], F[0], axes, dim=1, color=colour, alpha=.3)
            c = np.zeros(3)
            A = np.zeros([3, 3])
            A[0] = self.E_ij[ind]
            A[1] = self.E_ik[ind]
            A[2] = self.N_ijk[ind]

            c[0] = np.dot(self.E_ij[ind], self.mdp_ij[ind])
            c[1] = np.dot(self.E_ik[ind], self.mdp_ik[ind])
            c[2] = np.dot(self.N_ijk[ind], F[0])
            v_dual = np.linalg.solve(A, c)

            #print(f'np.linalg.norm(F[0] - mdp_ij[ind]) = {np.linalg.norm(F[0] - self.mdp_ij[ind])} = {np.linalg.norm(0.5 * self.L_ij[ind])}')
            h_ij = np.linalg.norm(0.5 * self.L_ij[ind])  # = 0.5*E_ik  + F[0] ?   = F[0] -(0.5*E_ik  + F[0]) = 0.5
            b_ij = np.linalg.norm(v_dual - self.mdp_ij[ind])
            C_ij = 0.5 * b_ij * h_ij
            #print(f'C_ij  = {C_ij}')
            #print(f'b_ij = {b_ij}')

            h_ik = np.linalg.norm(
                0.5 * self.L_ij[int(self.j_k[ind])])  # = 0.5*E_ik  + F[0] ?   = F[0] -(0.5*E_ik  + F[0]) = 0.5
            b_ik = np.linalg.norm(v_dual - self.mdp_ik[ind])
            C_ik = 0.5 * b_ik * h_ik
            #print(f'C_ik  = {C_ik}')
            #print(f'b_ik = {b_ik}')
            self.C_ijk[ind] = C_ij + C_ik  # the area dual to A_ijk (validated at 90deg/0 curvature)
            #print(f'C_ijk = {self.C_ijk}')
            axes.scatter(*v_dual, color=colour, alpha=1.0)
            axes = plot_n([self.N_ijk[ind]], axes=axes, v0=v_dual, color=colour)

        axes.set_xlabel('$x_1$')
        axes.set_ylabel('$x_2$')
        axes.set_zlabel('$x_3$')
        axes.set_xlim3d(-xlim, xlim)
        axes.set_ylim3d(-ylim, ylim)
        axes.set_zlim3d(-zlim, zlim)

        fig.show()
        return None



def g1(x):
    import numpy
    #return numpy.sqrt(-0.05 * (x[0] * x[1] - 5 ) ** 4 + 11) + 11
    #return -0.00000001 * (x[0] * x[1] - 5 ) ** 4 + 1 #+ (x[0] * x[1])**(-4)
    return (x[0] - 5)**2 + (x[1] - 5)**2 + 5 * numpy.sqrt(x[0] * x[1]) - 29


def g2(x):
    import numpy
    #return numpy.sqrt(-0.05 * (x[0] * x[1] - 5 ) ** 4 + 11) + 11
    #return -0.00000001 * (x[0] * x[1] - 5 ) ** 4 + 1 #+ (x[0] * x[1])**(-4)
    return (x[0] - 6)**4 - x[1] + 2

def g3(x):
    import numpy
    #return numpy.sqrt(-0.05 * (x[0] * x[1] - 5 ) ** 4 + 11) + 11
    #return -0.00000001 * (x[0] * x[1] - 5 ) ** 4 + 1 #+ (x[0] * x[1])**(-4)
    return 9 - x[1]


def f(x):  # Ursem01
    import numpy
    #if g1(x) >= 0:
    #    return numpy.inf
    #if g2(x) >= 0:
    #    return numpy.inf

    return -numpy.sin(2 * x[0] - 0.5 *numpy.pi)  - 3 * numpy.cos(x[1]) - 0.5 * x[0]

bl = 10  # boxlimits
points = numpy.array([[0.0, 0.0],
                      [bl, 0.0],
                      [0.0, bl],
                      [bl, bl],

                      ]
                     )



def find_neighbors_delaunay(pindex, triang):
    """
    Returns the indexes of points connected to ``pindex``  on the Gabriel
    chain subgraph of the Delaunay triangulation.

    """
    return triang.vertex_neighbor_vertices[1][
           triang.vertex_neighbor_vertices[0][pindex]:
           triang.vertex_neighbor_vertices[0][pindex + 1]]

def build_contour(fig, points, func, surface=True, contour=True):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm

    #X = points[:, 0]
    X = numpy.linspace(points[0][0], points[-1][1])
    #Y = points[:, 1]
    Y = numpy.linspace(points[0][0], points[-1][1])
    xg, yg = numpy.meshgrid(X, Y)
    Z = numpy.zeros((xg.shape[0],
                     yg.shape[0]))

    for i in range(xg.shape[0]):
        for j in range(yg.shape[0]):
            Z[i, j] = func([xg[i, j], yg[i, j]])

    if surface:
        #fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xg, yg, Z, rstride=1, cstride=1,
                        cmap=cm.coolwarm, linewidth=0,
                        antialiased=True, alpha=1.0, shade=True)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f$')

    if contour:
        plt.figure()
        cs = plt.contour(xg, yg, Z, cmap='binary_r', color='k')
        plt.clabel(cs)



def direct_2d(ax, func, V1, V2, vertex_plot_size=0.00, arrow_color='k'):
    """Draw a directed graph arrow between two vertices"""

    # NOTE: Can retrieve from stored class
    f_1 = func(V1)
    f_2 = func(V2)

    def vertex_diff(V_low, V_high, vertex_plot_size):
        # Note assumes bounds in R+ (0, inf)
        dV = [0, 0]
        for i in [0, 1]:
            if V_low[i] < V_high[i]:
                dV[i] = -(V_high[i] - V_low[i])  # + vertex_plot_size
            else:
                dV[i] = V_low[i] - V_high[i]  # - vertex_plot_size

            if dV[i] > 0:
                dV[i] -= vertex_plot_size
            else:
                dV[i] += vertex_plot_size

        return dV

    if f_1 > f_2:  # direct V2 --> V1
        dV = vertex_diff(V1, V2, vertex_plot_size)
        # print(dV)
        # ax.arrow(V2[0], V2[1], dV[0], dV[1], head_width=0.2, head_length=0.05, fc='k', ec='k', color='b')
        ax.arrow(V2[0], V2[1], 0.5 * dV[0], 0.5* dV[1], head_width=0.15,
                 head_length=0.15, fc=arrow_color, ec=arrow_color, color=arrow_color)

    elif f_1 < f_2:  # direct V1 --> V2
        pass
        # ax.arrow(V2[0], V2[1], dV[0], dV[1], head_width=0.2, head_length=0.05, fc='k', ec='k', color='b')

    return f_1, f_2  # TEMPORARY USE IN LOOP



def build_complex(points):
    import numpy

    from scipy.spatial import Delaunay

    disc = 10
    x = numpy.linspace(0, bl, disc)
    y = numpy.linspace(0, bl, disc)
    points = [[x0, y0] for x0 in x for y0 in y]
    points = numpy.array(points)
    print(points)

    # constraints
    if 0:
        points = points[g1(points.T) >= 0]
        points = points[g2(points.T) >= 0]

    tri = Delaunay(points)
    # Label edges
    edges = numpy.array(tri.simplices)  # numpy.zeros(len(tri.simplices))
    constructed_edges = []
    incidence_array = numpy.zeros(
        [numpy.shape(points)[0], numpy.shape(edges)[0]])
    print(edges)
    # contour
    #build_contour(SHc, surface=False, contour=True)

    # graph
    # Plot the usual graph
    def plot_usual_graph(color='k'):
        plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy(), color=color,
                    linewidth=lw)
        plt.plot(points[:, 0], points[:, 1], '.', color=color, markersize=5)

        ax = plt.axes()

        if 1:
            for i in range(points.shape[0]):
                for i2 in find_neighbors_delaunay(i, tri):
                    # Check g1 and g1
                    if g1(tri.points[i, :]) <= 0 or g2(tri.points[i, :]) <= 0:
                        if g1(tri.points[i2, :]) <= 0 or g2(tri.points[i2, :]) <= 0:
                            continue

                    # Draw arrow
                    f_1, f_2 = direct_2d(ax, f, tri.points[i, :],
                                         tri.points[i2, :], arrow_color=color)

                    # Find incidence on an edge
                    for edge, e in zip(edges, range(numpy.shape(edges)[0])):
                        # print(edge)
                        if e not in constructed_edges:
                            if i in edge:
                                if f_1 < f_2:
                                    incidence_array[i, e] += 1
                                elif f_1 > f_2:
                                    incidence_array[i, e] -= 1
                            if i2 in edge:
                                if f_2 < f_1:
                                    incidence_array[i2, e] += 1
                                elif f_2 > f_1:
                                    incidence_array[i2, e] -= 1

                            constructed_edges.append(e)

        return ax

    def draw_lines(p_to, p_from):  # f_higher, f_lower

        plt.plot([p_to[0], p_from[0]], [p_to[1], p_from[1]], '-', color=do,
                 linewidth=lw)
        plt.plot([p_to[0], p_from[0]], [p_to[1], p_from[1]], '.', color=do,
                 markersize=0.1)
        f_1, f_2 = direct_2d(ax, f, p_to, p_from, arrow_color=do)
        return

    # Add minimizer point
    if 1:
        ax = plot_usual_graph()
        plt.plot([0.5, 0.5], [0.5, 0.5], '.', color=do, markersize=5)
        strpath = './fig/TEST.pdf'

    # Shade most relevant planes
    strpath = './fig/build_non_linear.pdf'
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_xlim(-0.5, bl+0.5)
    ax.set_ylim(-0.5, bl+0.5)
    plt.show()


disc = 10
x = numpy.linspace(0, bl, disc)
y = numpy.linspace(0, bl, disc)
points = [[x0, y0] for x0 in x for y0 in y]
points = numpy.array(points)


if 1:
    fig = plt.figure()
    if 1:  #Function
        build_contour(fig, points, f, surface=True, contour=True)

    if 0:  #Constraint 1
        build_contour(fig, points, g1, surface=0, contour=True)

    if 0:  #Constraint 2
        build_contour(fig, points, g2, surface=0, contour=True)

    if 0:  #Constraint 3
        build_contour(fig, points, g3, surface=0, contour=True)

    if 0:  #Function
        build_contour(fig, points, f, surface=0, contour=True)

    build_complex(points)

    if 1:
        points = points[g1(points.T) >= 0]
        points = points[g2(points.T) >= 0]