import copy
import numpy as np

# ddg imports
from ddgclib import *
from hyperct import *
from ddgclib._curvatures import * #plot_surface#, curvature
from ddgclib._capillary_rise_flow import * #plot_surface#, curvature
from ddgclib._ellipsoid import *
from ddgclib._eos import *
from ddgclib._misc import *
from ddgclib._plotting import *

from matplotlib import pyplot as plot

# Field equation
def func(X):
    x, y = X
    a = -1.5
    b = 0.5
    P = x*(y-a*x)
    Q = y*(x + b + y)
    #return (P + Q)**2
    return 0.0# (P**2 + Q**2)  #**2
    #return P*Q#**2


# colors
# Solid particles:
do = numpy.array([235, 129, 27]) / 255  # Dark alert orange
# Fluid particles:
db = numpy.array([129, 160, 189]) / 255  # Dark alert orange
lb = numpy.array([176, 206, 234]) / 255  # Dark alert orange

# Minima: (0.0, 0.0), (1.0, -1.5), (0.0, 0.5)
# Domain
#bounds = [(-3.5, 3.5), (-3.5, 3.4)]
bounds = [(-10.0, 10.0),]*3
dim = 3
HC = Complex(dim, domain=bounds, sfield=func)  # , symmetry=symmetry)

# Solid nanoparticles:
np_1 = 0.0  # position of nanoparticle 1
np_2 = np.array([0.0, 0.0, 0.0])  # position of nanoparticle 1
R_1 = 2.0  # nm, radius of nanoparticle 1
delta_1 = 0.6  # nm, hydrate layer thickness of nanoparticle 1
np_2 = 4.5  # nm, radius of nanoparticle 2
np_2 = np.array([4.5, 0.0, 0.0])  # nm, radius of nanoparticle 2
R_2 = 2.0  # nm, radius of nanoparticle 2
delta_2 = 0.6  # nm, hydrate layer thickness of nanoparticle 2

#HC.triangulate()
HC.triangulate()
#for i in range(4):
for i in range(4):
#for i in range(3):
    HC.refine_all()

HC.V.process_pools()

print('='*20)
print('-'*20)
print('='*20)

vertices = copy.copy(HC.V)
del_set = set()
for v in vertices:
    #print(f'v = {v}')
   # print(f'v.x_a. = {v.x_a}')
    numpy.linalg.norm(v.x_a)
    # Remove fluid in solid particles
    if numpy.linalg.norm(v.x_a) <= R_1:
        del_set.add(v)
    if numpy.linalg.norm(v.x_a - np_2) <= R_2:
        del_set.add(v)
        #HC.V.remove(v)

    # Remove fluid outside overlapping layers and bridges
    ta1 = numpy.linalg.norm(v.x_a) > (R_1 + delta_1)#**2
    ta2 = numpy.linalg.norm(v.x_a - np_2) > (R_2 + delta_2)#**2

    if ta1 and ta2:
        del_set.add(v)

for v in del_set:
    HC.V.remove(v)

vertices = copy.copy(HC.V)
del_set = set()
for v in vertices:
    verticesnn = copy.copy(v.nn)
    for vn in verticesnn:
        v.disconnect(vn)
#HC.V.print_out()
if 1:
    print(f'Plotting...')
    #HC.plot_complex(pointsize=25)
    fig_complex, ax_complex, fig_surface, ax_surface = HC.plot_complex(
        arrow_width=0.2,
        point_color=db, line_color=db,
        complex_color_f=lb, complex_color_e=db
    )

    N = 100
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(0, np.pi, N)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax_complex.plot_surface((x + np_2[0]/2.0)*R_1, y*R_1, z*R_1, linewidth=0.0,
                            color=do, alpha=0.3
                            )

    ax_complex.plot_surface(x*R_1, y*R_1, z*R_1, linewidth=0.0,
                            color=do, alpha=0.3
                            )

    if 0:
        ax_complex.axes.set_xlim3d(left=bounds[0][0], right=bounds[0][1])
        ax_complex.axes.set_ylim3d(bottom=bounds[1][0], top=bounds[1][1])
        ax_complex.axes.set_zlim3d(bottom=bounds[2][0], top=bounds[2][1])
    ax_complex.axes.set_xlim3d(left=-5, right=10)
    ax_complex.axes.set_ylim3d(bottom=-5, top=10)
    ax_complex.axes.set_zlim3d(bottom=-5, top=10)
    if 0:
        msize = 4*np.pi*R_1**2 #* 3.4
        ax_complex.plot(0.0,0.0,0.0, marker='o',
                 #markersize=80,
                 markersize= msize,
                        color=do, alpha=0.5)
        ax_complex.plot(4.5,0.0,0.0, marker='o',
                 #markersize=80,
                 markersize= msize,
                        color=do, alpha=0.5)

             #markersize ** 2
    #ax_complex.scatter([0], [0], [0], s=22 ** 2)
    plot.show()

    #fig_complex, ax_complex, fig_surface, ax_surface

if 0:
    x, y = np.meshgrid(np.linspace(-3.5, 3.5, 20),
                       np.linspace(-3.5, 3.5, 20))
    u = 1
    v = -1
    a = -1.5
    b = 0.5
    x1 = x*(y-a*x)
    x2 = y*(x + b + y)
    ax_complex.quiver(x, y, x1, x2)
    #plot.quiver(x,y,x1,x2)
    plot.show()





