"""
Deprecated: use ddgclib.visualization.unified instead.

This module is a legacy compatibility shim. All plotting functions have been
moved to ddgclib.visualization.unified, which delegates mesh rendering to
hyperct._plotting.plot_complex.
"""
import warnings

warnings.warn(
    "ddgclib._plotting is deprecated. "
    "Use ddgclib.visualization.unified instead.",
    DeprecationWarning,
    stacklevel=2,
)

import collections

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
#from ipywidgets import *
from matplotlib.widgets import Slider
import polyscope as ps
from ddgclib._misc import *  # coldict is neeeded
from ddgclib._misc import coldict
#from ddgclib.operators.gradient import velocity_laplacian as du

plt.rcdefaults()
plt.rcParams.update({"text.usetex": True,'font.size' : 14,})

# Polyscope
def plot_polyscope(HC):
    # Initialize polyscope
    ps.init()
    ps.set_up_dir("z_up")

    do = coldict['db']
    lo = coldict['lb']
    HC.dim = 2  # The dimension has changed to 2 (boundary surface)
    HC.vertex_face_mesh()
    points = np.array(HC.vertices_fm)
    triangles = np.array(HC.simplices_fm_i)
    ### Register a point cloud
    # `my_points` is a Nx3 numpy array
    my_points = points
    ps_cloud = ps.register_point_cloud("my points", my_points)
    ps_cloud.set_color(tuple(do))
    #ps_cloud.set_color((0.0, 0.0, 0.0))
    verts = my_points
    faces = triangles
    ### Register a mesh
    # `verts` is a Nx3 numpy array of vertex positions
    # `faces` is a Fx3 array of indices, or a nested list
    surface = ps.register_surface_mesh("my mesh", verts, faces,
                             color=do,
                             edge_width=1.0,
                             edge_color=(0.0, 0.0, 0.0),
                             smooth_shade=False)

    # Add a scalar function and a vector function defined on the mesh
    # vertex_scalar is a length V numpy array of values
    # face_vectors is an Fx3 array of vectors per face
    if 0:
        ps.get_surface_mesh("my mesh").add_scalar_quantity("my_scalar",
                vertex_scalar, defined_on='vertices', cmap='blues')
        ps.get_surface_mesh("my mesh").add_vector_quantity("my_vector",
                face_vectors, defined_on='faces', color=(0.2, 0.5, 0.5))

    # View the point cloud and mesh we just registered in the 3D UI
    #ps.show()
    ps.show()


def plot_dual(vd, HC, vector_field=None, scalar_field=None, fn='', up="x_up"
              , stl=False, length_scale=1.0, point_radii=0.005):
    # Reset the indices for plotting:
    for i, v in enumerate(HC.V):
        v.index = i
    v1 = vd
    # Initialize polyscope
    ps.init()
    ps.set_up_dir('z_up')
    do = coldict['do']
    lo = coldict['lo']
    db = coldict['db']
    lb = coldict['lb']
    tg = coldict['tg']  # Tab:green colour
    # %% Plot Barycentric dual mesh
    # Loop over primary edges
    dual_points_set = set()
    ssets = []  # Sets of simplices
    v1 = vd
    for i, v2 in enumerate(v1.nn):
        # Find the dual vertex of e12:
        vc_12 = 0.5 * (v2.x_a - v1.x_a) + v1.x_a  # TODO: Should be done in the compute_vd function
        vc_12 = HC.Vd[tuple(vc_12)]

        # Find local dual points intersecting vertices terminating edge:
        dset = v2.vd.intersection(v1.vd)  # Always 5 for boundaries
        # Start with the first vertex and then build triangles, loop back to it:
        vd_i = list(dset)[0]
        if v1.boundary and v2.boundary:
            # print(f'len(dset) = {len(dset)}')
            # Remove the boundary edge which should already be in the set:
            if not (len(vd_i.nn.intersection(dset)) == 1):
                for vd in dset:
                    vd_i = vd
                    if len(vd_i.nn.intersection(dset)) == 1:
                        break
            # iter_len = 3
            # The set length much be different because all interior planes
            # are counted minus two boudary vertices which do not form triangles
            # such as the flux planes in the bulk
            iter_len = len(list(dset)) - 2
        else:
            iter_len = len(list(dset))

        # Main loop
        dsetnn = vd_i.nn.intersection(dset)  # Always 1 internal dual vertices
        vd_j = list(dsetnn)[0]
        # NOTE: In the boundary edges the last triangle does not have
        #      a final vd_j
        # print(f'dset = {dset}')
        for _ in range(iter_len):  # For boundaries should be length 2?
            ssets.append([vc_12, vd_i, vd_j])
            dsetnn_k = vd_j.nn.intersection(dset)  # Always 2 internal dual vertices in interior
            # print(f'dsetnn_k = {dsetnn_k}')
            dsetnn_k.remove(vd_i)  # Should now be size 1
            vd_i = vd_j
            try:
                # Alternatively it should produce an IndexError only when
                # _ = 2 (end of range(3) and we are on a boundary edge
                # so that (v1.boundary and v2.boundary) is true
                vd_j = list(dsetnn_k)[0]  # Retrieve the next vertex
            except IndexError:
                pass  # Should only happen for boundary edges

        # Find local dual points intersecting vertices terminating edge:
        dset = v2.vd.intersection(v1.vd)
        pi = []
        for vd in dset:
            # pi.append(vd.x + 1e-9 * np.random.rand())
            pi.append(vd.x)
            dual_points_set.add(vd.x)
        pi = np.array(pi)
        pi_2d = pi[:, :2] + 1e-9 * np.random.rand()

        # Plot dual points:
        dual_points = []
        for vd in dual_points_set:
            dual_points.append(vd)

        dual_points = np.array(dual_points)
        ps_cloud = ps.register_point_cloud("Dual points", dual_points)
        ps_cloud.set_color(do)
        ps_cloud.set_radius(point_radii)

    # Build the simplices for plotting
    faces = []
    vdict = collections.OrderedDict()  # Ordered cache of vertices to plot
    ind = 0
    # Now iterate through all the constructed simplices and find indexes
    for s in ssets:
        f = []
        for vd in s:
            if not (vd.x in vdict):
                vdict[vd.x] = ind
                ind += 1

            f.append(vdict[vd.x])
        faces.append(f)

    verts = np.array(list(vdict.keys()))
    faces = np.array(faces)

    print(f'verts = {verts}')
    dsurface = ps.register_surface_mesh(f"Dual face", verts, faces,
                                        color=do,
                                        edge_width=0.0,
                                        edge_color=(0.0, 0.0, 0.0),
                                        smooth_shade=False)

    dsurface.set_transparency(0.5)
    # Plot primary mesh
    HC.dim = 2  # The dimension has changed to 2 (boundary surface)
    HC.vertex_face_mesh()
    HC.dim = 3  # Reset the dimension to 3
    points = np.array(HC.vertices_fm)
    triangles = np.array(HC.simplices_fm_i)

    # %% Register the primary vertices as a point cloud
    # `my_points` is a Nx3 numpy array
    my_points = points
    ps_cloud = ps.register_point_cloud("Primary points", my_points)
    ps_cloud.set_color(tuple(db))
    ps_cloud.set_radius(point_radii)
    # ps_cloud.set_color((0.0, 0.0, 0.0))
    verts = my_points
    faces = triangles
    if stl:
        #  msh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                pass
                # msh.vectors[i][j] = verts[f[j], :]

        # msh.save(f'{fn}.stl')

    ### Plot the primary mesh
    # `verts` is a Nx3 numpy array of vertex positions
    # `faces` is a Fx3 array of indices, or a nested list
    if 1:
        surface = ps.register_surface_mesh("Primary surface", verts, faces,
                                           color=db,
                                           edge_width=1.0,
                                           edge_color=(0.0, 0.0, 0.0),
                                           smooth_shade=False)

        surface.set_transparency(0.3)
        # Add a scalar function and a vector function defined on the mesh
        # vertex_scalar is a length V numpy array of values
        # face_vectors is an Fx3 array of vectors per face

        # Scene options (New, not working for scaling
        # NOTE: VERY BROKEN AS IT SCALES THE DIFFERENT MESHES RELATIVELY: NEVER USE THIS:
        #ps.set_autocenter_structures(True)
        #ps.set_autoscale_structures(True)

        # View the point cloud and mesh we just registered in the 3D UI
        # ps.show()
        # Plot particles
        # Ground plane options
        ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
        ps.set_ground_plane_height_factor(0.1)  # adjust the plane height
        ps.set_shadow_darkness(0.2)  # lighter shadows
        ps.set_shadow_blur_iters(2)  # lighter shadows
        ps.set_transparency_mode('pretty')
        ps.set_length_scale(length_scale)
        #ps.set_length_scale(length_scale)
     #   ps.set_length_scale(length_scale)
        # ps.look_at((0., -10., 0.), (0., 0., 0.))
       # ps.look_at((1., -8., -8.), (0., 0., 0.))
        # ps.set_ground_plane_height_factor(x, is_relative=True)
        ps.set_screenshot_extension(".png")
        # Take a screenshot
        # It will be written to your current directory as screenshot_000000.jpg, etc
        ps.screenshot(fn)

    return ps, du

# Plot surface mesh
def pplot_surface(HC):

    HC.vertex_face_mesh()
    #print(f'verts = {HC.vertices_fm}')
    #print(f'faces = {HC.simplices_fm}')
    #print(f'faces i = {HC.simplices_fm_i}')
    # Initialize polyscope
   # print(f'verts = {np.array(HC.vertices_fm)}')
   # print(f'faces = {np.array(HC.simplices_fm)}')
   # print(f'faces i = {np.array(HC.simplices_fm_i)}')
    ps.init()

    ### Register a point cloud
    # `my_points` is a Nx3 numpy array
    if 0:
        my_points = np.array([[0, 0, 0],
                              [1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]
                              )
        ps.register_point_cloud("my points", my_points)

    #if 0:


    verts = np.array(HC.vertices_fm)
    faces = np.array(np.array(HC.simplices_fm_i))
    if 0:
        faces = np.array([[0, 1, 2],
                          [0, 2, 3],
                          [1, 2, 3],
                          #[0, 1, 3],
                          ]
                              )
    ### Register a mesh
    # `verts` is a Nx3 numpy array of vertex positions
    # `faces` is a Fx3 array of indices, or a nested list
    ps.register_surface_mesh("my mesh", verts, faces, smooth_shade=True)
    ps.set_up_dir('z_up')
    if 0:
        # Replot error data
        Nmax = 21
        lp_error = np.zeros(Nmax)
        N = list((range(Nmax)))

        plt.figure()
        plt.plot(N, lp_error)
        plt.xlabel(r'N (number of boundary vertices)')
        plt.ylabel(r'%')
        plt.show()

    ps.screenshot("mesh.png")
    return ps

# Plot Adam Bashforth profiles
def plot_Adam_Bash(): 
  fig, ax = plt.subplots(1)#, figsize=[columnWid, .6*columnWid])
  root =  'data/'
  #for inFile in sorted(os.listdir(root)):
  #for Bo in range(-4,5):
  Bo = 0 
  if True:
    #if 'adams' not in inFile: continue
    #fname = root + inFile
    fname = root + 'adams' + str(Bo) + '.txt'
    with open(fname, encoding = 'utf-8') as f:
      print('loadin ',fname)
      df = np.loadtxt(f)
      #col = int(fname[-5])
      #col = -Bo/5+.5
      #if fname[-6]!='-': col=-col
      #col = col+5
      #col = col/10
      r=0
      b=0
      if Bo<0: r=-Bo/4
      if Bo>0: b= Bo/4
      ax.plot(df[:,0]*1e3,df[:,1]*1e3, color=(r**.5,0,b**.5))
      #lbl='$Bo='+str(Bo/10)+'$'
      #if abs(Bo)==4:
        #ax.text(df[100,0]*1e3, df[100,1]*1e3, lbl, c=(r**.5,0,b**.5), rotation=-90)# fontsize=12)
      #ax.plot(df[:,0]*1e3,df[:,1]*1e3, color=mpl.colormaps['coolwarm'](col), label=lbl)
      #ax.scatter(x=0, y=0, c=col, cmap="coolwarm") 
  #im = ax.imshow(range(-4,5), cmap='coolwarm')
  #fig.colorbar(im, cax=ax, orientation='vertical')
  #plt.colorbar()
  ax.set_aspect('equal', adjustable='box')
  ax.tick_params(which='both', direction='in', top=True, right=True)
  #ax.legend(prop={'size':8}, loc='upper left') 
  ax.set_xlabel('$x/R$', rotation=0)
  ax.set_ylabel('$z/R$', rotation=0)
  ax.yaxis.set_label_coords(-.25,.45)
  ax.set_xlim([0,1.2])
  ax.set_ylim([-3.3,0])
  #ax.text(1.1, -.8, '$Bo=0.4$',c=(0,0,1), rotation=-70, fontsize=10, ha='center', va='center')
  #ax.text(.8, -.8, '$Bo=-0.4$',c=(1,0,0), rotation=-85, fontsize=10, ha='center', va='center')
  ax.text(.3, -1.45, '$Bo=-0.4$',c=(1,0,0), fontsize=10, ha='center', va='center')
  ax.text(.2, -2.05, '$Bo=0$',c=(0,0,0), fontsize=10, ha='center', va='center', rotation=10)
  ax.text(.68, -3.15, '$Bo=0.4$',c=(0,0,1), fontsize=10, ha='center', va='center')
  fname='data/AdamBash.pdf'
  print('savin ',fname)
  fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
  return

def plot_bubble_coords(): 
  fig, ax = plt.subplots(1)#, figsize=[columnWid, .6*columnWid])
  fname = 'data/adams0.txt'
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f)
  ax.plot(df[:81,0]*1e3,df[:81,1]*1e3, color=(0,0,0))
  ax.annotate("", xy=df[81,:]*1e3, xytext=df[80,:]*1e3,arrowprops=dict(arrowstyle="-|>",fc='k')) 
  x = df[61,0]*1e3
  y = df[61,1]*1e3
  ax.plot((x,x+.2), (y,y), color=(0,0,0))
  Xarr = []
  Yarr = []
  for phi in range(50):
    X = x + .08*np.sin(phi*np.pi/50)
    Y = y + .08*np.cos(phi*np.pi/50)
    if Y > y: continue
    if X < df[68,0]*1e3: continue
    Xarr.append(X)
    Yarr.append(Y)
  ax.plot(Xarr,Yarr,c=(0,0,0))
  ax.text(*df[71,:]*1e3+[.07,0], "$\\theta$", fontsize=12, ha='center', va='center') 
  ax.set_aspect('equal', adjustable='box')
  ax.tick_params(which='both', direction='in', top=True, right=True)
  ax.set_xlabel('$x/R$', rotation=0)
  ax.set_ylabel('$z/R$', rotation=0)
  ax.yaxis.set_label_coords(-.25,.45)
  ax.set_xlim([0,1.2])
  ax.set_ylim([-3.3,0])
  ax.text(*df[81,:]*1e3, '$s$',c=(0,0,0), fontsize=12, ha='center', va='center')
  fname = 'data/bubbleCoords.pdf'
  print('savin ',fname)
  fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
  return

def plot_cone(): 
  import matplotlib.image as mpimg
  fig, ax = plt.subplots(1)
  root =  'data/'
  fname = root + 'cone.png'
  image = mpimg.imread(fname)
  shp = np.shape(image)
  print('shp',shp)
  ax.imshow(image)
  ax.annotate('', xy=[.5*shp[1],150], xytext=[.5*shp[1],130],arrowprops=dict(arrowstyle="<|-",fc='k') )
  ax.plot([ .5*shp[1], .5*shp[1] ], [1465,1370],c='k') 
  ax.plot([ .5*shp[1], .5*shp[1] ], [160,300],c='k') 
  ax.text(.5*shp[1], 110, "${z}$", fontsize=20, ha='center', va='center') 
  plt.axis('off')
  fname = 'data/cone.pdf'
  print('savin ',fname)
  fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
  return

def BoColour(Bo):
  r=0
  g=0
  b=0
  #if Bo<0: r=min( (-Bo/4)**.5, 1)
  #if Bo>0: b=min( (Bo/4)**.5, 1)
  if Bo<0: r = -2*np.arctan(4*Bo)/np.pi
  if Bo>0: b = 2*np.arctan(4*Bo)/np.pi
  return (r,g,b)

def plot_centroid_vs_iteration(height): 
  fig, ax = plt.subplots(1)#, figsize=[columnWid, .6*columnWid])
  col=BoColour(4)
  fname = 'data/vol.txt'
  print(f'fname = {fname}')
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f, usecols=range(18))
  ax.plot(df[:,0],df[:,7], '.', mec=col, mfc='None', mew=.2, alpha=.5)
  ax.plot([df[0,0],df[-1,0]],[height,height], color=col, alpha=.5)#ls='dashed')
  ax.tick_params(which='both', direction='in', top=True, right=True)
  ax.set_xlabel('$n$', rotation=0)
  #ax.set_ylabel('$\\langle z \\rangle/R$')#, rotation=0)
  ax.set_ylabel('$\\frac{\\langle z \\rangle}{R}$', rotation=0, size=20, labelpad=10)
  ax.set_xlim([df[0,0],df[-1,0]])
  fname = 'data/centroid_vs_iteration.pdf'
  print('savin ',fname)
  fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
  return

def plot_centroid_vs_iteration_compare(height): 
  fig, ax = plt.subplots(1)#, figsize=[columnWid, .6*columnWid])
  col=BoColour(4)
  fname = 'oneProc/vol.txt'
  print(f'fname = {fname}')
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f, usecols=range(18))
  ax.plot(df[:,0],df[:,7], '.', mec='None', mfc=col, mew=.2, alpha=.5)
  ax.plot([df[0,0],df[-1,0]],[height,height], color=col, alpha=.5)#ls='dashed')
  fname = 'data/vol.txt'
  print(f'fname = {fname}')
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f, usecols=range(18))
  ax.plot(df[:,0],df[:,7], '.', mec=col, mfc='None', mew=.2, alpha=.5)
  ax.plot([df[0,0],df[-1,0]],[height,height], color=col, alpha=.5)#ls='dashed')
  ax.tick_params(which='both', direction='in', top=True, right=True)
  ax.set_xlabel('$n$', rotation=0)
  #ax.set_ylabel('$\\langle z \\rangle/R$')#, rotation=0)
  ax.set_ylabel('$\\frac{\\langle z \\rangle}{R}$', rotation=0, size=20, labelpad=10)
  ax.set_xlim([df[0,0],df[-1,0]])
  fname = 'data/centroid_vs_iteration.pdf'
  print('savin ',fname)
  fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
  return

def plot_vol(): 
  fig, ax = plt.subplots(1)#, figsize=[columnWid, .6*columnWid])
  fname = 'Bo0/vol.txt'
  V0 = 2*np.pi*1e-9/3
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f)
  ax.plot(df[:,0],df[:,1]/V0, '.', color=(0,0,0))
  ax.tick_params(which='both', direction='in', top=True, right=True)
  ax.set_xlabel('$n$', rotation=0)
  ax.set_ylabel('$V/V_0$', rotation=0)
  ax.set_xlim([0,540])
  #ax.set_ylim([.9,2.1])
  fname = 'data/vol.pdf'
  print('savin ',fname)
  fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
  return

def plot_profile(t): 
  fig, ax = plt.subplots(1)#, figsize=[columnWid, .6*columnWid])
  col=BoColour(4)
  folName = 'data/'
  fname = folName + 'pos' + str(t) + '.txt'
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f)
  height = 0#max(df[:,2])
  ax.plot(np.sqrt(df[:,0]**2 + df[:,1]**2), (df[:,2]-height), '.', mec=col, mfc='None', mew=.2, alpha=.5)
  fname = folName + 'adams0.4.txt'
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f)
  height = -df[-1,1]
  ax.plot(df[:,0], (df[:,1]+height), color=col, alpha=.5)
  ax.tick_params(which='both', direction='in', top=True, right=True)
  ax.set_xlabel('$r/R$')
  ax.set_ylabel('$z/R$')
  ax.set_ylim([0,1.5])
  ax.set_xlim([0,1.2])
  ax.set_aspect('equal', adjustable='box')
  fname = 'data/profile.pdf'
  print('savin ',fname)
  fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
  return

def plot_profile_with_grid_resolution(): 
  fig, ax = plt.subplots(1)#, figsize=[columnWid, .6*columnWid])
  col=BoColour(4)
  folName = 'radTopBy1/'
  fname = folName + 'pos1500.txt'
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f)
  height = 0#max(df[:,2])
  ax.plot(np.sqrt(df[:,0]**2 + df[:,1]**2), (df[:,2]-height), '.', mec=col, mfc='None', mew=.2, markersize=100/1, alpha=.5)
  fname = folName + 'adams0.4.txt'
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f)
  height = -df[-1,1]
  ax.plot(df[:,0], (df[:,1]+height), color=col, alpha=.5)
  col=BoColour(4)
  folName = 'radTopBy2/'
  fname = folName + 'pos2000.txt'
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f)
  height = 0#max(df[:,2])
  ax.plot(np.sqrt(df[:,0]**2 + df[:,1]**2), (df[:,2]-height), '.', mec=col, mfc='None', mew=.2, markersize=100/2, alpha=.5)
  folName = 'radTopBy4/'
  fname = folName + 'pos1000.txt'
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f)
  height = 0#max(df[:,2])
  ax.plot(np.sqrt(df[:,0]**2 + df[:,1]**2), (df[:,2]-height), '.', mec=col, mfc='None', mew=.2, markersize=100/4, alpha=.5)
  folName = 'radTopBy8/'
  fname = folName + 'pos1000.txt'
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f)
  height = 0#max(df[:,2])
  ax.plot(np.sqrt(df[:,0]**2 + df[:,1]**2), (df[:,2]-height), '.', mec=col, mfc='None', mew=.2, markersize=100/8, alpha=.5)
  folName = 'radTopBy16/'
  fname = folName + 'pos10000.txt'
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f)
  height = 0#max(df[:,2])
  ax.plot(np.sqrt(df[:,0]**2 + df[:,1]**2), (df[:,2]-height), '.', mec=col, mfc='None', mew=.2, markersize=100/16, alpha=.5)
  ax.tick_params(which='both', direction='in', top=True, right=True)
  ax.set_xlabel('$r/R$')#, rotation=0, labelpad=-5)
  ax.set_ylabel('$z/R$')#, rotation=0)
  ax.set_ylim([0,1.5])
  ax.set_xlim([0,1.2])
  ax.set_aspect('equal', adjustable='box')
  #ax.text(.3, -1.45, '$Bo=-0.4$',c=(1,0,0), fontsize=10, ha='center', va='center')
  #ax.text(.4, -3, '$Bo=0.4$',c=(0,0,1), fontsize=10, ha='center', va='center')
  fname = 'data/profileWithGridResolution.pdf'
  print('savin ',fname)
  fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
  return

def plot_drop_growing(radInd=-1, rads=()): 
  import os
  folName = 'data/'
  for r in range(len(rads)):
    fig, ax = plt.subplots(1)
    for fname in sorted(os.listdir(folName)):
      if 'txt' not in fname: continue
      if 'bub' not in fname: continue
      with open(folName+fname, encoding = 'utf-8') as f:
        df = np.loadtxt(f)
      if df.ndim<2: continue
      col = (min(rads[r]/np.pi,1), 0, 0)
      for p in range(len(df[:,0]) - 1, -1, -1):
        if (df[p,radInd]-rads[r])*(df[p-1,radInd]-rads[r])<0: ax.plot(df[:p,0], df[:p,1]-df[p,1], color=col)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_xlabel('$r/\\lambda$')
    ax.set_ylabel('$\\frac{ z }{\\lambda}$',rotation=0,size=22)
    ax.set_xlim(left=0)
    ax.set_ylim([0,4])
    ax.set_aspect('equal', adjustable='box')
    if radInd==0: fname = folName+f'pin{r}.pdf'
    if radInd==2: fname = folName+f'spread{r}.pdf'
    print('savin ',fname)
    fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
  return

def plot_drops_growing(radInd=-1, rads=()): 
  import os
  from matplotlib.patches import RegularPolygon
  folName = 'data/'
  figProf, axProf = plt.subplots(1)
  figProf.set_figwidth(12)
  if radInd==0: 
    outName = folName+f'pin.pdf'
    #spc=(.3,1.2,3,5,8.5,14)
    spc=(1,3,6,10,15,21,28,40,50,60)
    #axProf.set_xticklabels([])
    axProf.set_xlabel('$x/\\lambda$')
  if radInd==2: 
    outName = folName+f'spr.pdf'
    #spc=(.5,2,4,7,12,14)
    #spc=(.3,1.2,3,5,8.5,14,20,25,30)
    #spc=(.5,2,4.5,8,13,20)
    spc=(.7,2,4.5,8,13,20,30,40,50)
    axProf.set_xlabel('$x/\\lambda$')
  #spc=(.3,1.2,3,5,8.5,14,20,25,30)
  for r in range(len(rads)):
    for fname in reversed(sorted(os.listdir(folName))):
      if 'txt' not in fname: continue
      if 'bub' not in fname: continue
      with open(folName+fname, encoding = 'utf-8') as f:
        df = np.loadtxt(f)
      if df.ndim<2: continue
      #if df[0,5]>2: continue
      if df[0,5]<1: col = ( 1-df[0,5], 0, 0)
      else: col = ( 0, 0, df[0,5]-1)
      #else: col = ( 0, 0, 1-1/df[0,5])
      if not r: 
        axProf.plot((0,.2),(df[0,5],df[0,5]),c=col)
      for p in range(1, len(df[:,0])):
        if (df[p,radInd] - rads[r]) * (df[p-1,radInd] - rads[r]) >= 0: continue
        print(fname,radInd,df[p,6],2*np.pi*rads[r],df[p,0],rads[r])
        if radInd==0 and df[p,6] > 2*np.pi*rads[r]: continue
        x=np.concatenate(( -df[:p,0][::-1] , df[:p,0] ))
        x=x+spc[r]
        y=np.concatenate(( df[:p,1][::-1] - df[p,1] , df[:p,1] - df[p,1] ))
        axProf.plot(x,y, c=col, clip_on=False)
        #if '010' in fname and radInd==0 and r==len(rads)-1:
        if radInd!=0 or r!=0 or df[p,1]>=-2: continue
        axProf.plot(x,y, c=col, clip_on=False)
        print('radInd',radInd,r,fname)
        h=int(0.5*(len(x)))
        axProf.text(x[h], y[h]+.2, '$(a,h)$', va='bottom', ha='center', c='grey')
        axProf.plot(x[h],y[h],'o', c='grey')
        t=int(0.7*(len(x)))
        axProf.plot(x[h:t],y[h:t], c='grey')
        #axProf.annotate("", xy=[x[t], y[t]], xytext=[x[t-1], y[t-1]], arrowprops=dict(arrowstyle="-|>", color='grey')) 
        theta = np.arctan2(y[t+1]-y[t], x[t+1]-x[t])-np.pi/2
        print('theta',theta)
        tri = RegularPolygon((x[t], y[t]), 3, radius=0.17, orientation=theta, color='grey', zorder=3)
        axProf.add_patch(tri)
        axProf.text(x[t]+0.2, y[t], '$s$', va='center', ha='left', c='grey')
        t=int(0.63*(len(x)))
        xAn = x[t]
        yAn = y[t]
        axProf.plot((xAn,xAn+.3), (yAn,yAn), color='grey')
        Xarr = []
        Yarr = []
        for phi in range(51):
          X = xAn + .17*np.cos(phi*np.pi/50)
          Y = yAn + .17*np.sin(phi*np.pi/50)
          for i in range(len(x)):
            if x[i]>X: break
          if y[i]>Y: break
          Xarr.append(X)
          Yarr.append(Y)
        axProf.plot(Xarr,Yarr,c='grey')
        axProf.text(xAn+.1, yAn+.1, "$\\phi$", ha='left', va='bottom', c='grey') 
  axProf.tick_params(which='both', direction='in', top=True, right=True)
  axProf.set_ylabel('$\\frac{ z }{\\lambda}$',rotation=0,size=22,labelpad=10)
  axProf.set_ylim([0,4])
  axProf.set_xlim([0,24.5])
  axProf.set_aspect('equal', adjustable='box')
  print('savin ',outName)
  figProf.savefig(outName, bbox_inches='tight', transparent=True)
  return

def plot_drop_profile(name='pin spread'): 
  import os
  folName = 'data/'
  for cont in name.split():
    fig, ax = plt.subplots(1)
    for fname in sorted(os.listdir(folName)):
      if '.txt' not in fname: continue
      if cont not in fname: continue
      with open(folName+fname, encoding = 'utf-8') as f:
        print('open',folName+fname)
        df = np.loadtxt(f)
      if df.ndim<2: continue
      if 'spread' in cont and df[0,-1]==.566: continue
      col=BoColour(df[0,-1])
      col='k'
      if '0028' in fname: col='b'
      #ax.plot(df[:,0]/df[-1,0], (df[:,1]-df[-1,1])/df[-1,0], 'o', color=col)#, alpha=.9)
      ax.plot(df[:,0], df[:,1], color=col)#, alpha=.9)
      for p in range(1,len(df[:,0])):
        if df[p,3]*df[p-1,3]<0: ax.plot(df[p,0], df[p,1], '+', color=col)#, alpha=.9)
      #logBo = int(np.log2(df[-1,-1]))
      if False:#logBo==-2: 
        if 'spread' in cont:
          ax.plot([df[-1,0],df[-1,0]+.15], [df[-1,1],df[-1,1]], color='k', linestyle='dashed')
          ax.text(df[-1,0]+.15, df[-1,1]+.1, f'$\\phi$', ha='center', va='center')
        elif 'pin' in cont:
          ax.plot([df[-1,0],0], [df[-1,1],df[-1,1]], color='k', linestyle='dashed')
          ax.text(df[-1,0]/2, df[-1,1]+.1, '$r_\\mathrm{con}$', ha='center', va='center')
      #if logBo>2: continue
      #if logBo<-1: continue
      #ax.text(*df[-1,:2], f'${df[-1,-1]**.5:.4g}$', ha='left', va='center')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_xlabel('$r/\\lambda$')
    ax.set_ylabel('$\\frac{ z }{\\lambda}$',rotation=0,size=22)
    #ax.set_ylim([-2,2])
    #ax.set_xlim([0,1])
    ax.set_aspect('equal', adjustable='box')
    fname = folName+'pin_'+cont+'.pdf'
    print('savin ',fname)
    fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
  return

def plot_drop_height_vs_rad(nam='rad ang bub'): 
  import os
  from matplotlib.patches import RegularPolygon
  from ddgclib._bubble import AdamsBashforthProfile
  def colRB(cont,r):
    if r<1: 
      colVal = r**.5
      if 'rad' in cont: return ( colVal, 0, 0 )
      else:             return ( 0, 0, colVal )
    else: 
      colVal = 1 - 1/r
      if 'rad' in cont: return ( 1, colVal, colVal )
      else:             return ( colVal, colVal, 1 )
  folName = 'data/'
  figProf, axProf = plt.subplots(2, sharex=True)
  fig, ax = plt.subplots(1, 2, sharey=True)
  for cont in nam.split():
    figV, axVV = plt.subplots(2, sharex=True)
    figV.subplots_adjust(hspace=0.07)
    fig2, ax2 = plt.subplots(1)
    x=[]
    dfDet=[]
    z=[]
    zM=[]
    for fname in reversed(sorted(os.listdir(folName))):
      if 'prof' in fname: continue
      if 'txt' not in fname: continue
      if cont not in fname: continue
      with open(folName+fname, encoding = 'utf-8') as f: df = np.loadtxt(f)
      if df.ndim<2: continue
      df[:,0] /= df[:,4]
      df[:,1] /= df[:,4]
      df[:,5] /= df[:,4]
      df[:,6] /= df[:,4]**3
      df[:,7] /= df[:,4]**2
      df[:,8] /= df[:,4]
      df[:,4] /= df[:,4]
      indVol = np.argmax(df[:,6])
      angl = 1 - df[indVol,2]/np.pi
      if 'ang' in cont: 
        if angl>185/180: x.append(np.nan)
        else: x.append(angl)
        z.append(df[indVol,0])
        zM.append(np.max(df[:indVol+1,0]))
        axInd=1
      if 'rad' in cont: 
        x.append(df[indVol,0])
        z.append( 1 - df[indVol,2]/np.pi )
        zM.append(np.min( 1 - df[:indVol+1,2]/np.pi ))
        axInd=0
      if 'bub' in cont: 
        x.append(df[indVol,5])
        z.append(df[indVol,0])
        zM.append(np.max(df[:indVol+1,0]))
        axInd=0
      dfDet.append(df[indVol,:])
      if '0.txt' not in fname: continue
      ax[axInd].plot(df[:indVol+1,6], -df[:indVol+1,1], c='lightgrey', lw=.5)
      #if 'ang' in cont and round(angl*100)%10!=0:continue
      if 'ang' in cont and round(angl*100)%20!=0:continue
      if 'ang' in cont and angl>50/180: ax[axInd].text(df[indVol,6], -df[indVol,1]+.05, rf"${angl:.1f}$", va='bottom', ha='center')
      #if 'rad' in cont and round(df[0,0]*10)%5!=0: continue 
      if 'rad' in cont and round(df[0,0]*10)%10!=5: continue 
      if 'rad' in cont and df[0,0]<3.2 and df[0,0]>.05: ax[axInd].text(df[indVol,6], -df[indVol,1]+.05, rf"${df[0,0]:.1f}$", va='bottom', ha='center')
      if 'bub' in cont and df[0,5]>.5 and df[0,5]<=1: ax[axInd].text(df[indVol,6], -df[indVol,1]+.05, rf"${df[0,5]:.1f}$", va='center', ha='left')
      ax[axInd].plot(df[:indVol+1,6], -df[:indVol+1,1], c='k', zorder=3)
      if 'rad' in cont and round(df[0,0]*10)%10==0: continue 
      if 'ang' in cont and round(angl*100)%20!=0:continue
      axRt = axProf[axInd].inset_axes((15/18.5, 2.6/3, (18.1-15)/18.5, .2/3))
      axRt.set_xscale('log')
      axRt.set_xlabel('$R_h/\\lambda$')
      axRt.set_xlim([.1,10])
      axRt.set_yticks([])
      axRt.tick_params(which='both', direction='in', top=True, right=True)
      for ri in range(21):
        Rt=10**( (ri-10)/10 )
        axRt.plot( (Rt,Rt), (0,1), lw=6, c=colRB(cont,Rt), zorder=-1)
      #axProf[axInd].plot( (5.5,5.5), (2.5,1.5), c='grey')
      #tri = RegularPolygon( (5.5,1.5), 3, radius=0.1, orientation=np.pi, color='grey', zorder=3)
      #axProf[axInd].add_patch(tri)
      #axProf[axInd].text(6, 2, '$g$', va='center', ha='left', c='grey')
      if 'rad' in cont: spac=(1,3.5,8,14.5,20,30,40,50)[ round(df[0,0]-.5) ] 
      if 'ang' in cont: spac=(.5,1.8,4.1,8,14.5,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160)[ round( 4-df[indVol,2]*5/np.pi) ]
      if spac>6 and spac<10: drawCoord=True
      else: drawCoord=False
      for hei in range(5):#5
        if drawCoord and hei<4: continue
        heiInd = np.argmin( abs( (hei+1)*df[indVol,1]/5 - df[:indVol+1,1] ) )
        #AdamsBashforthProfile(1, df[heiInd,5], fname=folName+f'prof{hei:05}'+fname)
        with open(folName+f'prof{hei:05}'+fname, encoding = 'utf-8') as f: prof = np.loadtxt(f)
        footInd = np.argmin( abs( df[heiInd,6] - prof[:,6] ))
        ax[axInd].plot(prof[footInd,6], -prof[footInd,1], 'o', ms=5, c=colRB( cont, df[heiInd,5] ), clip_on=False, zorder=4)#, mfc='None'
        xProf=np.concatenate(( -prof[:footInd,0][::-1] , prof[:footInd,0] ))
        xProf=xProf+spac
        yProf=np.concatenate(( prof[:footInd,1][::-1] - prof[footInd,1] , prof[:footInd,1] - prof[footInd,1] ))
        axProf[axInd].plot(xProf,yProf, c=colRB(cont,df[heiInd,5]), clip_on=False)
        #if hei!=4:continue
        #if 'rad' in cont: axProf[axInd].plot(xProf,yProf*0, c='w', clip_on=False)
        #if 'ang' in cont: continue
        #if round(df[0,0]-.5): continue
        if not drawCoord: continue
        xAn = xProf[-1]
        yAn = yProf[-1]
        #axProf[axInd].plot((xAn,xAn+.3), (yAn,yAn), color='grey')
        Xarr = []
        Yarr = []
        for phi in range(51):
          X = xAn + .17*np.cos(phi*np.pi/50)
          Y = yAn + .17*np.sin(phi*np.pi/50)
          for i in range(len(xProf)):
            if xProf[i]>X: break
          if yProf[i]>Y: break
          Xarr.append(X)
          Yarr.append(Y)
        axProf[axInd].plot(Xarr,Yarr,c='grey')
        if 'rad' in cont: axProf[axInd].text(xAn+.1, yAn+.1, "$\\phi_0$", ha='left', va='bottom', c='grey') 
        if 'ang' in cont: axProf[axInd].text(xAn+.1, yAn+.1, "$\\phi_c$", ha='left', va='bottom', c='grey') 
        axProf[axInd].plot((spac,xProf[0]), (0,0), 'o', ls='solid', color='grey', clip_on=False, zorder=3)
        #axProf[axInd].plot((spac,xProf[0]), (0,0), c='grey', clip_on=False, zorder=3)
        if 'rad' in cont: axProf[axInd].text( (spac+xProf[0])/2, .1, "$r_c$", ha='center', va='bottom', c='grey') 
        if 'ang' in cont: axProf[axInd].text( (spac+xProf[0])/2, .1, "$r_0$", ha='center', va='bottom', c='grey') 
        h=int(0.5*(len(xProf)))
        axProf[axInd].plot( (xProf[h],xProf[h]), (0,yProf[h]), c='grey')
        axProf[axInd].text( xProf[h]-.1, yProf[h]/2, "$h$", ha='right', va='center', c='grey') 
        axProf[axInd].plot(xProf[h],yProf[h],'o', c='grey')
        t=int(0.8*(len(xProf)))
        axProf[axInd].plot(xProf[h:t],yProf[h:t], c='grey')
        theta = np.arctan2(yProf[t+1]-yProf[t], xProf[t+1]-xProf[t])-np.pi/2
        tri = RegularPolygon( (xProf[t], yProf[t]), 3, radius=0.1, orientation=theta, color='grey', zorder=3)
        axProf[axInd].add_patch(tri)
        axProf[axInd].text(xProf[t]+0.1, yProf[t], '$s$', va='center', ha='left', c='grey')
        t=int(0.63*(len(xProf)))
        xAn = xProf[t]
        yAn = yProf[t]
        axProf[axInd].plot((xAn,xAn+.3), (yAn,yAn), color='grey')
        Xarr = []
        Yarr = []
        for phi in range(51):
          X = xAn + .17*np.cos(phi*np.pi/50)
          Y = yAn + .17*np.sin(phi*np.pi/50)
          for i in range(len(xProf)):
            if xProf[i]>X: break
          if yProf[i]>Y: break
          Xarr.append(X)
          Yarr.append(Y)
        axProf[axInd].plot(Xarr,Yarr,c='grey')
        axProf[axInd].text(xAn+.1, yAn+.1, "$\\phi$", ha='left', va='bottom', c='grey') 
        gravX=18
        gravTailY=1.8
        gravHeadY=1
        axProf[axInd].plot([gravX,gravX],[gravTailY,gravHeadY], c='grey')
        tri = RegularPolygon( (gravX, gravHeadY), 3, radius=0.1, orientation=np.pi, color='grey', zorder=3)
        axProf[axInd].add_patch(tri)
        axProf[axInd].text(gravX+0.1, (gravHeadY+gravTailY)/2, '$g$', va='center', ha='left', c='grey')
    x = np.asarray(x)
    dfDet = np.asarray(dfDet)
    z = np.asarray(z)
    ax2.plot(x,dfDet[:,5],c='b')
    ax2.plot(x,-dfDet[:,1],c='k')
    ax2.set_ylim([0,3.219])
    axV = axVV[0]
    axM = axVV[1]
    maxVind=np.argmax(dfDet[:,6])
    print('maxVol',dfDet[maxVind,:])
    maxVind=np.argmax(-dfDet[:,1])
    print('maxHeight',dfDet[maxVind,:])
    axM.tick_params(direction='in')
    ax[axInd].tick_params(which='both', direction='in', top=True, right=True)
    ax[axInd].set_xlabel('$V/\\lambda^3$')
    axM.text(5e-3,.99,'$\\mathrm{(b)}$',transform=axM.transAxes,va='top',ha='left')
    ax2.tick_params(which='both', direction='in', top=True, right=True)
    ax[0].set_ylabel('$\\frac{h}{\\lambda}$',rotation=0,size=22,labelpad=10)
    axProf[axInd].tick_params(which='both', direction='in', top=True, right=True)
    axProf[axInd].set_ylabel('$\\frac{ z }{\\lambda}$',rotation=0,size=22,labelpad=15)
    axProf[axInd].set_ylim([0,3])
    axProf[axInd].set_xlim([0,18.5])
    axProf[axInd].set_aspect('equal', adjustable='box')
    ax[axInd].set_ylim([0,3])
    ax[axInd].set_xlim([0,20])
    axV.set_ylim([0,30])
    axV.tick_params(which='both', direction='in', top=True, right=True)
    axM.tick_params(which='both', direction='in', top=True, right=True)
    figV.set_figwidth(6)
    fig2.set_figwidth(5)
    figV.set_figheight(6)
    fig2.set_figheight(3)
    if 'bub' in cont:
      axV.plot(x,dfDet[:,6],c='k',clip_on=False)
      axV.set_xlabel('$R_t$')
      ax2.plot(x,z, c='k',clip_on=False)
    if 'ang' in cont:
      axV.plot(x,dfDet[:,6],c='b',clip_on=False)
      axI = inset_axes(axV, width="40%", height="50%", loc='upper left')
      axI.yaxis.set_label_position("right")
      axI.yaxis.tick_right()
      axI.tick_params(which='both', direction='in', top=True, left=True, right=True, pad=6)
      axI.set_xscale('log')
      axI.set_yscale('log')
      axI.set_xlim([.07,1])
      axI.set_ylim([.01,100])
      axI.plot(x,dfDet[:,6],c='b')
      axM.plot( x[::30], z[::30], '.', c='b', clip_on=False, zorder=3)
      axM.plot( x, zM, '-', c='b', clip_on=False, zorder=3)
      fig2.subplots_adjust(left=0.1, right=0.97, bottom=0.2, top=0.98)
      #figProf.subplots_adjust(left=0.05, right=0.97, bottom=0.2, top=0.98)
      xx=np.linspace(0,1)
      axV.plot(xx, 4*np.pi*(.0104*xx*180)**3/3, ls='dashed', c='k')
      axI.plot(xx, 4*np.pi*(.0104*xx*180)**3/3, ls='dashed', c='k')
      axM.plot(xx, 3.219*xx**2, ls='dashed', c='k', zorder=3)
      #axM.plot(xx, .887*(np.pi*xx)**3 /2/np.pi/np.sin(np.pi*xx), ls='solid', c='k')
      #axM.plot(xx, np.sqrt( 6 * np.sin(np.pi*xx) * np.cos(np.pi*xx)**3 / (2 + np.sin(np.pi*xx) ) / (1 - np.sin(np.pi*xx) )**2 ), ls='dashed', c='k')
      print(4*np.pi*(.0104*180)**3/3, 'dotted')
      axM.set_xlabel('$\\phi_c/\\pi$')
      ax2.set_xlabel('$\\phi_c/\\pi$')
      ax2.text(-.08,.7,'$\\frac{h}{\\lambda}$',c='k',transform=ax2.transAxes,size=22,ha='center')
      ax2.text(-.08,.5,'$\\frac{R_t}{\\lambda}$',c='b',transform=ax2.transAxes,size=22,ha='center')
      ax2.text(-.08,.3,'$\\frac{r_0}{\\lambda}$',c='r',transform=ax2.transAxes,size=22,ha='center')
      axProf[axInd].text(5e-3,.99,'$\\mathrm{(b)}$',transform=axProf[axInd].transAxes,va='top',ha='left')
      ax[axInd].text(5e-3,.99,'$\\mathrm{(b)}$',transform=ax[axInd].transAxes,va='top',ha='left')
      ax2.set_xlim([0,1])
      ax2.plot(x,z, c='r',clip_on=False)
      axV.set_xlim([0,1])
      axV.text(1-5e-3,.99,'$\\mathrm{(a)}$',transform=axV.transAxes,va='top',ha='right')
      axV.set_ylabel('$\\frac{V_s}{\\lambda^3}$',size=22,rotation=0,labelpad=15)
      axM.set_ylabel('$\\frac{r_0}{\\lambda}$',size=22,rotation=0,labelpad=10)
      axM.set_ylim([0,4])
      fname = 'demirkir24life.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f, skiprows=1)
      for i in range(len(df[:,0])):
        rad = df[i,1]*1e-6
        density = df[i,2] -	0.08988*1e-6
        surf = df[i,3]*1e-3
        capLen = (surf/density/9.81)**.5
        mid = (df[i,0]+df[i,4])/2/180
        if df[i,0]-mid*180 > 20: continue
        print(i, [df[i,0]-mid])
        axV.errorbar( mid, 4*np.pi/3 * rad**3 / capLen**3, xerr=[ [df[i,0]/180-mid], [mid-df[i,4]/180] ], fmt='^', c='b', mfc='None',clip_on=False, zorder=3)
        axI.errorbar( mid, 4*np.pi/3 * rad**3 / capLen**3, xerr=[ [df[i,0]/180-mid], [mid-df[i,4]/180] ], fmt='^', c='b', mfc='None',clip_on=False, zorder=3)
      fname = 'allred21role.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f, skiprows=1)
      for i in range(len(df[:,0])):
        if (max(df[i,2:]) - min(df[i,2:])) > 20: continue
        if max(df[i,2:]) < 20: continue
        rad = df[i,1]/2
        capLen = df[i,0]/df[i,4]/.0208/2**.5
        vol = 4*np.pi/3 * rad**3 / capLen**3
        mn=df[i,2]/180
        mid=df[i,4]/180
        mx=df[i,3]/180
        if mid<mn: continue
        if mid>mx: continue
        axV.plot(mid, vol, 'v', c='b', mfc='None', zorder=3)
        axV.plot([mn,mx], [vol,vol], c='b', zorder=3)
        axI.plot(mid, vol, 'v', c='b', mfc='None', zorder=3)
        axI.plot([mn,mx], [vol,vol], c='b', zorder=3)
      fname = 'huang25effects.txt'
      print('open',fname)
      surf=72.25e-3
      density=998
      capLen = (surf/density/9.81)**.5
      with open(fname) as f: df = np.loadtxt(f, skiprows=1)
      for i in range(len(df[:,0])):
        if df[i,0]<50: continue
        axV.errorbar(df[i,0]/180, df[i,3]/capLen**3, xerr=[ [ df[i,1]/180-df[i,0]/180 ] , [ df[i,0]/180-df[i,2]/180 ] ], fmt='d', c='b', mfc='None', clip_on=False, zorder=3)
        axI.errorbar(df[i,0]/180, df[i,3]/capLen**3, xerr=[ [ df[i,1]/180-df[i,0]/180 ] , [ df[i,0]/180-df[i,2]/180 ] ], fmt='d', c='b', mfc='None', clip_on=False, zorder=3)
      #ax[axInd].set_yticklabels([])
      #ax[axInd].set_ylabel('')
      #fig.subplots_adjust(left=0.03, right=0.86, bottom=0.2, top=0.98)
      rads = np.pi*np.arange(0.8, -0.1, -0.2)
      print('rads',rads/np.pi)
    if 'rad' in cont:
      axV.plot((3.832,*x),(0,*dfDet[:,6]),c='r',clip_on=False)
      from mpl_toolkits.axes_grid1.inset_locator import inset_axes
      axI = inset_axes(axV, width="40%", height="50%", loc='upper left')
      axI.yaxis.set_label_position("right")
      axI.yaxis.tick_right()
      axI.tick_params(which='both', direction='in', top=True, left=True, right=True)
      axI.set_xscale('log')
      axI.set_yscale('log')
      axI.set_xlim([6e-2,1.2])
      axI.set_ylim([.3,8])
      axI.plot(x,dfDet[:,6],c='r')
      #axM.plot( x, z, c='r', clip_on=False, zorder=3)
      #axM.plot( (x[:-2]+x[1:-1]+x[2:])/3, (z[:-2]+z[1:-1]+z[2:])/3, c='r', clip_on=False, zorder=3)
      axM.plot( x[::15], z[::15], '.', c='r', clip_on=False, zorder=3)
      axM.plot( x, zM, c='r', clip_on=False, zorder=3)
      fig2.subplots_adjust(left=0.1, right=0.88, bottom=0.2, top=0.98)
      xx=np.linspace(0,4)
      axV.plot( xx, 2*np.pi*xx, linestyle='dashed', c='k')
      axI.plot( xx, 2*np.pi*xx, linestyle='dashed', c='k')
      axM.plot( xx, (xx/3.5)**.5, linestyle='dashed', c='k', zorder=3)
      axM.set_xlabel('$r_c/\\lambda$')
      #axI.set_xlabel('$r_c/\\lambda$',labelpad=-5)
      ax2.set_xlabel('$r_c/\\lambda$')
      axP = ax2.twinx()
      axP.tick_params(direction='in')
      ax2.tick_params(right=False)
      ax2.text(-.08,.4,'$\\frac{R_t}{\\lambda}$',c='b',transform=ax2.transAxes,size=22,ha='center')
      ax2.text(-.08,.6,'$\\frac{h}{\\lambda}$',c='k',transform=ax2.transAxes,size=22,ha='center')
      ax2.text(1.12,.5,'$\\frac{\\phi_0}{\\pi}$',c='r',transform=ax2.transAxes,size=22,ha='center')
      axP.set_ylim([.5,1])
      axM.set_ylim([0,1.05])
      axM.set_yticks([0,.25,.5,.75,1])
      axV.axvspan(3.219, 4, color='lightgrey')
      axV.text(1-5e-3,.99,'$\\mathrm{(a)}$',transform=axV.transAxes,va='top',ha='right')
      axV.set_ylabel('$\\frac{V_p}{\\lambda^3}$',size=22,rotation=0,labelpad=15)
      #axI.set_ylabel('$\\frac{V_p}{\\lambda^3}$',size=22,rotation=0,labelpad=15)
      axM.axvspan(3.219, 4, color='lightgrey')
      axM.set_ylabel('$\\frac{\\phi_0}{\\pi}$',size=22,rotation=0,labelpad=10)
      axProf[axInd].text(5e-3,.99,'$\\mathrm{(a)}$',transform=axProf[axInd].transAxes,va='top',ha='left')
      ax[axInd].text(5e-3,.99,'$\\mathrm{(a)}$',transform=ax[axInd].transAxes,va='top',ha='left')
      ax2.set_xlim([0,4])
      axP.plot( x, z, c='r',clip_on=False, zorder=3)#,'.',ms=5
      fname = 'LesageVolVsContRadSq.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f)
      for i in range(len(df[:,0])):
        if df[i,2]>1:continue
        axV.plot(df[i,0]**.5, df[i,1]*df[i,0]**1.5, 's', mec='r', mfc='None', clip_on=False, zorder=3)
        axI.plot(df[i,0]**.5, df[i,1]*df[i,0]**1.5, 's', mec='r', mfc='None', zorder=3)
      fname = 'MoriVolByContCubeVsContSqByCapSq.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f)
      axV.plot(.5/df[:,0]**.5, df[:,1]/df[:,0]**1.5, 'd', mec='r', mfc='None', clip_on=False, zorder=3)
      axI.plot(.5/df[:,0]**.5, df[:,1]/df[:,0]**1.5, 'd', mec='r', mfc='None', zorder=3)
      fname = 'sasetty23stability.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f)
      axV.plot(df[:,2]/df[:,3]/2, df[:,1]/(df[:,3]*1e-3)**3, 'v', mec='r', mfc='None', clip_on=False)
      axI.plot(df[:,2]/df[:,3]/2, df[:,1]/(df[:,3]*1e-3)**3, 'v', mec='r', mfc='None')
      fname = 'gunde01measurement.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f, skiprows=2)
      capLen=(df[:,2]*1e-3/df[:,1]/9.81)**.5
      axV.plot(df[:,0]*1e-3/capLen, df[:,4]*1e-6*1e-3/capLen**3, '^', mec='r', mfc='None', clip_on=False)
      axI.plot(df[:,0]*1e-3/capLen, df[:,4]*1e-6*1e-3/capLen**3, '^', mec='r', mfc='None')
      axV.set_xlim([0,4])
      #fig.subplots_adjust( left=0.14, right=0.97, bottom=0.2, top=0.98)
    fname = folName+'MaxVolVs_'+cont+'.pdf'
    print('savin ',fname)
    figV.savefig(fname, transparent=True, format='pdf', bbox_inches='tight', pad_inches=0)
    fname = folName+'ax2_'+cont+'.pdf'
    print('savin ',fname)
    fig2.savefig(fname, transparent=True, format='pdf')
  fname = folName+'heightVsVol_'+cont+'.pdf'
  print('savin ',fname)
  fig.set_figwidth(10)
  fig.set_figheight(3)
  fig.tight_layout(pad=.7)
  fig.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0)
  figProf.set_figwidth(12)
  axProf[1].set_xlabel('$x/\\lambda$',labelpad=-5)
  figProf.subplots_adjust(hspace=-.05)
  outName = folName+f'pin.pdf'
  print('savin ',outName)
  figProf.savefig(outName, transparent=True, bbox_inches='tight', pad_inches=0)
  return

def plot_drop_size_vs_rad(): 
  import os
  folName = 'data/'
  fig, ax = plt.subplots(1)
  figAng, axAng = plt.subplots(1)
  x=np.linspace(0,5)
  axAng.plot(x, (1.5*x)**(1./3), linestyle='dotted', c='k')
  radBase=[]
  radDeta=[]
  for fname in sorted(os.listdir(folName)):
    if 'txt' not in fname: continue
    if '0.txt' not in fname and '2.txt' not in fname and '4.txt' not in fname and '6.txt' not in fname and '8.txt' not in fname: continue
    if 'rad' not in fname: continue
    #print(r, -z, psi, dPsi, capLen, RadTop, Volume, area, centroid, file=ang_txt)
    with open(folName+fname, encoding = 'utf-8') as f:
      print('plot',folName+fname)
      df = np.loadtxt(f)
    if df.ndim<2: continue
    #col=( min(df[0,0]/5, 1), 0, 0)
    col=( min(df[0,0]/np.pi, 1), 0, 0)
    #ind = np.argsort(df[:,5])
    ind = np.argsort(df[:,1])
    indVol = np.argmax(df[:,6])
    #ax.plot(df[ind,5], (df[ind,6]*3/4/np.pi)**(1/3), '.', color=col, ms=.2)
    #ax.plot((df[ind,6]*3/4/np.pi)**(1/3), -df[ind,1], color=col)
    for i in range(0):#len(df[:,6])):
      #if df[indVol,1]>df[i,1]: df[i,6]=np.nan
      #if df[indVol,5]>df[i,5]: df[i,6]=np.nan
      if df[indVol,1]/df[indVol,6]**.333>df[i,1]/df[i,6]**.333: df[i,6]=np.nan
      #if df[i,2]<np.pi/2: col='r'
      #else: col='b'
      #ax.plot((df[i,6]*3/4/np.pi)**(1/3), -df[i,1], '.', color=col, ms=.2)
    ax.plot((df[ind,6]*3/4/np.pi)**(1/3), -df[ind,1], color=col)
    radBase.append(df[indVol,0])
    radDeta.append((df[indVol,6]*3/4/np.pi)**(1/3))
  axAng.plot(radBase, radDeta, color='k')
  fname = 'LesageVolVsContRadSq.txt'
  print('open',fname)
  with open(fname) as f:
    df = np.loadtxt(f)
  for i in range(len(df[:,0])):
    if df[i,2]>1:continue
    axAng.plot(df[i,0]**.5, (.75*df[i,1]/np.pi)**(1/3)*df[i,0]**.5, 's', mec=(0,0,df[i,2]/3), mfc='None', clip_on=False)
  fname = 'MoriVolByContCubeVsContSqByCapSq.txt'
  print('open',fname)
  with open(fname) as f:
    df = np.loadtxt(f)
  axAng.plot(.5/df[:,0]**.5, (.75*df[:,1]/np.pi)**(1/3)/df[:,0]**.5, 'd', mec=(0,0,0), mfc='None', clip_on=False)
  ax.tick_params(which='both', direction='in', top=True, right=True)
  ax.set_xlabel('$R_V/\\lambda$')
  ax.set_ylabel('$\\frac{h}{\\lambda}$',rotation=0,size=22)
  ax.set_ylim([0,4])
  ax.set_xlim([0,2])
  #ax.set_xscale('log')
  axAng.tick_params(which='both', direction='in', top=True, right=True)
  axAng.set_xlabel('$R_b$')
  axAng.set_ylabel('$\\frac{ R_d }{\\lambda}$',rotation=0,size=22)
  fname = folName+'RadSphVsRadTopBase.pdf'
  print('savin ',fname)
  fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf')
  #fname = folName+'MaxRadVsBaseRad.pdf'
  #print('savin ',fname)
  #figAng.savefig(fname, bbox_inches='tight', transparent=True, format='pdf')
  return

def plot_centroid_vs_grid_size(): 
  fig, ax = plt.subplots(1)
  ABcentroid = 0.4728934922628638
  col=BoColour(4)
  for res in (1,2,4,8,16):
    fname = 'radTopBy'+str(res)+'/vol.txt'
    print('open',fname)
    with open(fname) as f:
      for line in f:
        pass
    centroid = float(line.split()[7])
    print(centroid)
    ax.plot(res, centroid/ABcentroid-1, '.', markersize=100/res, mec=col, mfc='None')#, mew=.2, alpha=.5)
  ax.set_xlabel('$R/l$', rotation=0, labelpad=-5)
  #ax.set_ylabel('$\\frac{\\langle z_l\\rangle-\\langle z_0\\rangle}{\\langle z_0\\rangle}$', rotation=0, size=20)
  ax.set_ylabel('$\\langle z_l\\rangle/\\langle z_0\\rangle-1$')
  ax.set_xscale('log')
  ax.set_yscale('log')
  fname = 'data/centroidVsGridSize.pdf'
  print('savin ',fname)
  fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
  return

def plot_detach_radius_vs_cont_angle(): 
  import os
  folName='data/'
  BoProfile=[]
  for fname in sorted(os.listdir(folName)):
    if 'pin' not in fname: continue
    if 'txt' not in fname: continue
    with open(folName+fname, encoding = 'utf-8') as f:
      BoProfile.append(float(f.readline().strip().split()[-1]))
  try: BoProfile.remove(0.566)
  except ValueError as e: print(f"Could not remove 0.566: {e}")
  fig, ax = plt.subplots(1)
  fname = 'data/fritz.txt'
  print('open',fname)
  with open(fname) as f:
    df = np.loadtxt(f)
  #x=np.concatenate(([0],df[:,1]/np.pi,[1]))
  #y=np.concatenate(([0],df[:,2],[0]))
  x=df[:,1]/np.pi*180
  y=df[:,2]
  ax.plot(x, y, c='k', clip_on=False)
  ax.plot(x, .0104*x, '--', c='k')
  R = lambda p: 3**.5 * np.sin(p) / 2**(1/6.) / (1-np.cos(p)) / (2+np.cos(p))**.5 
  #ax.plot(180-x, R(x*np.pi/180), linestyle='dotted', c='k')
  for i in range(len(df[:,0])):
    if df[i,0] in BoProfile:
      ax.plot(df[i,1]/np.pi*180, df[i,2], 'o', mec=BoColour(df[i,0]), mfc='None', clip_on=False)
      va='top'
      xShift=0
      if df[i,0]>65: continue
      elif df[i,0]<.06: continue
      elif df[i,0]<1:
        ha='left'
        va='center'
        xShift=.015*180
      elif df[i,0]>8: ha='right'
      else: ha='center'
      if 1.2<df[i,0] and df[i,0]<1.3:  xShift=.015
      ax.text(df[i,1]/np.pi*180+xShift, df[i,2]-.015, f'${df[i,0]:.4g}$', ha=ha, va=va)
  if True:
    fname = 'Ling25effect.txt'
    print('open',fname)
    with open(fname) as f:
      df = np.loadtxt(f)
    for i in range(len(df[:,0])):
      #ax.plot(df[i,2], (.75*df[i,4]/np.pi)**(1/3)/df[i,5], 's', mec=(0,0,df[i,2]/3), mfc='None', clip_on=False)
      ax.plot(df[i,2], (.75*df[i,4]*1e-6/np.pi)**(1/3)/df[i,5]/1e-3, 's', mec='k', mfc='None', clip_on=False)
  ax.tick_params(which='both', direction='in', top=True, right=True)
  ax.set_xlabel('$\\phi$', rotation=0)
  #ax.set_ylabel('$\\frac{R_{det}}{\\lambda}$', rotation=0, size=22, labelpad=15)
  ax.set_ylabel('$R_\\mathrm{det}\\sqrt\\frac{\\rho g}{\\sigma}$')#, rotation=0, size=22, labelpad=15)
  ax.set_xlim([0,180])
  #ax.set_ylim([0,1])
  #ax.set_xscale('log')
  #ax.set_yscale('log')
  degrees = [0, 30, 60, 90, 120, 150, 180]
  ax.set_xticks(degrees)
  ax.set_xticklabels([f"{d}°" for d in degrees])
  fname = 'data/detachRadVsContAngle.pdf'
  print('savin ',fname)
  fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
  return

def plot_detach_radius_vs_cont_radius(): 
  import os
  folName='data/'
  fig, ax = plt.subplots(1)
  BoProfile=[]
  for fname in sorted(os.listdir(folName)):
    if 'spread' not in fname: continue
    if 'txt' not in fname: continue
    with open(folName+fname, encoding = 'utf-8') as f:
      BoProfile.append(float(f.readline().strip().split()[-1]))
  fname = 'data/fritz.txt'
  print('open',fname)
  with open(fname) as f:
    df = np.loadtxt(f)
  #x=np.concatenate(([0],df[:,4],[df[-1,4]]))
  #y=np.concatenate(([0],df[:,5],[0]))
  x=df[:,4]
  y=df[:,5]
  ax.plot(x, y, c='k', clip_on=False)
  #ax.plot(x, (1.5*x)**(1./3), linestyle='dotted', c='k')
  for i in range(len(df[:,0])):
    if df[i,0] in BoProfile:
      logBo = int(np.log2(df[i,0]))
      ax.plot(df[i,4], df[i,5], 'o', mec=BoColour(df[i,0]), mfc='None', clip_on=False )
      va='top'
      yShift=-.015
      if logBo>6: continue
      elif logBo<-4: continue
      elif logBo<-1: ha='left'
      elif logBo>3: ha='right'
      else: ha='center'
      if df[i,0]==0.566: 
        va='bottom'
        yShift=0.005
      ax.text(df[i,4], df[i,5]+yShift, f'${df[i,0]:.4g}$', ha=ha, va=va)
  if True:
    fname = 'LesageVolVsContRadSq.txt'
    print('open',fname)
    with open(fname) as f:
      df = np.loadtxt(f)
    for i in range(len(df[:,0])):
      if df[i,2]>1:continue
      ax.plot(df[i,0]**.5, (.75*df[i,1]/np.pi)**(1/3)*df[i,0]**.5, 's', mec=(0,0,df[i,2]/3), mfc='None', clip_on=False)
    fname = 'MoriVolByContCubeVsContSqByCapSq.txt'
    print('open',fname)
    with open(fname) as f:
      df = np.loadtxt(f)
    ax.plot(.5/df[:,0]**.5, (.75*df[:,1]/np.pi)**(1/3)/df[:,0]**.5, 'd', mec=(0,0,0), mfc='None', clip_on=False)
  ax.tick_params(which='both', direction='in', top=True, right=True)
  ax.set_xlabel('$r_\\mathrm{con}\\sqrt\\frac{\\rho g}{\\sigma} $')
  #ax.set_ylabel('$R_\\mathrm{det}\\sqrt\\frac{\\rho g}{\\sigma}$', rotation=0, size=22, labelpad=15)
  ax.set_ylabel('$R_\\mathrm{det}\\sqrt\\frac{\\rho g}{\\sigma}$')#, rotation=0, size=22, labelpad=15)
  ax.set_xlim([0,2.5])
  ax.set_ylim([0,1.3])
  fname = 'data/detachRadVsContRad.pdf'
  print('savin ',fname)
  fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
  return


