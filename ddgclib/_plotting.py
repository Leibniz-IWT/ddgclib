import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
#from ipywidgets import *
from matplotlib.widgets import Slider
import polyscope as ps
from ddgclib._misc import *  # coldict is neeeded
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

def plot_drop_profile(name='pin spread'): 
  import os
  folName = 'data/'
  for cont in name.split():
    fig, ax = plt.subplots(1)
    for fname in sorted(os.listdir(folName)):
      if 'txt' not in fname: continue
      if cont not in fname: continue
      with open(folName+fname, encoding = 'utf-8') as f:
        df = np.loadtxt(f)
      if df.ndim<2: continue
      if 'spread' in cont and df[0,-1]==.566: continue
      col=BoColour(df[0,-1])
      #ax.plot(df[:,0]/df[-1,0], (df[:,1]-df[-1,1])/df[-1,0], 'o', color=col)#, alpha=.9)
      ax.plot(df[0,0], df[0,1], '+', color=col)#, alpha=.9)
      ax.plot(df[:,0], df[:,1], color=col)#, alpha=.9)
      logBo = int(np.log2(df[-1,-1]))
      if False:#logBo==-2: 
        if 'spread' in cont:
          ax.plot([df[-1,0],df[-1,0]+.15], [df[-1,1],df[-1,1]], color='k', linestyle='dashed')
          ax.text(df[-1,0]+.15, df[-1,1]+.1, f'$\\phi$', ha='center', va='center')
        elif 'pin' in cont:
          ax.plot([df[-1,0],0], [df[-1,1],df[-1,1]], color='k', linestyle='dashed')
          ax.text(df[-1,0]/2, df[-1,1]+.1, '$r_\\mathrm{con}$', ha='center', va='center')
      if logBo>2: continue
      if logBo<-1: continue
      #ax.text(*df[-1,:2], f'${df[-1,-1]**.5:.4g}$', ha='left', va='center')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    #ax.set_xlabel('$r/R_\\mathrm{top}$')
    #ax.set_xlabel('$r/r_\\mathrm{con}$')
    ax.set_xlabel('$r/\\lambda$')
    #ax.set_ylabel('$\\frac{ z-z_\\mathrm{top} }{ R_\\mathrm{top} }$',rotation=0,size=22)
    #ax.set_ylabel('$\\frac{ z }{ r_\\mathrm{con} }$',rotation=0,size=22,labelpad=15)
    ax.set_ylabel('$\\frac{ z }{\\lambda}$',rotation=0,size=22)
    ax.set_ylim([-2,2])
    #ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.set_aspect('equal', adjustable='box')
    fname = folName+'profile_'+cont+'.pdf'
    print('savin ',fname)
    fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
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
  ax.tick_params(which='both', direction='in', top=True, right=True)
  ax.set_xlabel('$\\phi$', rotation=0)
  #ax.set_ylabel('$\\frac{R_{det}}{\\lambda}$', rotation=0, size=22, labelpad=15)
  ax.set_ylabel('$R_\\mathrm{det}\\sqrt\\frac{\\rho g}{\\sigma}$')#, rotation=0, size=22, labelpad=15)
  ax.set_xlim([0,180])
  ax.set_ylim([0,1])
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
  if False:
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
