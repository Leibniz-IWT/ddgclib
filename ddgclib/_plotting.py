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
  folName = 'data/'
  for cont in nam.split():
    fig, ax = plt.subplots(1)
    figV, axVV = plt.subplots(2, sharex=True)
    plt.subplots_adjust(hspace=0.07)
    fig2, ax2 = plt.subplots(1)
    figProf, axProf = plt.subplots(1)
    x=[]
    y=[]
    z=[]
    zM=[]
    rTopDet=[]
    heigDet=[]
    volDet=[]
    radInd=0 
    #rads=np.array((.5,1,1.5,2,2.5,3,3.5))
    rads=np.array((0.5,1.5,2.5,3.5))
    for fname in reversed(sorted(os.listdir(folName))):
      if 'Lo' in fname: continue
      if 'Hi' in fname: continue
      if 'txt' not in fname: continue
      if cont not in fname: continue
      with open(folName+fname, encoding = 'utf-8') as f:
        df = np.loadtxt(f)
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
      if 'rad' in cont: 
        x.append(df[indVol,0])
        z.append( 1 - df[indVol,2]/np.pi )
        zM.append(np.min( 1 - df[:indVol+1,2]/np.pi ))
      if 'bub' in cont: 
        x.append(df[indVol,5])
        z.append(df[indVol,0])
        zM.append(np.max(df[:indVol+1,0]))
      r = df[:,6]
      y.append(r[indVol])
      height = -df[:,1]
      #height = df[:,0]
      #height = 1 - df[:,2]/np.pi
      #height = df[:,5]
      rTopDet.append(df[indVol,5])
      #rTopDet.append(np.max(df[:indVol+1,5]))
      heigDet.append(-df[indVol,1])
      #heigDet.append(np.max(-df[:indVol+1,1]))
      volDet.append(df[indVol,6])
      for i in range(0):#1, len(r)):
        if abs(r[i]-r[i-1]) > 3: r[i-1]=np.nan
      zord=3
      if '0.txt' not in fname: continue
      if 'ang' in cont: 
        if angl>181/180: continue
        if round(angl*100)%10==0:
          col='k'
          if angl>50/180:
            ax.text(r[indVol], height[indVol], rf"${angl:.1f}$", va='bottom', ha='center')
            zord=3
        else: 
          col='silver'
          zord=1
      if 'rad' in cont: 
        if round(df[0,0]*10)%5==0 and df[0,0]<3.7 and df[0,0]>.05:
          col='k'
          ax.text(r[indVol], height[indVol], rf"${df[0,0]:.1f}$", va='bottom', ha='center', c=col)
          print(r[indVol], height[indVol], rf"${df[0,0]}$", fname)
          zord=3
        else: 
          col='silver'
          zord=1
      if 'bub' in cont: 
        if '9.txt' in fname or df[0,5]>5:
          col='k'
          zord=3
          if df[0,5]>.5 and df[0,5]<=1: ax.text(r[indVol], height[indVol], rf"${df[0,5]:.1f}$", va='center', ha='left')
        else: 
          col='silver'
          zord=1
      ax.plot(r[:indVol+1], height[:indVol+1], c=col, zorder=zord)
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    ax2.plot(x,rTopDet,c='b')
    ax2.plot(x,heigDet,c='k')
    ax2.set_ylim([0,3.219])
    #idx = np.argsort(x)
    axV = axVV[0]
    axM = axVV[1]
    #axV.plot(x,y,c='k',clip_on=False)
    maxVind=np.argmax(y)
    print('x',x[0],x[maxVind],x[-1])
    print('y',y[0],y[maxVind],y[-1])
    print('rTopDet',rTopDet[0],rTopDet[maxVind],rTopDet[-1])
    #axM = axV[1]#.twinx()
    #axM.set_zorder(axV.get_zorder() - 1)
    axM.tick_params(direction='in')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_xlabel('$V/\\lambda^3$')
    axV.text(5e-3,.99,'$\\mathrm{(a)}$',transform=axV.transAxes,va='top',ha='left')
    axM.text(5e-3,.99,'$\\mathrm{(b)}$',transform=axM.transAxes,va='top',ha='left')
    #figV.subplots_adjust(left=.1, right=.88, bottom=.18,top=.96)
    #axV.text(-.1,.5,'$\\frac{V}{\\lambda^3}$',transform=axV.transAxes,size=22,ha='center')
    axV.set_ylabel('$\\frac{V}{\\lambda^3}$',size=22,rotation=0,labelpad=15)
    ax2.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_ylabel('$\\frac{h}{\\lambda}$',rotation=0,size=22,labelpad=10)
    axProf.tick_params(which='both', direction='in', top=True, right=True)
    axProf.set_ylabel('$\\frac{ z }{\\lambda}$',rotation=0,size=22,labelpad=15)
    axProf.set_ylim([0,2.5])
    #axProf.set_xlim([0,14])
    axProf.set_xlim([0,18.5])
    axProf.set_aspect('equal', adjustable='box')
    ax.set_ylim([0,3])
    ax.set_xlim([0,20])
    axV.set_ylim([0,30])
    axV.tick_params(which='both', direction='in', top=True, right=True)
    axM.tick_params(which='both', direction='in', top=True, right=True)
    fig.set_figwidth(5)
    figV.set_figwidth(5)
    fig2.set_figwidth(5)
    figProf.set_figwidth(10)
    fig.set_figheight(3)
    figV.set_figheight(6)
    fig2.set_figheight(3)
    figProf.set_figheight(2.3)
    if 'bub' in cont:
      axV.plot(x,y,c='k',clip_on=False)
      axV.set_xlabel('$R_t$')
      axV.set_xscale('log')
      axV.set_yscale('log')
      ax2.plot(x,z, c='k',clip_on=False)
    if 'ang' in cont:
      axV.plot(x,y,c='b',clip_on=False)
      axM.plot( x, z, c='b', clip_on=False, zorder=3)
      axM.plot( x, zM, '--', c='b', clip_on=False, zorder=3)
      fig2.subplots_adjust(left=0.1, right=0.97, bottom=0.2, top=0.98)
      figProf.subplots_adjust(left=0.05, right=0.97, bottom=0.2, top=0.98)
      xx=np.linspace(0,1)
      axV.plot(xx, 4*np.pi*(.0104*xx*180)**3/3, ls='dotted', c='b')
      print(4*np.pi*(.0104*180)**3/3, 'dotted')
      axM.set_xlabel('$\\phi_0/\\pi$')
      ax2.set_xlabel('$\\phi_0/\\pi$')
      ax2.text(-.08,.7,'$\\frac{h}{\\lambda}$',c='k',transform=ax2.transAxes,size=22,ha='center')
      ax2.text(-.08,.5,'$\\frac{R_t}{\\lambda}$',c='b',transform=ax2.transAxes,size=22,ha='center')
      ax2.text(-.08,.3,'$\\frac{r_0}{\\lambda}$',c='r',transform=ax2.transAxes,size=22,ha='center')
      axProf.text(5e-3,.99,'$\\mathrm{(b)}$',transform=axProf.transAxes,va='top',ha='left')
      ax.text(5e-3,.99,'$\\mathrm{(b)}$',transform=ax.transAxes,va='top',ha='left')
      ax2.set_xlim([0,1])
      ax2.plot(x,z, c='r',clip_on=False)
      axV.set_xlim([0,1])
      #axV.text(1.08,.5,'$\\frac{r_0}{\\lambda}$',c='r',transform=axV.transAxes,size=22,ha='center')
      axM.set_ylabel('$\\frac{r_0}{\\lambda}$',size=22,rotation=0,labelpad=10)
      axM.set_ylim([0,4])
      fname = 'demirkir24life.txt'
      print('open',fname)
      with open(fname) as f:
        df = np.loadtxt(f, skiprows=1)
      for i in range(len(df[:,0])):
        rad = df[i,1]*1e-6
        density = df[i,2] -	0.08988*1e-6
        surf = df[i,3]*1e-3
        capLen = (surf/density/9.81)**.5
        mid = (df[i,0]+df[i,4])/2/180
        if df[i,0]-mid*180 > 20: continue
        print(i, [df[i,0]-mid])
        axV.errorbar( mid, 4*np.pi/3 * rad**3 / capLen**3, xerr=[ [df[i,0]/180-mid], [mid-df[i,4]/180] ], fmt='^', c='b', mfc='None',clip_on=False, zorder=3)
      fname = 'allred21role.txt'
      print('open',fname)
      with open(fname) as f:
        df = np.loadtxt(f, skiprows=1)
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
      fname = 'huang25effects.txt'
      print('open',fname)
      surf=72.25e-3
      density=998
      capLen = (surf/density/9.81)**.5
      with open(fname) as f:
        df = np.loadtxt(f, skiprows=1)
      for i in range(len(df[:,0])):
        if df[i,0]<50: continue
        axV.errorbar(df[i,0]/180, df[i,3]/capLen**3, xerr=[ [ df[i,1]/180-df[i,0]/180 ] , [ df[i,0]/180-df[i,2]/180 ] ], fmt='d', c='b', mfc='None', clip_on=False, zorder=3)
      ax.set_yticklabels([])
      ax.set_ylabel('')
      fig.subplots_adjust(left=0.03, right=0.86, bottom=0.2, top=0.98)
      radInd=2
      rads = np.pi*np.arange(0.8, -0.1, -0.2)
      print('rads',rads/np.pi)
      outName = folName+f'spr.pdf'
      #spc = (30, 20, 25, 10, 6, 4, 2, .7, 0)
      spc=(1,3,6,11,15)
      axProf.set_xlabel('$x/\\lambda$')
    if 'rad' in cont:
      axV.plot((3.832,*x),(0,*y),c='r',clip_on=False)
      axM.plot( x, z, c='r', clip_on=False, zorder=3)
      axM.plot( x, zM, '--', c='r', clip_on=False, zorder=3)
      fig2.subplots_adjust(left=0.1, right=0.88, bottom=0.2, top=0.98)
      figProf.subplots_adjust(left=0.05, right=0.97, bottom=0.02, top=0.98)
      xx=np.linspace(0,4)
      axV.plot( xx, 2*np.pi*xx, linestyle='dotted', c='r', lw=2)
      axM.set_xlabel('$r_0/\\lambda$')
      ax2.set_xlabel('$r_0/\\lambda$')
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
      axM.axvspan(3.219, 4, color='lightgrey')
      #axV.text(1.12,.5,'$\\frac{\\phi_0}{\\pi}$',c='r',transform=axV.transAxes,size=22,ha='center')
      axM.set_ylabel('$\\frac{\\phi_0}{\\pi}$',size=22,rotation=0,labelpad=10)
      axProf.text(5e-3,.99,'$\\mathrm{(a)}$',transform=axProf.transAxes,va='top',ha='left')
      ax.text(5e-3,.99,'$\\mathrm{(a)}$',transform=ax.transAxes,va='top',ha='left')
      ax2.set_xlim([0,4])
      #idx = np.argmax(x<.5)
      #mdx = np.argmax(x>5)+1
      #ax2.plot( (*x[mdx:idx],0), (*z[mdx:idx],.5), c='k',clip_on=False)#,'.',ms=5
      axP.plot( x, z, c='r',clip_on=False, zorder=3)#,'.',ms=5
      fname = 'LesageVolVsContRadSq.txt'
      print('open',fname)
      with open(fname) as f:
        df = np.loadtxt(f)
      for i in range(len(df[:,0])):
        if df[i,2]>1:continue
        axV.plot(df[i,0]**.5, df[i,1]*df[i,0]**1.5, 's', mec='r', mfc='None', clip_on=False, zorder=3)
      fname = 'MoriVolByContCubeVsContSqByCapSq.txt'
      print('open',fname)
      with open(fname) as f:
        df = np.loadtxt(f)
      axV.plot(.5/df[:,0]**.5, df[:,1]/df[:,0]**1.5, 'd', mec='r', mfc='None', clip_on=False, zorder=3)
      fname = 'sasetty23stability.txt'
      print('open',fname)
      with open(fname) as f:
        df = np.loadtxt(f)
      axV.plot(df[:,2]/df[:,3]/2, df[:,1]/(df[:,3]*1e-3)**3, 'v', mec='r', mfc='None', clip_on=False)
      fname = 'gunde01measurement.txt'
      print('open',fname)
      with open(fname) as f:
        df = np.loadtxt(f, skiprows=2)
      capLen=(df[:,2]*1e-3/df[:,1]/9.81)**.5
      axV.plot(df[:,0]*1e-3/capLen, df[:,4]*1e-6*1e-3/capLen**3, '^', mec='r', mfc='None', clip_on=False)
      axV.set_xlim([0,4])
      fig.subplots_adjust( left=0.14, right=0.97, bottom=0.2, top=0.98)
      outName = folName+f'pin.pdf'
      #spc=(1,3,6,10,15,21,28,40,50,60)
      #spc=(1.5,5,10.5)
      spc=(1,3.5,8,14.5)
      axProf.set_xticklabels([])
    for rad in range(len(rads)):
      for fname in reversed(sorted(os.listdir(folName))):
        if 'bub' not in fname: continue
        if 'txt' not in fname: continue
        with open(folName+fname, encoding = 'utf-8') as f:
          df = np.loadtxt(f)
        if df.ndim<2: continue
        if df[0,5]<1: col = ( 1-df[0,5], 0, 0)
        else: col = ( 0, 0, df[0,5]-1)
        #if not rad: 
        #  axProf.plot((0,.2),(df[0,5],df[0,5]),c=col)
        for p in range(1, len(df[:,0])):
          if (df[p,radInd] - rads[rad]) * (df[p-1,radInd] - rads[rad]) >= 0: continue
          volInd = np.argmin( abs( volDet - df[p,6] ))
          if -df[p,1]>heigDet[volInd]:continue
          ax.plot(df[p,6], -df[p,1], '.', c=col, clip_on=False, zorder=3)
          x=np.concatenate(( -df[:p,0][::-1] , df[:p,0] ))
          x=x+spc[rad]
          y=np.concatenate(( df[:p,1][::-1] - df[p,1] , df[:p,1] - df[p,1] ))
          axProf.plot(x,y, c=col, clip_on=False)
          if radInd!=0 or rad!=0 or df[p,1]>=-1: continue
          axProf.plot(x,y, c=col, clip_on=False)
          h=int(0.5*(len(x)))
          #axProf.text(x[h], y[h]+.2, '$(a,h)$', va='bottom', ha='center', c='grey')
          axProf.plot(x[h],y[h],'o', c='grey')
          t=int(0.7*(len(x)))
          axProf.plot(x[h:t],y[h:t], c='grey')
          theta = np.arctan2(y[t+1]-y[t], x[t+1]-x[t])-np.pi/2
          print('theta',theta)
          tri = RegularPolygon((x[t], y[t]), 3, radius=0.1, orientation=theta, color='grey', zorder=3)
          axProf.add_patch(tri)
          axProf.text(x[t]+0.1, y[t], '$s$', va='center', ha='left', c='grey')
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
    print('savin ',outName,spc)
    figProf.savefig(outName, transparent=True)
    fname = folName+'heightVsVol_'+cont+'.pdf'
    #fname = folName+'radFootVsVol_'+cont+'.pdf'
    #fname = folName+'contAngVsVol_'+cont+'.pdf'
    #fname = folName+'radTopVsVol_'+cont+'.pdf'
    print('savin ',fname)
    fig.savefig(fname, transparent=True, format='pdf')
    fname = folName+'MaxVolVs_'+cont+'.pdf'
    print('savin ',fname)
    figV.savefig(fname, transparent=True, format='pdf', bbox_inches='tight')
    fname = folName+'ax2_'+cont+'.pdf'
    print('savin ',fname)
    fig2.savefig(fname, transparent=True, format='pdf')
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
