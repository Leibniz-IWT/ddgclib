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
  fig, ax = plt.subplots(1)
  fig.set_figwidth(12)
  if radInd==0: 
    outName = folName+f'pin.pdf'
    #spc=(.3,1.2,3,5,8.5,14)
    spc=(1,3,6,10,15,21,28)
    #ax.set_xticklabels([])
    ax.set_xlabel('$x/\\lambda$')
  if radInd==2: 
    outName = folName+f'spr.pdf'
    #spc=(.5,2,4,7,12,14)
    #spc=(.3,1.2,3,5,8.5,14,20,25,30)
    #spc=(.5,2,4.5,8,13,20)
    spc=(.7,2,4.5,8,13,20)
    ax.set_xlabel('$x/\\lambda$')
  #spc=(.3,1.2,3,5,8.5,14,20,25,30)
  for r in range(len(rads)):
    for fname in reversed(sorted(os.listdir(folName))):
      if 'txt' not in fname: continue
      if 'bub' not in fname: continue
      with open(folName+fname, encoding = 'utf-8') as f:
        df = np.loadtxt(f)
      if df.ndim<2: continue
      if df[0,5]>2: continue
      if df[0,5]<1: col = ( 1-df[0,5], 0, 0)
      else: col = ( 0, 0, df[0,5]-1)
      #else: col = ( 0, 0, 1-1/df[0,5])
      if not r: 
        ax.plot((0,.2),(df[0,5],df[0,5]),c=col)
      for p in range(1, len(df[:,0])):
        if (df[p,radInd] - rads[r]) * (df[p-1,radInd] - rads[r]) < 0: 
          x=np.concatenate(( -df[:p,0][::-1] , df[:p,0] ))
          x=x+spc[r]
          y=np.concatenate(( df[:p,1][::-1] - df[p,1] , df[:p,1] - df[p,1] ))
          #ax.plot(x,y, c=col, clip_on=False)
          #if '010' in fname and radInd==0 and r==len(rads)-1:
          if radInd==0 and r==0 and df[p,1]<-2:
            ax.plot(x,y, c=col, clip_on=False)
            print('radInd',radInd,r,fname)
            h=int(0.5*(len(x)))
            ax.text(x[h], y[h]+.2, '$(a,h)$', va='bottom', ha='center', c='grey')
            ax.plot(x[h],y[h],'o', c='grey')
            t=int(0.7*(len(x)))
            ax.plot(x[h:t],y[h:t], c='grey')
            #ax.annotate("", xy=[x[t], y[t]], xytext=[x[t-1], y[t-1]], arrowprops=dict(arrowstyle="-|>", color='grey')) 
            theta = np.arctan2(y[t+1]-y[t], x[t+1]-x[t])-np.pi/2
            print('theta',theta)
            tri = RegularPolygon((x[t], y[t]), 3, radius=0.17, orientation=theta, color='grey', zorder=3)
            ax.add_patch(tri)
            ax.text(x[t]+0.2, y[t], '$s$', va='center', ha='left', c='grey')
            t=int(0.63*(len(x)))
            xAn = x[t]
            yAn = y[t]
            ax.plot((xAn,xAn+.3), (yAn,yAn), color='grey')
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
            ax.plot(Xarr,Yarr,c='grey')
            ax.text(xAn+.1, yAn+.1, "$\\phi$", ha='left', va='bottom', c='grey') 
          else: continue
  ax.tick_params(which='both', direction='in', top=True, right=True)
  ax.set_ylabel('$\\frac{ z }{\\lambda}$',rotation=0,size=22,labelpad=10)
  ax.set_ylim([0,4])
  ax.set_xlim([0,24.5])
  ax.set_aspect('equal', adjustable='box')
  print('savin ',outName)
  fig.savefig(outName, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
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
  folName = 'data/'
  degrees = [0, 30, 60, 90, 120, 150, 180]
  rads=[]
  for cont in nam.split():
    fig, ax = plt.subplots(1)
    figV, axV = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    ax.set_ylabel('$\\frac{h}{\\lambda}$',rotation=0,size=22,labelpad=10)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.2, top=0.98)
    figV.subplots_adjust(left=0.14, right=0.97, bottom=0.2, top=0.98)
    if 'ang' in cont:
      #x=np.array([0,180])
      x=np.linspace(0,180)
      axV.plot(x, 4*np.pi*(.0104*x)**3/3, ls='dotted', c='k')
      #axV.plot(x, (4*np.pi/3)**(1/3)*.0104*x, ls='dotted', c='k')
      print('fritz', 4*np.pi*.0104**3/3)
      axV.set_xlabel('$\\phi_0$')
      ax2.set_xlabel('$\\phi$')
      axV.set_xticks(degrees)
      ax2.set_xticks(degrees)
      ax2.set_ylabel('$\\frac{x_0-a}{\\lambda}$',rotation=0,size=22,labelpad=10)
      ax2.set_xlim([0,180])
      ax2.set_ylim([0,3.219])
    if 'rad' in cont:
      x=np.linspace(0,10)
      axV.plot( x, 2*np.pi*x, linestyle='dashed', c='k')
      ax2.set_xlabel('$(x_0-a)/\\lambda$')
      ax2.set_ylabel('$\\phi$',rotation=0)
      ax2.set_yticks(degrees)
      ax2.set_ylim([90,180])
      #ax2.set_xlim([0,3.219])
      ax2.set_xlim([0,6])
      #ax2.set_xlim(left=0)
    if 'bub' in cont:
      axV.set_xlabel('$R_t$')
      axV.set_xscale('log')
      axV.set_yscale('log')
    x=[]
    y=[]
    z=[]
    for fname in reversed(sorted(os.listdir(folName))):
    #for fname in sorted(os.listdir(folName)):
      if 'txt' not in fname: continue
      if cont not in fname: continue
      #if 'loop' not in fname: continue
      with open(folName+fname, encoding = 'utf-8') as f:
        df = np.loadtxt(f)
      if df.ndim<2: continue
      indVol = np.argmax(df[:,6])
      angl = 180 - df[indVol,2]*180/np.pi
      #for indSide in range(len(df[:,0])):
      #  an = 180 - df[indSide,2]*180/np.pi
      #  if an>180.5 and df[indSide,3]<0: break
      #angls = 180 - df[:,2]*180/np.pi
      #indSide = np.argmax(df[:,2]<0)
      #indSide = np.argmax(angls>180)
      #if indSide<indVol: 
      #  print(fname,indSide,indVol)
      #  indVol=indSide
      if 'ang' in cont: 
        if angl>185: x.append(np.nan)
        else: x.append(angl)
        z.append(df[indVol,0])
        #col=( min(abs(df[0,2]/np.pi)**.5, 1), 0, 0)
        #ax2.plot(x, df[indVol,0], '.', c=col ,clip_on=False)
      if 'rad' in cont: 
        x.append(df[indVol,0])
        z.append( 180 - df[indVol,2]*180/np.pi )
        #col=( min(df[0,0]/np.pi, 1), 0, 0)
        #col=( min(abs(df[indVol,2]/np.pi)**.5, 1), 0, 0)
        #ax2.plot(x, angl, '.', c=col ,clip_on=False)
      if 'bub' in cont: 
        x.append(df[indVol,5])
        #col=( min(df[0,5]/np.pi, 1), 0, 0)
      #r = (df[:,6]*3/4/np.pi) ** (1/3)
      r = df[:,6]
      y.append(r[indVol])
      height = -df[:,1]
      #height = 180 - df[:,2]*180/np.pi
      #height = df[:,5]+df[:,1]
      #height = df[:,7]-df[:,8]
      #height = ( height[1:] - height[:-1] ) / (r[1:] - r[:-1])
      #height = abs(height)**.1*np.sign(height)
      #r = (r[1:] + r[:-1])/2
      #r=r**.1
      #if indVol>=len(r):indVol=-1
      #height = df[:,8]
      for i in range(1, len(r)):
        if abs(r[i]-r[i-1]) > 3: r[i-1]=np.nan
      #  if abs(r[i]-r[i-1]) > .2: r[i-1]=np.nan
      #if indVol<len(r)-1: axV.plot(x, r[indVol], '.', c=col, clip_on=False)#, markersize=.2)
      #else: axV.plot(x, r[indVol], 'o', c=col, mfc='None', ms=4, clip_on=False)
      if 'Lo' in fname: continue
      if 'Hi' in fname: continue
      zord=3
      if 'ang' in cont: 
        if angl>181: continue
        if round(angl)%30==0:
          col='k'
          #col=(0,0,angl/250)
          #ax.text(r[indVol]-.1, height[indVol], rf"${angl:.0f}$", va='center', ha='right', c=col)
          if angl>70:
            #ax.text(r[indVol]-.1, height[indVol], rf"${angl:.0f}$", va='center', ha='right')
            ax.text(r[indVol-1]-.1, height[indVol-1], rf"${angl:.0f}$", va='center', ha='right')
            zord=3
        else: 
          col='silver'
          zord=1
      if 'rad' in cont: 
        if round(df[0,0]*10)%5==0:
          #ax.text(r[indVol]-.1, height[indVol], rf"${df[0,0]:.1f}$", va='center', ha='right')
          col='k'
          #col=(0,0,df[0,0]/6)
          ax.text(r[indVol-1]-.1, height[indVol-1], rf"${df[0,0]:.1f}$", va='center', ha='right', c=col)
          #indLbl=np.argmax(r>1)
          #ax.text(r[indLbl], height[indLbl], rf"${df[0,0]:.1f}$", va='center', ha='right', c=col)
          #print(r[indLbl], height[indLbl], rf"${df[0,0]:.1f}$")
          zord=3
        else: 
          col='silver'
          zord=1
      if 'bub' in cont: 
        if '9.txt' in fname or df[0,5]>5:
          col='k'
          zord=3
          if df[0,5]>.5 and df[0,5]<=1: ax.text(r[indVol], height[indVol], rf"${df[0,5]:.1f}$", va='center', ha='left')
          if df[0,5]>1 and '99.txt' in fname: ax.text(r[indVol], height[indVol], rf"${df[0,5]:.0f}$", va='top', ha='left')
        else: 
          col='silver'
          zord=1
      #print('indVol',indVol,len(r),cont,x,r[indVol])
      #ax.plot(r[:indVol+1], height[:indVol+1], c=col, zorder=zord)#, alpha=.5)#
      ax.plot(r[:indVol+1], height[:indVol+1], '.', c=col, zorder=zord, ms=1)#, alpha=.5)#
      #ax.plot(r[:indVol], height[:indVol], c=col, zorder=zord)#, alpha=.5)#
      #ax.plot(r, height, c=col, zorder=zord)#, alpha=.5)#
    if 'ang' in cont:
      axV.set_xlim([0,180])
      #axV.set_xlim([10,180])
      #axV.set_ylim([1e-2,1e2])
      if False:
        fname = 'Ling25effectRadTopVsVol.txt'
        print('open',fname)
        with open(fname) as f:
          df = np.loadtxt(f)
        col=( (1-159/180)**.5, 0, 0)
        for i in range(0):#len(df[:,0])):
          ax.plot((.75*df[i,0]/np.pi)**(1/3)/27e-4, 27e-4/df[i,1], 's', mec=col, mfc='None', clip_on=False)
        fname = 'Ling25effect.txt'
        print('open',fname)
        with open(fname) as f:
          df = np.loadtxt(f)
        for i in range(len(df[:,0])):
          #axV.plot(df[i,2], (.75*df[i,4]*1e-6/np.pi)**(1/3)/df[i,5]/1e-3, 's', mec='k', mfc='None', clip_on=False)
          print(i,fname,df[i,4]*1e-6 / (df[i,5]*1e-3)**3)
          axV.plot(df[i,2], df[i,4]*1e-6 / (df[i,5]*1e-3)**3, 'v', mec='k', mfc='None', clip_on=False)
        fname = 'phan09surface.txt'
        print('open',fname)
        with open(fname) as f:
          df = np.loadtxt(f)
        capLen = df[0,1]/df[0,0]/.02*1e-3
        for i in range(1,len(df[:,0])):
          print(i,df[i,0])
          axV.plot(df[i,0], 4*np.pi/3 * (df[i,1]*1e-3)**3 / capLen**3, '^', mec='k', mfc='None', clip_on=False)
      if True:
        fname = 'demirkir24life.txt'
        print('open',fname)
        with open(fname) as f:
          df = np.loadtxt(f, skiprows=1)
        for i in range(len(df[:,0])):
          rad = df[i,1]*1e-6
          density = df[i,2] -	0.08988*1e-6
          surf = df[i,3]*1e-3
          capLen = (surf/density/9.81)**.5
          #axV.plot(df[i,4], 4*np.pi/3 * rad**3 / capLen**3, '+', mec='k', mfc='None', clip_on=False)
          #axV.plot(df[i,0], 4*np.pi/3 * rad**3 / capLen**3, 'd', mec='k', mfc='None', clip_on=False)
          mid = (df[i,0]+df[i,4])/2
          if df[i,0]-mid > 20: continue
          print(i, [df[i,0]-mid])
          axV.errorbar( mid, 4*np.pi/3 * rad**3 / capLen**3, xerr=[ [df[i,0]-mid], [mid-df[i,4]] ], fmt='^', c='b', mfc='None',clip_on=False, zorder=3)
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
          #axV.plot(df[i,2], 4*np.pi/3 * rad**3 / capLen**3, '<', mec='k', mfc='None', clip_on=False)
          #axV.plot(df[i,4], 4*np.pi/3 * rad**3 / capLen**3, 'o', mec='k', mfc='None', clip_on=False)
          #axV.plot(df[i,3], 4*np.pi/3 * rad**3 / capLen**3, '>', mec='k', mfc='None', clip_on=False)
          #mx=max(df[i,2:])
          #mn=min(df[i,2:])
          #mid = (mx+mn)/2
          mn=df[i,2]
          mid=df[i,4]
          mx=df[i,3]
          if mid<mn: continue
          if mid>mx: continue
          #axV.errorbar( mid, 4*np.pi/3 * rad**3 / capLen**3, xerr=[[mx-mid], [mid-mn]], fmt='v', c='silver', mfc='None')
          axV.plot(mid, vol, 'v', c='b', mfc='None', zorder=3)
          axV.plot([mn,mx], [vol,vol], c='b', zorder=3)
      if True:
        fname = 'huang25effects.txt'
        print('open',fname)
        surf=72.25e-3
        density=998
        capLen = (surf/density/9.81)**.5
        with open(fname) as f:
          df = np.loadtxt(f, skiprows=1)
        for i in range(len(df[:,0])):
          if df[i,0]<50: continue
          #axV.plot(df[i,0], df[i,1]/capLen**3, 'd', mec='k', mfc='None', clip_on=False)
          axV.errorbar(df[i,0], df[i,3]/capLen**3, xerr=[ [ df[i,1]-df[i,0] ] , [ df[i,0]-df[i,2] ] ], fmt='d', c='b', mfc='None', clip_on=False, zorder=3)
      #ax.set_yticklabels([])
      #axV.set_yticklabels([])
      #ax.set_ylabel('')
      #fig.subplots_adjust(left=0.03, right=0.86, bottom=0.2, top=0.98)
      #figV.subplots_adjust(left=0.03, right=0.86, bottom=0.2, top=0.98)
      print('angWid',.86-.03)
    if 'rad' in cont:
      axV.plot([3.219,3.219],[1e-1,1e2],c='k',ls='dotted')#,clip_on=False)
      ax2.plot([3.219,3.219],[0,180],c='k',ls='dotted')#,clip_on=False)
      fname = 'LesageVolVsContRadSq.txt'
      print('open',fname)
      with open(fname) as f:
        df = np.loadtxt(f)
      for i in range(len(df[:,0])):
        if df[i,2]>1:continue
        #axV.plot(df[i,0]**.5, (.75*df[i,1]/np.pi)**(1/3)*df[i,0]**.5, 's', mec=(0,0,df[i,2]/3), mfc='None', clip_on=False)
        #axV.plot(df[i,0]**.5, df[i,1]*df[i,0]**1.5 -2*np.pi*df[i,0]**.5, 's', mec=(0,0,df[i,2]/3), mfc='None', clip_on=False)
        axV.plot(df[i,0]**.5, df[i,1]*df[i,0]**1.5, 's', mec='b', mfc='None', clip_on=False, zorder=3)
      fname = 'MoriVolByContCubeVsContSqByCapSq.txt'
      print('open',fname)
      with open(fname) as f:
        df = np.loadtxt(f)
      #axV.plot(.5/df[:,0]**.5, (.75*df[:,1]/np.pi)**(1/3)/df[:,0]**.5, 'd', mec=(0,0,0), mfc='None', clip_on=False)
      #axV.plot(.5/df[:,0]**.5, df[:,1]/df[:,0]**1.5 -2*np.pi*.5/df[:,0]**.5, 'd', mec=(0,0,0), mfc='None', clip_on=False)
      axV.plot(.5/df[:,0]**.5, df[:,1]/df[:,0]**1.5, 'd', mec=(0,0,1), mfc='None', clip_on=False, zorder=3)
      if False:
        fname = 'zhang95experimental.txt'
        density=996
        surf=73e-3
        capLen=(surf/density/9.81)**.5
        print('open',fname)
        with open(fname) as f:
          df = np.loadtxt(f)
        axV.plot(df[:,0]/capLen, df[:,1]*df[:,0]**3/capLen**3, '+', mec=(0,0,1), mfc='None', clip_on=False)
        fname = 'arogeti25evaluating.txt'
        density=997
        surf=72.0e-3
        vol=15e-6*1e-3
        print('open',fname)
        with open(fname) as f:
          df = np.loadtxt(f)
        for i in range(len(df[:,0])):
          capLen=(surf/density/9.81/df[i,1])**.5
          #axV.plot(df[i,0]/capLen/2, vol/capLen**3 -2*np.pi*df[i,0]/capLen/2, 'o', mec=(0,0,1), mfc='None', clip_on=False)
          axV.plot(df[i,0]/capLen/2, vol/capLen**3, 'o', mec=(0,0,1), mfc='None', clip_on=False)
      fname = 'sasetty23stability.txt'
      print('open',fname)
      with open(fname) as f:
        df = np.loadtxt(f)
      #axV.plot(df[:,2]/df[:,3]/2, df[:,1]/(df[:,3]*1e-3)**3 -2*np.pi*df[:,2]/df[:,3]/2, 'v', mec=(0,0,1), mfc='None', clip_on=False)
      axV.plot(df[:,2]/df[:,3]/2, df[:,1]/(df[:,3]*1e-3)**3, 'v', mec=(0,0,1), mfc='None', clip_on=False)
      liquids='wasg'
      #for i in range(len(liquids)):
      #  axV.text(df[i,2]/df[i,3]/2, df[i,1]/(df[i,3]*1e-3)**3, liquids[i], c='b')
      if False:
        fname = 'kumar70formation.txt'
        print('open',fname)
        with open(fname) as f:
          df = np.loadtxt(f)
        density = 786*df[:,3] + 998*(1-df[:,3])
        capLen = (df[:,2]*1e-3/density/9.81)**.5
        vol = (1e-6)*df[:,1]/capLen[:]**3
        print('capLen',capLen,'vol', df[:,1], vol)
        axV.plot(.4050e-2/capLen, vol, '<', mec=(0,0,1), mfc='None', clip_on=False)
      fname = 'gunde01measurement.txt'
      print('open',fname)
      with open(fname) as f:
        df = np.loadtxt(f, skiprows=2)
      capLen=(df[:,2]*1e-3/df[:,1]/9.81)**.5
      axV.plot(df[:,0]*1e-3/capLen, df[:,4]*1e-6*1e-3/capLen**3, '^', mec=(0,0,1), mfc='None', clip_on=False)
      axV.set_xlabel('$(x_0-a)/\\lambda$',rotation=0)
      axV.set_ylabel('$\\frac{V_m}{\\lambda^3}$',rotation=0,size=22,labelpad=10)
      axV.set_xlim([0,5])
      #axV.set_xlim([5e-2,6])
      #axV.set_ylim([1e-1,20])
      #axV.set_ylim([1e-1,1e2])
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    idx = np.argsort(x)
    #axV.plot(x[idx],y[idx]-2*np.pi*x[idx],c='k',clip_on=False)
    axV.plot(x[idx],y[idx],c='k')#,clip_on=False)
    if 'bub' not in cont: ax2.plot(x[idx],z[idx],c='k',clip_on=False)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_xlabel('$V/\\lambda^3$')
    axV.set_ylabel('$\\frac{V_m}{\\lambda^3}$',rotation=0,size=22,labelpad=10)
    #ax.set_ylim([175,185])
    ax.set_ylim([0,3])
    ax.set_xlim([0,20])
    axV.set_ylim([0,20])
    #axV.set_ylim([1e-2,1e2])
    #axV.set_ylim([0,1.8])
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #axV.set_xscale('log')
    #axV.set_yscale('log')
    axV.tick_params(which='both', direction='in', top=True, right=True)
    fname = folName+'heightVsVol_'+cont+'.pdf'
    #fname = folName+'enVsVol_'+cont+'.pdf'
    #fname = folName+'dEnergydVVsVol_'+cont+'.pdf'
    #fname = folName+'centroidVsVol_'+cont+'.pdf'
    #fname = folName+'radTopVsVol_'+cont+'.pdf'
    #fname = folName+'pressureVsVol_'+cont+'.pdf'
    #fname = folName+'anglVsVol_'+cont+'.pdf'
    fig.set_figwidth(5)
    figV.set_figwidth(5)
    fig2.set_figwidth(5)
    fig.set_figheight(3)
    figV.set_figheight(3)
    fig2.set_figheight(3)
    ax2.tick_params(which='both', direction='in', top=True, right=True)
    print('savin ',fname)
    fig.savefig(fname, transparent=True, format='pdf')
    fname = folName+'MaxVolVs_'+cont+'.pdf'
    print('savin ',fname)
    figV.savefig(fname, transparent=True, format='pdf')
    fname = folName+'ax2_'+cont+'.pdf'
    print('savin ',fname)
    if 'bub' not in cont: fig2.savefig(fname, bbox_inches='tight', transparent=True, format='pdf')
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
    for i in range(len(df[:,6])):
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
