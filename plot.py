"""
Created 2025 Jan 22
@author: ianto-cannon
plot bubbles on electrodes
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def AdamBash(): 
  plt.rcdefaults()
  plt.rcParams.update({"text.usetex": True,'font.size' : 12,})
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

def bubbleCoords(): 
  plt.rcdefaults()
  plt.rcParams.update({"text.usetex": True,'font.size' : 12,})
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

def cone(): 
  import matplotlib.image as mpimg
  plt.rcdefaults()
  plt.rcParams.update({"text.usetex": True,'font.size' : 12,})
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

def height(): 
  plt.rcdefaults()
  plt.rcParams.update({"text.usetex": True,'font.size' : 12,})
  fig, ax = plt.subplots(1)#, figsize=[columnWid, .6*columnWid])
  fname = 'data/vol.txt'
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f)
  ax.plot(df[:,0],df[:,4]*1000, color=(0,0,0))
  ax.tick_params(which='both', direction='in', top=True, right=True)
  ax.set_xlabel('$n$', rotation=0)
  ax.set_ylabel('$h/R$', rotation=0)
  ax.set_xlim([0,600])
  ax.set_ylim([.9,2.1])
  fname = 'data/height.pdf'
  print('savin ',fname)
  fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
  return

def vol(): 
  plt.rcdefaults()
  plt.rcParams.update({"text.usetex": True,'font.size' : 12,})
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

def BoColour(Bo):
  r=0
  g=0
  b=0
  if Bo<0: r=(-Bo/4)**.5
  if Bo>0: b= (Bo/4)**.5
  return (r,g,b)

def profile(): 
  plt.rcdefaults()
  plt.rcParams.update({"text.usetex": True,'font.size' : 12,})
  fig, ax = plt.subplots(1)#, figsize=[columnWid, .6*columnWid])
  col=BoColour(-4)
  folName = 'BoMP4/'
  fname = folName + 'pos2000.txt'
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f)
  height = max(df[:,2])
  ax.plot(np.sqrt(df[:,0]**2 + df[:,1]**2)*1000, (df[:,2]-height)*1000, '.', mec=col, mfc='None', mew=.2, alpha=.5)
  fname = folName + 'adams-0.4.txt'
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f)
  height = 0#-df[-1,1]
  ax.plot(df[:,0]*1e3, (df[:,1]+height)*1e3, color=col, alpha=.5)
  col=BoColour(4)
  #folName = 'BoP4TallPressureAsVolTo400/'
  folName = 'data/'
  fname = folName + 'pos500.txt'
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f)
  height = max(df[:,2])
  ax.plot(np.sqrt(df[:,0]**2 + df[:,1]**2)*1e3, (df[:,2]-height)*1e3, '.', mec=col, mfc='None', mew=.2, alpha=.5)
  fname = folName + 'adams0.4.txt'
  with open(fname, encoding = 'utf-8') as f:
    print('loadin ',fname)
    df = np.loadtxt(f)
  height = 0#-df[-1,1]
  ax.plot(df[:,0]*1e3, (df[:,1]+height)*1e3, color=col, alpha=.5)
  ax.tick_params(which='both', direction='in', top=True, right=True)
  ax.set_xlabel('$r/R$', rotation=0, labelpad=-5)
  ax.set_ylabel('$z/R$', rotation=0)
  #ax.set_ylim([-3.1,0.1])
  ax.set_xlim([0,1.2])
  ax.set_aspect('equal', adjustable='box')
  ax.text(.3, -1.45, '$Bo=-0.4$',c=(1,0,0), fontsize=10, ha='center', va='center')
  ax.text(.4, -3, '$Bo=0.4$',c=(0,0,1), fontsize=10, ha='center', va='center')
  fname = 'profile.pdf'
  print('savin ',fname)
  fig.savefig(fname, bbox_inches='tight', transparent=True, format='pdf', dpi=600)
  return

#AdamBash()
#bubbleCoords()
#cone()
#height()
#vol()
profile()

