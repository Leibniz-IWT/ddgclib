#!/usr/bin/env python
# coding: utf-8

# Imports and physical parameters
import copy
import sys
import numpy as np
import polyscope as ps

# Local library
from ddgclib import *
from ddgclib._curvatures import HC_curvatures_sessile
from ddgclib._complex import *
from ddgclib._sphere import *
from ddgclib._sessile import *
from ddgclib._capillary_rise_flow import * #plot_surface#, curvature
from ddgclib._eos import *
from ddgclib._plotting import plot_polyscope

def sector_volume(HC):
#Compute volume of the complex by splitting it into sectors centred on origin.
  (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
    HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
    Theta_i_cache) = HC_curvatures_sessile(HC, bV, RadFoot, theta_p, printout=0)
  total_bubble_volume=0.0
  totalArea=0.0
  for v in HC.V:
    dualArea = sum(C_ij_cache[v.x])
    totalArea += dualArea
    H = HNdA_i_cache[v.x]
    N_approx = - normalized(H)[0]
    total_bubble_volume += dualArea * sum( v.x_a[:] * N_approx[:] ) / 3.0
  return total_bubble_volume

def prism_volume(HC):
#Compute volume of the complex by splitting it into prisms over the surface.
  (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
    HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
    Theta_i_cache) = HC_curvatures_sessile(HC, bV, RadFoot, theta_p, printout=0)
  total_bubble_volume=0.0
  totalArea=0.0
  for v in HC.V:
    dualArea = sum(C_ij_cache[v.x])
    totalArea += dualArea
    H = HNdA_i_cache[v.x]
    N_approx = outward_normal(v,H)
    total_bubble_volume += dualArea * N_approx[2] * v.x_a[2] #+ np.dot(H, v.x_a) 
  return total_bubble_volume

def save_neighbours(fname):
#Make a text file listing the vertices (id from 0) in first column,
#with the ids of the connected vertices in next columns.
  vNum=0
  posDic={}
  for v in HC.V:
    posDic[v.x]=vNum
    vNum+=1
  with open(fname, "w") as nei_txt:
    print('saving',fname)
    vNum=0
    for v in HC.V:
      print(vNum, ' ', file=nei_txt, end='')
      for nei in v.nn:
        try:
          print(posDic[nei.x], ' ', file=nei_txt, end='')
        except KeyError:
          print('KeyError caught, not in dictionary',nei.x)
      print('', file=nei_txt)
      vNum+=1
  return

def save_vert_positions(t):
#Save the xyz position of all vertices in a text file.
  fname='data/pos'+str(t)+'.txt'
  with open(fname, "w") as pos_txt:
    print('saving',fname)
    for v in HC.V:
      print(*v.x,file=pos_txt)
  save_neighbours('data/nei'+str(t)+'.txt')

def load_complex(t):
#Build a complex from a position file and a neighbour file.
  fname='data/pos'+str(t)+'.txt'
  with open(fname) as f:
    print('loading',fname)
    pos = [[float(x) for x in line.split()] for line in f]
  fname='data/nei'+str(t)+'.txt'
  with open(fname) as f:
    print('loading',fname)
    nn = [[int(x) for x in line.split()[1:]] for line in f]
  HC = construct_HC(pos, nn)   
  bV = set()
  for v in HC.V:
    if abs(v.x[2]) < 1e-9: bV.add(v)
  return HC, bV

def refine_edges(HC, dist):
#Put vertices on edges longer than dist.
#Like fig 2a of Unverdi and Tryggvason J comp. phys. 1992.
#Neighbouring edges are not refined.
  to_split_1D=[]
  to_split=[]
  for i, v1 in enumerate(HC.V):
    if v1 in to_split_1D: continue
    for v2 in v1.nn:
      if v2 in to_split_1D: continue
      if np.linalg.norm(v1.x_a-v2.x_a) > dist: 
        common_neigh = list(v1.nn.intersection(v2.nn))
        #Only refine if the vertices are in a planar region
        if len(common_neigh) != 2: continue
        if common_neigh[0] in to_split_1D: continue
        if common_neigh[1] in to_split_1D: continue
        to_split.append((v1,v2,*common_neigh))
        to_split_1D.extend((v1,v2,*common_neigh))
  if len(to_split)==0:return 0
  for (v1, v2, v3, v4) in to_split:
    v1.disconnect(v2)
    # Compute vertex on centre of edge:
    v_pos = 0.5*v1.x_a + 0.5*v2.x_a
    v_new = HC.V[tuple(v_pos)]
    # Connect to original 2 vertices to the new centre vertex
    v_new.connect(v1)
    v_new.connect(v2)
    v_new.connect(v3)
    v_new.connect(v4)
  return len(to_split)

def reconnect_long_diagonals(HC):
#Find bisected quadrilaterals and
#Like fig 2c of Unverdi and Tryggvason J comp. phys. 1992.
#Neighbouring edges are not refined.
  to_reconnect=[]
  for i, v1 in enumerate(HC.V):
    for v2 in v1.nn:
      common_neigh = list(v1.nn.intersection(v2.nn))
      #Only reconnect if the vertices are in a planar region
      if len(common_neigh) != 2: continue
      v3 = common_neigh[0]
      v4 = common_neigh[1]
      common_neigh = list(v3.nn.intersection(v4.nn))
      if common_neigh != [v1,v2] and common_neigh != [v2,v1]: continue
      #Make sure quadrilaterals do not overlap
      if any( [len(set(quad).intersection((v1,v2,v3,v4))) > 2 for quad in to_reconnect] ): continue
      if np.linalg.norm(v1.x_a-v2.x_a) > 1.5*np.linalg.norm(v3.x_a-v4.x_a): 
        to_reconnect.append((v1,v2,v3,v4))
  if len(to_reconnect)==0: return 0
  for (v1, v2, v3, v4) in to_reconnect:
    v1.disconnect(v2)
    v3.connect(v4)
  return len(to_reconnect)

def outward_normal(vert, meanCurv):
#Compute the normal, and ensure it points away from the z axis.
#If the surface is flat we use the direction from the origin.
  if numpy.linalg.norm(meanCurv) > 1e-10:
    N_approx = - normalized(meanCurv)[0]
  else:
    N_approx = normalized(vert.x_a)[0]
  if sum(N_approx[:3]*vert.x_a[:3])<0: N_approx = -N_approx
  return N_approx

def reduce_energy(HC, bV, forcePrev, posPrev, params, tau, print_out=False, pinned_line=False):
#Move vertices to reduce the surface, pressure, and gravitaional energy
  (gamma, rho, g, RadFoot, theta_p) = params
  maxMove=.005*RadTop#.05*RadTop
  # Compute interior curvatures
  (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
    HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
    Theta_i_cache) = HC_curvatures_sessile(HC, bV, RadFoot, theta_p, printout=0)
  total_bubble_volume = prism_volume(HC)
  print("total_bubble_volume",total_bubble_volume)
  if total_bubble_volume != total_bubble_volume: return -1
  gasPressure = P_0*Volume/total_bubble_volume
  forceDict = {}
  posDict = {}
  nConcave=0
  maxForce = 0.0
  for v in HC.V:
    netSurfForce = np.array([0.0,0.0,0.0])
    netBuoyancy = np.array([0.0,0.0,0.0])
    netCompressForce = np.array([0.0,0.0,0.0])
    force = 0.0
    # Compute boundary movements
    # Note: boundaries are fixed for now, this is legacy:
    if v not in bV:
      H = HNdA_i_cache[v.x]
      df = gamma * H
      netSurfForce += df
      N_approx = outward_normal(v,H)
      dualArea = sum(C_ij_cache[v.x])
      dc = gasPressure * N_approx  * dualArea
      netCompressForce += dc
      liquidPressure = P_0 #- rho * g * v.x_a[2]
      if liquidPressure<0: print('bubble is too tall, liquidPressure=',liquidPressure)
      dg = - liquidPressure * N_approx  * dualArea
      netBuoyancy += dg
      force = df + dc + dg 
      maxForce=max( maxForce, np.linalg.norm(force) ) 
      forceDict[v.x] = force
  for v in HC.V:
    if v.x in forceDict:
      normFor = maxMove/maxForce*forceDict[v.x]
      f_k = v.x_a + maxMove/maxForce*forceDict[v.x]
      HC.V.move(v, tuple(f_k))
      if np.linalg.norm(normFor)>2*maxMove: 
        print('warning, normFor', np.linalg.norm(normFor))
        print('maxForce', maxForce)
        print('maxMove', maxMove)
      if np.linalg.norm(v.x_a)>3e-3: print('warning, vertex at ', v.x_a)
  if maxForce>2*cdist: 
    print('Warning: maxForce is greater than cdist')
    print('cdist=',cdist)
    print('unmerged vertices may crossover and the interface may overlap')
  print(t,total_bubble_volume,gasPressure,maxForce,*netSurfForce,*netCompressForce,*netBuoyancy,file=vol_txt)
  vol_txt.flush()
  return HC, bV, forceDict, posDict

def spherical_cap_init(RadFoot, theta_p, NFoot=4, refinement=0, cdist=-1):
#Create a complex in the shape of a sphere with contact angle theta_p
  RadSphere = RadFoot / np.sin(theta_p)
  Cone = []
  nn = []
  Cone.append(np.array([0.0, 0.0, 0.0])) 
  nn.append([])
  ind = 0
  #Make cone
  for phi in np.linspace(0.0, 2 * np.pi, NFoot + 1):
    ind += 1
    Cone.append(np.array([np.sin(phi), np.cos(phi), 1])) #IC contact line circle
    # Define connections:
    nn.append([])
    if ind > 0:
      nn[0].append(ind)
      nn[ind].append(0)
      nn[ind].append(ind - 1)
      nn[ind].append((ind + 1) % NFoot)
  # clean Cone
  for f in Cone:
    for i, fx in enumerate(f):
      if abs(fx) < 1e-15:
        f[i] = 0.0
  Cone = np.array(Cone)
  nn[1][1] = ind
  # Construct complex from the cone geometry:
  HC = construct_HC(Cone, nn)   
  v0 = HC.V[tuple(Cone[0])]
  # Compute boundary vertices
  V = set()
  for v in HC.V:
    V.add(v)
  bV = V - set([v0])
  for i in range(refinement):
    V = set()
    for v in HC.V:
      V.add(v)
    HC.refine_all_star(exclude=bV)
  # Move to spherical cap
  #thetaResolve=.95*theta_p
  #thetaFoot=.01*theta_p
  #thetaTop=100000#.2*theta_p
  #C = 1/thetaTop
  #B = 3*theta_p - 1/thetaFoot + 2/thetaTop
  #A = 1/thetaFoot - 3/thetaTop - 2*theta_p 
  #theta = lambda z : A*z**3 + B*z**2 + C*z 
  theta = lambda z : theta_p*min(2*z, .01*(z-1) + 1) 
  for v in HC.V:
    #theta = theta_p * v.x_a[2]**0.1
    thet = theta_p * v.x_a[2]
    #thet = theta(v.x_a[2])
    #if theta <= thetaResolve: theta = 0.5 * theta_p * v.x_a[2] / 
    phi = np.arctan2(v.x_a[1],v.x_a[0])
    x = RadSphere * np.cos(phi) * np.sin(thet)
    y = RadSphere * np.sin(phi) * np.sin(thet)
    z = RadSphere * np.cos(thet) - RadSphere * np.cos(theta_p)
    if abs(z) < RadSphere*1e-6: z = 0.0
    HC.V.move(v, tuple((x,y,z)))
  if 0:#cdist>0: 
    for i in range(6):
      lenH=0
      for v in HC.V: lenH+=1
      print('merge',i,lenH)
      HC.V.merge_all(cdist=cdist)
  # Rebuild set after moved vertices (appears to be needed)
  bV = set()
  for v in HC.V:
    if abs(v.x[2]) < RadSphere*1e-4: bV.add(v)
  return HC, bV

def cone_init(RadFoot, Volume, NFoot=4, refinement=0, cdist=-1):
#Make a cone shaped complex
  height = 3*Volume / np.pi / RadFoot**2
  print('height',height)
  Cone = []
  nn = []
  Cone.append(np.array([0.0, 0.0, height])) #IC middle vertex
  nn.append([])
  ind = 0
  #Make cone
  for phi in np.linspace(0.0, 2 * np.pi, NFoot + 1):
    ind += 1
    Cone.append(np.array([np.sin(phi), np.cos(phi), 0])) #IC contact line circle
    # Define connections:
    nn.append([])
    if ind > 0:
      nn[0].append(ind)
      nn[ind].append(0)
      nn[ind].append(ind - 1)
      nn[ind].append((ind + 1) % NFoot)
  # clean Cone
  for f in Cone:
    for i, fx in enumerate(f):
      if abs(fx) < 1e-15:
        f[i] = 0.0
  Cone = np.array(Cone)
  nn[1][1] = ind
  # Construct complex from the cone geometry:
  HC = construct_HC(Cone, nn)   
  v0 = HC.V[tuple(Cone[0])]
  # Compute boundary vertices
  V = set()
  for v in HC.V:
    V.add(v)
  bV = V - set([v0])
  #plot_polyscope(HC)
  for i in range(refinement):
    V = set()
    for v in HC.V:
      V.add(v)
    HC.refine_all_star(exclude=bV)
  #move refined vertices to circular cone
  for v in HC.V:
    z = v.x_a[2]
    Rad = RadFoot * (height - z) / height
    phi = np.arctan2(v.x_a[1],v.x_a[0])
    x = Rad* np.cos(phi)
    y = Rad* np.sin(phi)
    HC.V.move(v, tuple((x,y,z)))
  #plot_polyscope(HC)
  # Rebuild set after moved vertices (appears to be needed)
  bV = set()
  for v in HC.V:
    if abs(v.x[2]) < height*1e-4: bV.add(v)
  return HC, bV

# ### Parameters
## Numerical
tau = 1 #0.001  # 0.1 works
cdist=1e-3#7

## Physical
Bo=.5#.0955#100*rho*g*RadSphere**2/gamma
P_0 = 101.325e3  # Pa, Ambient pressure
gamma = 72.8e-3  # # N/m
rho = 998.2071  # kg/m3, density, STP
#rho_1 = 1.0 # kg/m3, density of air
g = 9.81  # m/s2
pinned_line = True  # pin the three phase contact if set to True
theta_p = 0 #.9*np.pi #179 #2*(63 / 75) * 20 + 30  # 46.8  40 #IC
RadTop = 1e-3#(Bo*gamma/rho/g)**.5 #250e-6 #8.426e-4 #300e-6
delPressu = 2*gamma/RadTop #Bo*gamma/g/RadTop**2
#RadFoot = .1*RadSphere# 2e-6 #1e-4#((44 / 58) * 0.25 + 1.0) * 1e-3  # height mm --> m  Radius 10e-6
#theta_p = np.pi - np.asin(RadFoot / RadSphere)
#RadFoot = (180 - theta_p)*RadSphere
#theta_p = theta_p * np.pi / 180.0
#height =  ((3 / 58) * 0.25 + 0.5) * 1e-3  # height mm --> m 10e-6
#height = RadFoot / np.sin(theta_p) * ( 1 - np.cos(theta_p) )
#Volume = np.pi * (3 * RadFoot ** 2 * height + height ** 3) / 6.0  # Volume in m3 (Segment of a sphere, see note above)

dt=[.01,.001,.0001]
for d in dt:
  psi=0
  r=0
  z=0
  Volume=0
  fname='data/adams'+str(d)[2:]+'.txt'
  with open(fname, "w") as adams_txt:
    print('saving',fname)
    for i in range(int(4/d)):
      r += d * np.cos(psi)
      dz = d * np.sin(psi)
      z += dz
      Volume += np.pi*r**2*dz
      if i*d*100%1 == 0: print(r*RadTop, -z*RadTop, file=adams_txt)
      psi += d * (2 - Bo*z - np.sin(psi)/r)
      if psi > np.pi/2: break

# define params tuple used in solver:
#Volume = Volume*RadTop**3
#RadFoot = r*RadTop
RadFoot = RadTop
Volume = 2*np.pi/3*RadTop**3
print(f'RadFoot = {RadFoot}')
params =  (gamma, rho, g, RadFoot, theta_p)
# Colour scheme for surfaces
db = np.array([129, 160, 189]) / 255  # Dark blue
lb = np.array([176, 206, 234]) / 255  # Light blue

tInit=670
t=tInit
if tInit==0:
  HC,bV = cone_init(RadFoot, Volume, NFoot=6, refinement=3, cdist=cdist)
else:
  HC, bV = load_complex(t)
refine_edges(HC, .1*RadTop)
reconnect_long_diagonals(HC)
plot_polyscope(HC)
initial_volume = Volume
fname='data/vol.txt'
with open(fname, "a") as vol_txt:
  print('saving',fname)
  forcePrev = {}
  posPrev = {}
  #Surface energy minimisation
  while t<=tInit+3000:
    t+=1
    print("t",t)
    HC, bV, forcePrev, posPrev = reduce_energy(HC, bV, forcePrev, posPrev, params, tau=tau, pinned_line=pinned_line)
    print('merge', HC.V.merge_nn(cdist=.05*RadTop, exclude=bV) )
    print('refine', refine_edges(HC, .1*RadTop) )
    print('reconnect', reconnect_long_diagonals(HC))
    for v in HC.V:
      if len(v.nn)<2:  
          print('v.nn',v.nn)
          print('remove',v.x_a)
          HC.V.remove(v)
    if t%10==0:
      save_vert_positions(t)
plot_polyscope(HC)
plt.show()
