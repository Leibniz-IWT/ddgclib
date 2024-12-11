#!/usr/bin/env python
# coding: utf-8

# Imports and physical parameters
import copy
import sys
import numpy as np
import polyscope as ps
#import scipy
#from matplotlib import pyplot as plt
#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.patches import FancyArrowPatch
#from mpl_toolkits.mplot3d import proj3d

# Local library
from ddgclib import *
from ddgclib._curvatures import HC_curvatures_sessile
from ddgclib._complex import *
from ddgclib._sphere import *
from ddgclib._sessile import *
from ddgclib._capillary_rise_flow import * #plot_surface#, curvature
from ddgclib._eos import *
from ddgclib._plotting import plot_polyscope

if 0:
 class vert:
  mHNdA_ij
  mHN_i = []
  mC_ij = []
  mK_H_i = []
  mHNdA_i_Cij = []
  mTheta_i = []
  mlastMove = []

def sector_volume(HC):
  #compute the volume of the complex by splitting it into sectors centred on origin
  (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
    HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
    Theta_i_cache) = HC_curvatures_sessile(HC, bV, RadFoot, theta_p, printout=0)
  total_bubble_volume=0.0
  totalArea=0.0
  for v in HC.V:
    #dualArea = np.linalg.norm(C_ij_cache[v.x])
    #print('v.x',v.x)
    #print('C_ij_cache[v.x]',C_ij_cache[v.x])
    #print('C_ij[v.x]',C_ij[v.x])
    dualArea = sum(C_ij_cache[v.x])
    totalArea += dualArea
    H = HNdA_i_cache[v.x]
    #ToDo: Convex should be negative, concave parts should be positive.
    N_approx = - normalized(H)[0]
    total_bubble_volume += dualArea * sum( v.x_a[:] * N_approx[:] ) / 3.0
  print('totalArea',totalArea)
  return total_bubble_volume

def prism_volume(HC):
  #compute the volume of the complex by splitting it into prisms over the surface
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
    total_bubble_volume += dualArea * N_approx[2] * v.x_a[2] #+ np.dot(H, v.x_a) 
  print('totalArea',totalArea)
  return total_bubble_volume

def saveNeigh(fname):
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

def saveComplex(t):
  fname='data/pos'+str(t)+'.txt'
  with open(fname, "w") as pos_txt:
    print('saving',fname)
    for v in HC.V:
      print(*v.x,file=pos_txt)
  saveNeigh('data/nei'+str(t)+'.txt')

def loadComplex(t):
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

def refineCentroids():
  new_verts=[]
  neigbours=[]
  for i, v1 in enumerate(HC.V):
    #if i!=10: continue
    for v2 in v1.nn:
      for v3 in v2.nn:
        if v3==v1: continue
        for v4 in v3.nn:
          if v4!=v1: continue
          area = 0.5 * np.linalg.norm( np.cross(v1.x_a,v2.x_a) 
                                     + np.cross(v2.x_a,v3.x_a) 
                                     + np.cross(v3.x_a,v1.x_a) )
          if area < cdist**2: continue
          posNew = (v1.x_a + v2.x_a + v3.x_a) / 3
          new_verts.append(HC.V[tuple(posNew)])
          # Connect to original 2 vertices to the new centre vertex
          neigbours.append([v1,v2,v3])
  for i, new_vert in enumerate(new_verts):
    for neigh in neigbours[i]:
      new_vert.connect(neigh)
  return

def refine_ed(HC, dist):
  to_split_1D=[]
  to_split=[]
  for i, v1 in enumerate(HC.V):
    if v1 in to_split_1D: continue
    if i!=10: continue
    for v2 in v1.nn:
      if v2 in to_split_1D: continue
      if sum(v1.x_a*v2.x_a) > dist**2: 
        common_neigh = list(v1.nn.intersection(v2.nn))
        if len(common_neigh) != 2: continue
        if common_neigh[0] in to_split_1D: continue
        if common_neigh[1] in to_split_1D: continue
        to_split.append((v1,v2,*common_neigh))
        to_split_1D.extend((v1,v2,*common_neigh))
        return to_split
        
def refine_edges(HC, dist):
  #to_split=refine_ed(HC, dist)
  to_split_1D=[]
  to_split=[]
  for i, v1 in enumerate(HC.V):
    if v1 in to_split_1D: continue
    #if i!=10: continue
    for v2 in v1.nn:
      if v2 in to_split_1D: continue
      if np.linalg.norm(v1.x_a-v2.x_a) > dist: 
        common_neigh = list(v1.nn.intersection(v2.nn))
        if len(common_neigh) != 2: continue
        if common_neigh[0] in to_split_1D: continue
        if common_neigh[1] in to_split_1D: continue
        to_split.append((v1,v2,*common_neigh))
        to_split_1D.extend((v1,v2,*common_neigh))
  if len(to_split)==0:return
  for (v1, v2, v3, v4) in to_split:
    #v_new = HC.split_edge(v1.x,v2.x)
    v1.disconnect(v2)
    # Compute vertex on centre of edge:
    v_pos = 0.5*v1.x_a + 0.5*v2.x_a
    v_new = HC.V[tuple(v_pos)]
    # Connect to original 2 vertices to the new centre vertex
    v_new.connect(v1)
    v_new.connect(v2)
    v_new.connect(v3)
    v_new.connect(v4)
    #HC.split_edge(v1, v2)
  return

def find_quad(v1):
  for v2 in v1.nn:
    common_neigh = list(v1.nn.intersection(v2.nn))
    if len(common_neigh) != 2: continue
    v3 = common_neigh[0]
    v4 = common_neigh[1]
    common_neigh = list(v3.nn.intersection(v4.nn))
    if common_neigh != [v1,v2] and common_neigh != [v2,v1]: continue
    for quad in to_reconnect:
      if len(quad.intersection((v1,v2,v3,v4)))>2: return
    if np.linalg.norm(v1.x_a-v2.x_a) > 1.5*np.linalg.norm(v3.x_a-v4.x_a): 
      to_reconnect.append((v1,v2,v3,v4))

def reconnect_long_diagonals(HC):
  to_reconnect=[]
  for i, v1 in enumerate(HC.V):
    for v2 in v1.nn:
      common_neigh = list(v1.nn.intersection(v2.nn))
      if len(common_neigh) != 2: continue
      v3 = common_neigh[0]
      v4 = common_neigh[1]
      common_neigh = list(v3.nn.intersection(v4.nn))
      if common_neigh != [v1,v2] and common_neigh != [v2,v1]: continue
      if any( [len(set(quad).intersection((v1,v2,v3,v4))) > 2 for quad in to_reconnect] ): continue
      #cont = False
      #for quad in to_reconnect: 
      #  if len(quad.intersection((v1,v2,v3,v4))) > 2:
      #    cont=True 
      #    break
      #if cont: continue
      if np.linalg.norm(v1.x_a-v2.x_a) > 1.5*np.linalg.norm(v3.x_a-v4.x_a): 
        to_reconnect.append((v1,v2,v3,v4))
  if len(to_reconnect)==0:return
  for (v1, v2, v3, v4) in to_reconnect:
    v1.disconnect(v2)
    v3.connect(v4)
  return

def mean_flow(HC, bV, forcePrev, posPrev, params, tau, print_out=False, pinned_line=False):
  (gamma, rho, g, RadFoot, theta_p) = params
  if print_out:
    print('.')
  # Compute interior curvatures
  (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
    HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
    Theta_i_cache) = HC_curvatures_sessile(HC, bV, RadFoot, theta_p, printout=0)

  total_bubble_volume = prism_volume(HC)
  print("total_bubble_volume",total_bubble_volume)
  if total_bubble_volume != total_bubble_volume: return -1
  gasPressure = P_0 + 1e-2*P_0 * ( initial_volume - total_bubble_volume ) / total_bubble_volume
  #if gasPressure<P_0: gasPressure = P_0 
  #gasPressure = P_0 * initial_volume / total_bubble_volume
  print("gasPressure",gasPressure)

  # Move boundary vertices:
  bV_new = set()
  forceDict = {}
  posDict = {}
  for v in HC.V:
    if print_out:
      print(f'v.x = {v.x}')
      if v.x[0] == 0.0 and v.x[1] == 0.0: print(f'='*10)
    netSurfForce = np.array([0.0,0.0,0.0])
    netBuoyancy = np.array([0.0,0.0,0.0])
    netCompressForce = np.array([0.0,0.0,0.0])
    maxMove = 0.0
    # Compute boundary movements
    # Note: boundaries are fixed for now, this is legacy:
    if v in bV:
      rati = (np.pi - np.array(Theta_i) / 2 * np.pi)
      #ToDo: THis is not the actual sector ration (wrong angle)
      # rati = np.array(Theta_i) / (2 * np.pi)
      # print(f' rati = 2 * np.pi /np.array(Theta_i)= { rati}')
      if print_out:
        print('')
        print('HN_i',HN_i_cache[v.x])
        print('C_ij',C_ij_cache[v.x]) #area of each triangle in the dual hexagon
        print('K_H_i',K_H_i_cache[v.x])
        print('HNdA_i_Cij',HNdA_i_Cij_cache[v.x])
        print('Theta_i',Theta_i_cache[v.x])
      if print_out:
        print('-'*10)
        print('Boundary vertex:')
        print('-'*10)
      if 0:
        #ToDo: len(bV) is sector fraction
        H_K = HNda_v_cache[v.x] * np.array([0, 0, -1]) * len(bV)
        print(f'K_H in bV = {H_K }')
        K_H = ((np.sum(H_K) / 2.0) / C_ijk_v_cache[v.x] ) ** 2
        K_H = ((np.sum(H_K) / 2.0)  ) ** 2
        print(f'K_H in bV = {K_H}')
      K_H_dA = K_H_i_cache[v.x] * np.sum(C_ij_cache[v.x])
      #ToDo: Adjust for other geometric approximations:
      l_a = 2 * np.pi * RadFoot / len(bV)  # arc length
      Xi = 1
      # Gauss-Bonnet: int_M K dA + int_dM kg ds = 2 pi Xi
      # Note: Area should be height of spherical cap
      # height = R - RadFoot * 4np.tan(theta_p)
      # Approximate radius of the great shpere K = (1/R)**2:
      R_approx = 1 / np.sqrt(K_H_i_cache[v.x])
      theta_p_approx = np.arccos(np.min([RadFoot / R_approx, 1]))
      height = R_approx - RadFoot * np.tan(theta_p_approx)
      A_approx = 2 * np.pi * R_approx * height  # Area of spherical cap
      # A_approx  # Approximate area of the spherical cap
      kg_ds = 2 * np.pi * Xi - K_H_i_cache[v.x] * (A_approx)
      # ToDo: This is NOT the correct arc length (wrong angle)
      ds = 2 * np.pi * RadFoot  # Arc length of whole spherical cap
      k_g = kg_ds / ds  # / 2.0
      if print_out: print(f' R_approx * k_g = {R_approx * k_g}')
      phi_est = np.arctan(R_approx * k_g)
      # Compute boundary forces
      # N m-1
      if print_out:
        print(f' phi_est = { phi_est}')
        print(f' theta_p = {theta_p}')
      gamma_bt = gamma * (np.cos(phi_est) - np.cos(theta_p)) * np.array([0, 0, 1.0])
      F_bt = gamma_bt * l_a  # N
      F_bt = np.zeros_like(F_bt) # Fix boundaries for now
      new_vx = v.x + tau * F_bt
      #new_vx = v.x + tau * ( F_bt + dc )
      new_vx[2] = 0  # Boundary condition fix to z=0 plane
      new_vx = tuple(new_vx)
      # Move line if the contact angle is not pinned: 
      if not pinned_line: HC.V.move(v, new_vx)
      bV_new.add(HC.V[new_vx])
      #print('-' * 10)
    # Current main code:
    else:
      if print_out:
        print('-'*10)
        print('Interior vertex:')
        print('-'*10)
        print(f' v.x_a = {v.x_a}')
      H = HNdA_i_cache[v.x]
      df = gamma * H
      if print_out:print(f' df = {df}')
      netSurfForce += df
      #ToDo: Convex should be negative, concave parts should be positive.
      if numpy.linalg.norm(H) > 1e-10:
        N_approx = - normalized(H)[0]
      else:#if the surface is flat, we use the direction from the origin
        N_approx = - normalized(v.x_a)[0]
      #dualArea = np.linalg.norm(C_ij_cache[v.x])
      dualArea = sum(C_ij_cache[v.x])
      dc = gasPressure * N_approx  * dualArea
      #print(f' dc = {dc}')
      netCompressForce += dc
      liquidPressure = P_0 #- rho * g * v.x_a[2]
      if liquidPressure<0: print('bubble is too tall, liquidPressure=',liquidPressure)
      dg = - liquidPressure * N_approx  * dualArea
      #print(f' dg = {dg}')
      netBuoyancy += dg
      force = df + dc + dg 
      # Scale forces to characteristic dimension:
      #print('v.x',v.x)
      if v.x in forcePrev:
        #print('AB',forcePrev)
        f_k = v.x_a + tau * ( 1.5*force - 0.5*forcePrev[v.x] )
        #f_k = v.x_a - force * (v.x_a - posPrev[v.x]) / (force - forcePrev[v.x]) 
        #numer=0
        #denom=0
        #for i in range(3):
        #  numer += force[i] * (v.x_a[i] - posPrev[v.x][i]) 
        #  denom += force[i] * (force[i] - forcePrev[v.x][i]) 
        #f_k = v.x_a - force * numer / denom 
      else:
        #print('Euler',forcePrev)
        f_k = v.x_a + tau * force
      maxMove=max(maxMove,np.linalg.norm(df+dg+dc)) 
      #print(f'f_k = {f_k }')
      #ICNov2024 f_k[2] = np.max([f_k[2], 0.0])  # Floor constraint
      new_vx = tuple(f_k)
      old_vx = v.x_a
      # Move interior complex
      HC.V.move(v, new_vx)
      #print('v.x2',v.x)
      forceDict[v.x] = force
      posDict[v.x] = old_vx
      #print('-' * 10)
  if print_out: print(f'bV_new = {bV_new}')
  print('netSurfForce',netSurfForce)
  print('netBuoyancy',netBuoyancy)
  print('netCompressForce',netCompressForce)
  print('maxMove',maxMove)
  if maxMove>2*cdist: 
    print('Warning: maxMove is greater than cdist')
    print('cdist=',cdist)
    print('unmerged vertices may crossover and the interface may overlap')
  print(t,total_bubble_volume,maxMove,*netSurfForce,*netCompressForce,*netBuoyancy,file=vol_txt)
  vol_txt.flush()
  #print(str(total_bubble_volume),file='vol_txt')
  return HC, bV_new, forceDict, posDict

def incr(HC, bV, forcePrev, posPrev, params, tau=1e-5, plot=False, verbosity=1, pinned_line=False):
  HC.dim = 3  # Rest in case visualization has changed
  if verbosity == 2:
    print_out = True
  else:
    print_out = False
  # Update the progress
  HC, bV, forcePrev, posPrev = mean_flow(HC, bV, forcePrev, posPrev, params, tau=tau, print_out=print_out, pinned_line=pinned_line)
  return HC, bV, forcePrev, posPrev

def ps_inc(surface, HC):
  HC.dim = 2  # The dimension has changed to 2 (boundary surface)
  HC.vertex_face_mesh()
  points = np.array(HC.vertices_fm)
  triangles = np.array(HC.simplices_fm_i)
  ### Register a point cloud
  # `my_points` is a Nx3 numpy array
  my_points = points
  ps_cloud = ps.register_point_cloud("my points", my_points)
  # ps_cloud.set_color((0.0, 0.0, 0.0))
  verts = my_points
  newPositions = verts
  surface.update_vertex_positions(newPositions)
  try:
    with timeout(0.1, exception=RuntimeError):
      # perform a potentially very slow operation
      ps.show()
  except RuntimeError:
    pass

def spherical_cap_init(RadFoot, theta_p, NFoot=4, refinement=0, cdist=-1):
  RadSphere = RadFoot / np.sin(theta_p)
  print('theta_p',theta_p) 
  print('RadSphere',RadSphere)
  print('RadFoot',RadFoot)
  Cone = []
  nn = []
  Cone.append(np.array([0.0, 0.0, 0.0])) #IC middle vertex
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
  ##Add layer
  #N=len(Cone)
  #for i in range(N):
  #  if Cone[i][2]==.5
  #  Cone.append(Cone[i][0], np.cos(phi), 1])) #IC contact line circle
  #  # Define connections:
  #  nn.append([])
  #  if ind > 0:
  #    nn[0].append(ind)
  #    nn[ind].append(0)
  #    nn[ind].append(ind - 1)
  #    nn[ind].append((ind + 1) % NFoot)
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
  #plot_polyscope(HC)
  # Move to spherical cap
  #thetaResolve=.95*theta_p
  #thetaFoot=.01*theta_p
  #thetaTop=100000#.2*theta_p
  #C = 1/thetaTop
  #B = 3*theta_p - 1/thetaFoot + 2/thetaTop
  #A = 1/thetaFoot - 3/thetaTop - 2*theta_p 
  #theta = lambda z : A*z**3 + B*z**2 + C*z 
  theta = lambda z : theta_p*min(2*z, .01*(z-1) + 1) 
  #print('A=',A)
  #print('B=',B)
  #print('C=',C)
  #print('theta_p',theta_p)
  #print('theta(0)',theta(0))
  #print('theta(.5)',theta(.5))
  #print('theta(1)',theta(1))
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
tau = .001 #0.001  # 0.1 works
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
RadTop = (Bo*gamma/rho/g)**.5 #250e-6 #8.426e-4 #300e-6
#print('RadSphere',RadSphere)
#RadFoot = .1*RadSphere# 2e-6 #1e-4#((44 / 58) * 0.25 + 1.0) * 1e-3  # height mm --> m  Radius 10e-6
#theta_p = np.pi - np.asin(RadFoot / RadSphere)
#RadFoot = (180 - theta_p)*RadSphere
#theta_p = theta_p * np.pi / 180.0
#height =  ((3 / 58) * 0.25 + 0.5) * 1e-3  # height mm --> m 10e-6
#height = RadFoot / np.sin(theta_p) * ( 1 - np.cos(theta_p) )
#Volume = np.pi * (3 * RadFoot ** 2 * height + height ** 3) / 6.0  # Volume in m3 (Segment of a sphere, see note above)
#print(f'theta_p = {theta_p * 180.0 / np.pi}')
#print(f'RadFoot = {RadFoot * 1e3} mm')
#print(f'height = {height * 1e3} mm')
#print(f'Volume = {Volume} m^3')

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
RadFoot = 1e-3
Volume = np.pi/3*RadTop**3
print(f'RadFoot = {RadFoot}')
params =  (gamma, rho, g, RadFoot, theta_p)
# Colour scheme for surfaces
db = np.array([129, 160, 189]) / 255  # Dark blue
lb = np.array([176, 206, 234]) / 255  # Light blue

tInit=0
t=tInit
if tInit==0:
  HC,bV = cone_init(RadFoot, Volume, NFoot=6, refinement=1, cdist=cdist)
else:
  HC, bV = loadComplex(t)
#for mer in range():
#  HC.V.merge_nn(cdist=cdist)
#ps.init()
#plt.show()
#HC.refine_all_star(exclude=bV)
plot_polyscope(HC)
#(HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
#  HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
#  Theta_i_cache) = HC_curvatures_sessile(HC, bV, RadFoot, theta_p, printout=0)

  #dualArea = sum(C_ij_cache[v.x])
  #if i==10:#dualArea>10*cdist**2:
  #  HC.refine_star(v)

refine_edges(HC, .002)#*cdist)
reconnect_long_diagonals(HC)
plot_polyscope(HC)
initial_volume = Volume#prism_volume(HC)
fname='data/vol.txt'
with open(fname, "a") as vol_txt:
  print('saving',fname)
  forcePrev = {}
  posPrev = {}
  #Surface energy minimisation
  while t<=tInit+10:
    t+=1
    print("t",t)
    HC, bV, forcePrev, posPrev = incr(HC, bV, forcePrev, posPrev, params, tau=tau, plot=0, verbosity=0, pinned_line=pinned_line)
    #HC.V.merge_nn(cdist=cdist)
    #refineCentroids()
    refine_edges(HC, .002)#*cdist)
    reconnect_long_diagonals(HC)
    plot_polyscope(HC)
    for v in HC.V:
      if len(v.nn)<2:  
          print('v.nn',v.nn)
          print('remove',v.x_a)
          HC.V.remove(v)
    if t%10==0:saveComplex(t)
      #ps.frame_tick()
plot_polyscope(HC)
plt.show()
