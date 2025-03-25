#Ianto Cannon 2025 Feb 20
#Functions for making complexes and measuring their volume
#Adapted from code written by Stefan Endres

import numpy as np
# Allow for relative imports from main library:
#import sys
#import os
#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.dirname(SCRIPT_DIR))
from ddgclib._curvatures import HC_curvatures_sessile, construct_HC, normalized

def cross_prod(a,b):
#homemade function which returns the cross product of two vectors
#you may use this if you wear a tinfoil hat and do not trust the numpy.cross
  return np.array([ a[1]*b[2]-a[2]*b[1],
                    a[2]*b[0]-a[0]*b[2],
                    a[0]*b[1]-a[1]*b[0]  ])

def outward_normal(vert, meanCurv):
#Compute the normal, and ensure it points away from the z axis.
#If the surface is flat we use the direction from the origin.
  if sum(meanCurv[:]**2) > 1e-5:
    N_approx = - normalized(meanCurv)[0]
  else:
    N_approx = normalized(vert.x_a)[0]
  if sum(N_approx[:3]*vert.x_a[:3])<0: N_approx = -N_approx
  return N_approx


def sector_volume(HC):
#Compute volume of the complex by splitting it into sectors centred on origin.
  (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
    HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
    Theta_i_cache) = HC_curvatures_sessile(HC, set(), 1, np.pi/2, printout=0)
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
#Compute volume of the complex as a sum of prisms.
#The solid surface is the prism base, and the vertex dual area is the top.
#To do: include the boundaries
  (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
    HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
    Theta_i_cache) = HC_curvatures_sessile(HC, set(), 100, np.pi/10, printout=0)
  total_bubble_volume=0.0
  totalArea=0.0
  for v in HC.V:
    dualArea = sum(C_ij_cache[v.x])
    totalArea += dualArea
    H = HNdA_i_cache[v.x]
    N_approx = outward_normal(v,H)
    total_bubble_volume += dualArea * N_approx[2] * v.x_a[2] #+ np.dot(H, v.x_a) 
  return total_bubble_volume

def triangle_prism_volume(HC):
#Compute volume of the complex as a sum of triangular prisms.
#The solid surface is the prism base, and the triangular simplex is the top.
  total_bubble_volume=0.0
  total_bubble_area=0.0
  bubble_centroid=np.array([0.0, 0.0, 0.0])
  for v in HC.V:
    for vn1 in v.nn:
      for vn2 in v.nn:
        if vn2 in vn1.nn:
          triArea = .5*cross_prod(vn1.x_a-v.x_a, vn2.x_a-v.x_a)
          #set the area vector pointing away from the z axis
          if sum(triArea[:3]*v.x_a[:3])<0: triArea = - triArea
          triangle_centroid = ( v.x_a + vn1.x_a + vn2.x_a )/3
          prism_volume = triArea[2] * triangle_centroid[2] /2/3 #+ np.dot(H, v.x_a) 
          total_bubble_volume += prism_volume
          bubble_centroid += triangle_centroid*prism_volume
          total_bubble_area += np.linalg.norm(triArea)/2/3
          #divide by 2 because vn1 and vn2 are equivalent
          #divide by 3 because each triangle is counted at the three vertices
  bubble_centroid /= total_bubble_volume
  #the centre of mass is halfway above the base
  bubble_centroid[2] /= 2
  return total_bubble_volume, total_bubble_area, bubble_centroid

def spherical_cap_contact_angle(RadFoot, vol):
#Get the inside contact angle of a spherical cap with a circular foot of radius RadFoot and volume vol
#Iteratively solve the following equation for the height
#vol = np.pi / 6 * height * (3*RadFoot**2 + height**2)
  height = RadFoot
  alpha = .1
  for i in range(100):
    height = height*(1-alpha) + alpha* 6*vol / np.pi / (3*RadFoot**2 + height**2)
  RadSphere = ( RadFoot**2 + height**2 ) / 2 / height
  contacAng = np.atan2( RadFoot, RadSphere - height )
  if False:
    foundVol = np.pi/3 * ( RadFoot / np.sin(contacAng) )**3 * ( 2 + np.cos(contacAng) ) * ( 1 - np.cos(contacAng) )**2
    print('cap vol error', (foundVol - vol)/vol)
  return RadSphere, contacAng


def spherical_cap_init(RadSphere, theta_p, NFoot=6, maxEdge=-1):
#Create a complex in the shape of a sphere with contact angle theta_p
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
  if maxEdge>0:
    #from ddgclib._plotting import plot_polyscope
    for i in range(5):
      #print('refine',i)
      #plot_polyscope(HC)
      v1 = list(v0.nn)[0]
      if sum((v0.x_a[:]-v1.x_a[:])**2) < (maxEdge/RadSphere)**2: break
      HC.refine_all_star()#exclude=bV)
      HC.V.merge_all(cdist=RadSphere*1e-3)
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
  # Rebuild set after moved vertices (appears to be needed)
  bV = set()
  for v in HC.V:
    if abs(v.x[2]) < RadSphere*1e-4: bV.add(v)
  return HC, bV

def cone_init(RadFoot, Volume, NFoot=6, maxEdge=-1):
#Make a cone shaped complex with radius RadFoot.
# use NFoot vertices at the base, and refine the edges if maxEdge is positive
  height = 3*Volume / np.pi / RadFoot**2
  Cone = []
  nn = []
  Cone.append(np.array([0.0, 0.0, height])) #IC middle vertex
  nn.append([])
  ind = 0
  #Make cone
  for phi in np.linspace(0.0, 2 * np.pi, NFoot + 1):
    ind += 1
    Cone.append(np.array([RadFoot*np.sin(phi), RadFoot*np.cos(phi), 0])) #IC contact line circle
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
  # link the beginning and end of the boundary
  nn[1][1] = ind
  # Construct complex from the cone geometry:
  HC = construct_HC(Cone, nn)   
  v0 = HC.V[tuple(Cone[0])]
  # Compute boundary vertices
  V = set()
  for v in HC.V:
    V.add(v)
  bV = V - set([v0])
  if maxEdge>0:
    #from ddgclib._plotting import plot_polyscope
    for i in range(5):
      #print('refine',i)
      #plot_polyscope(HC)
      v1 = list(v0.nn)[0]
      if sum((v0.x_a[:]-v1.x_a[:])**2) < maxEdge**2: break
      HC.refine_all_star()#exclude=bV)
      HC.V.merge_all(cdist=.01*maxEdge)
      #refine_edges(HC, maxEdge)
      #refine_boundaries(HC, bV, maxEdge)
      #reconnect_long_diagonals(HC, bV)
  #move refined vertices to circular cone
  for v in HC.V:
    z = v.x_a[2]
    Rad = RadFoot * (height - z) / height
    phi = np.arctan2(v.x_a[1],v.x_a[0])
    x = Rad* np.cos(phi)
    y = Rad* np.sin(phi)
    HC.V.move(v, tuple((x,y,z)))
  # Rebuild set after moved vertices (appears to be needed)
  bV = set()
  for v in HC.V:
    if abs(v.x[2]) < height*1e-4: bV.add(v)
    if sum(abs(v.x_a[:])) < height*1e-4: HC.V.remove(v)
    #if v.x[0] > 1e-4: HC.V.remove(v)
  return HC, bV

