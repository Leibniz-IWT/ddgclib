#Ianto Cannon 2025 Fab 13. 
#Functions for calculating the interface shape of bubbles on surfaces
#Adapted from code written by Stefan Endres
import numpy as np
from ddgclib._curvatures import HC_curvatures_sessile, construct_HC
from ddgclib._volume import triangle_prism_volume, cross_prod

def save_neighbours(fname,HC):
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

def save_vert_positions(t,HC):
#Save the xyz position of all vertices in a text file.
  fname='data/pos'+str(t)+'.txt'
  with open(fname, "w") as pos_txt:
    print('saving',fname)
    for v in HC.V:
      print(*v.x,file=pos_txt)
  save_neighbours('data/nei'+str(t)+'.txt',HC)

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
        if any(n in to_split_1D for n in common_neigh): continue
        to_split.append((v1,v2,*common_neigh))
        to_split_1D.extend((v1,v2,*common_neigh))
  if len(to_split)==0:return 0
  for (v1, v2, *common_neigh) in to_split:
    v1.disconnect(v2)
    # Compute vertex on centre of edge:
    v_pos = 0.5*v1.x_a + 0.5*v2.x_a
    v_new = HC.V[tuple(v_pos)]
    # Connect to original 2 vertices to the new centre vertex
    v_new.connect(v1)
    v_new.connect(v2)
    for n in common_neigh:
      v_new.connect(n)
  return len(to_split)

def refine_boundaries(HC, bV, dist):
#Put vertices on edges longer than dist.
#Like fig 2a of Unverdi and Tryggvason J comp. phys. 1992.
#Neighbouring edges are not refined.
  to_split=[]
  to_connect=[]
  for i, v1 in enumerate(bV):
    for v2 in v1.nn:
      if v2 not in bV: continue
      if (v2,v1) in to_split: continue
      if np.linalg.norm(v1.x_a-v2.x_a) > dist: 
        common_neigh = list(v1.nn.intersection(v2.nn))
        #Only refine if the vertices are in a planar region
        if len(common_neigh) != 1: continue
        #if any(n in to_split_1D for n in common_neigh): continue
        if any(n in bV for n in common_neigh): continue
        to_split.append((v1,v2))
        to_connect.append(common_neigh[0])
  if len(to_split)==0: return 0
  for i in range(len(to_split)):
    (v1, v2) = to_split[i]
    n = to_connect[i]
    v1.disconnect(v2)
    # Compute vertex on centre of edge:
    v_pos = 0.5*v1.x_a + 0.5*v2.x_a
    v_new = HC.V[tuple(v_pos)]
    # Connect to original 2 vertices to the new centre vertex
    v_new.connect(v1)
    v_new.connect(v2)
    v_new.connect(n)
    bV.add(v_new)
  return len(to_split)

def reconnect_long_diagonals(HC, bV):
#Find bisected quadrilaterals and
#Like fig 2c of Unverdi and Tryggvason J comp. phys. 1992.
#Neighbouring edges are not refined.
  to_reconnect=[]
  for v1 in HC.V:
    for v2 in v1.nn:
      common_neigh = list(v1.nn.intersection(v2.nn))
      #Only reconnect if the vertices are in a planar region
      if len(common_neigh) != 2: continue
      v3 = common_neigh[0]
      v4 = common_neigh[1]
      #don't connect boundary verts
      if v3 in bV and v4 in bV: continue
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

def remesh(HC, minEdge, maxEdge, bV=set()):
#Remesh into roughly equilateral triangles with a size between minEdge and maxEdge
#Using the method in fig 2 of Unverdi and Tryggvason J comp. phys. 1992.
  HC.V.merge_nn(cdist=minEdge, exclude=bV)
  refine_edges(HC, maxEdge)
  refine_boundaries(HC, bV, maxEdge)
  reconnect_long_diagonals(HC, bV)
  #Remove vertices that get disconnected due to merging
  for v in HC.V:
    if len(v.nn)<2: HC.V.remove(v)

def get_forces(HC, bV, t, params):
#get the surface tension and pressure forces on the vertices
  RadFoot=1
  theta_p=np.pi/2
  # Compute interior curvatures
  (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
    HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
    Theta_i_cache) = HC_curvatures_sessile(HC, bV, RadFoot, theta_p, printout=0)
  total_bubble_volume, total_bubble_area, bubble_centroid = triangle_prism_volume(HC)
  #print(t,'vol',total_bubble_volume)
  if total_bubble_volume != total_bubble_volume: raise ValueError('The bubble volume is not a number')
  #gasPressure = params['P_0'] * (params['initial_volume']/total_bubble_volume - 1)
  #RadBub = (3 * total_bubble_volume / 2 / np.pi) ** (1/3)
  #gasPressure = 2*params['gamma']/RadBub * (params['initial_volume']/total_bubble_volume)**(.5)
  gasPressure = params['P_0'] * (params['initial_volume']/total_bubble_volume)**5
  forceDict = {}
  posDict = {}
  maxForce = 0.0
  net_interf_force = np.array([0.0,0.0,0.0])
  net_liq_force = np.array([0.0,0.0,0.0])
  net_gas_force = np.array([0.0,0.0,0.0])
  net_solid_force= np.array([0.0,0.0,0.0])
  height = max([v.x_a[2] for v in HC.V])
  for v in HC.V:
    force = np.array([0.0,0.0,0.0])
    #height = max(height,v.x_a[2])
    # Compute boundary movements
    # Note: boundaries are fixed for now, this is legacy:
    if v in bV:
      #get boundary sector length and normal, perhaps with b_curvatures
      for vn in v.nn:
        if vn in bV:
          cLineLen = np.linalg.norm(v.x_a - vn.x_a)
          common_neigh = list(v.nn.intersection(vn.nn))
          if len(common_neigh)==1:
            midpoint = .5*v.x_a + .5*vn.x_a
            pullDirection = midpoint - common_neigh[0].x_a
            ContactForce = params['gamma'] * cLineLen * pullDirection / np.linalg.norm(pullDirection) / 2
            #divide by 2 as two vertices per edge
            net_solid_force += ContactForce
          else: print('len common_neigh',len(common_neigh))
    else:
      H = HNdA_i_cache[v.x]
      interf_force = params['gamma'] * H
      net_interf_force += interf_force
      if False:
        dualNormal = outward_normal(v,H)
        dualArea = sum(C_ij_cache[v.x])
        gas_force = gasPressure * dualNormal  * dualArea
        liquidPressure = P_0 - rho * g * v.x_a[2]
        if liquidPressure<0: print('bubble is too tall, liquidPressure=',liquidPressure)
        liq_force = - liquidPressure * dualNormal  * dualArea
      else:
        gas_force = 0
        liq_force = 0
        for vn1 in v.nn:
          for vn2 in v.nn:
            if vn2 in vn1.nn:
              triArea = .5*cross_prod(vn1.x_a-v.x_a, vn2.x_a-v.x_a)
              #set the area vector pointing away from the z axis
              centroid = ( v.x_a + vn1.x_a + vn2.x_a )/3
              if sum( triArea[:3] * centroid[:3] ) < 0: triArea = - triArea
              #divide by 2 because vn1 and vn2 can be swapped
              #divide by 3 because each triangle contributes to 3 vertices
              gas_force += triArea*gasPressure /2 /3
              liquidPressure = params['rho'] * params['g'] * (height - centroid[2]) #+ params['P_0']
              if liquidPressure<0: raise ValueError('bubble is too tall, height =', centroid[2])
              liq_force -= triArea*liquidPressure /2 /3
      #gasByInter = sum(gas_force[:]**2) / sum(interf_force[:]**2)
      #if gasByInter>4:
      #  #print('gasByInter',gasByInter)
      #  gas_force *= 2/gasByInter**.5
      net_gas_force += gas_force
      net_liq_force += liq_force
      #if v.x_a[2]>.00302:
      #  print('liquidPressure',liquidPressure)
      #  print('gasPressure',gasPressure)
      #  print('HNdA_i_Cij',HNdA_i_Cij_cache[v.x])
      force = interf_force + liq_force + gas_force 
      maxForce=max( maxForce, np.linalg.norm(force) ) 
    forceDict[v.x] = force
  with open('data/vol.txt', "a") as vol_txt:
    print(t,total_bubble_volume,gasPressure,maxForce,height,bubble_centroid,*net_interf_force,*net_gas_force,*net_liq_force,*net_solid_force,file=vol_txt)
    vol_txt.flush()
  return forceDict, maxForce

def grad_energy(posArray, *args):
#compute the gradient of the total energy wrt to the position of each vertex. 
#I.e., an array of -force 
#used by scipy line_search
  (HC, bV, t, params) = args
  HC_temp = HC
  bV_temp = set()
  for i,(v_temp, v) in enumerate(zip(HC_temp.V,HC.V)):
    HC_temp.V.move(v_temp, tuple(posArray[i*3:(i+1)*3]))
    if v in bV: bV_temp.add(v_temp)
  forceDict, maxF  = get_forces(HC_temp, bV_temp, t, params)
  forArray = np.array([fi for f in forceDict.values() for fi in f])
  return - forArray

def get_energy(HC, t, params):
#Compute the energy of the interface
  total_bubble_volume, total_bubble_area, bubble_centroid = triangle_prism_volume(HC)
  idealGasEn = params['P_0']*params['initial_volume']*np.log(params['initial_volume']/total_bubble_volume)
  interfaceEn = params['gamma']*total_bubble_area
  gravityEn = - params['rho']*params['g']*bubble_centroid*total_bubble_volume
  fname='data/energy.txt'
  with open(fname, "a") as en_txt:
    print(t, idealGasEn, interfaceEn, gravityEn, file=en_txt)
    en_txt.flush()
  return interfaceEn + gravityEn + idealGasEn 

def get_energy_from_array(posArray, *args):
#Compute the energy of the interface from an array of vertex positions.
#used by scipy line_search
  (HC, bV, t, params) = args
  HC_temp = HC
  for i, v_temp in enumerate(HC_temp.V):
    HC_temp.V.move(v_temp, tuple(posArray[i*3:(i+1)*3]))
  total_bubble_volume, total_bubble_area, bubble_centroid = triangle_prism_volume(HC_temp)
  idealGasEn = params['P_0']*params['initial_volume']*np.log(params['initial_volume']/total_bubble_volume)
  interfaceEn = params['gamma']*total_bubble_area
  gravityEn = - params['rho']*params['g']*bubble_centroid*total_bubble_volume
  fname='data/energy.txt'
  with open(fname, "a") as en_txt:
    print(t, idealGasEn, interfaceEn, gravityEn, file=en_txt)
    en_txt.flush()
  return idealGasEn + interfaceEn + gravityEn


def correct_the_volume(HC, bV, target_volume):
#Move the vertices out or in along the local normal to set the bubble volume
  # Compute interior curvatures
  (HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
    HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
    Theta_i_cache) = HC_curvatures_sessile(HC, bV, 1, np.pi/2, printout=0)
  total_bubble_volume, total_bubble_area, bubble_centroid = triangle_prism_volume(HC)
  shift = ( target_volume - total_bubble_volume ) / total_bubble_area
  for v in HC.V:
    if v not in bV:
      H = HNdA_i_cache[v.x]
      #dualNormal = outward_normal(v,H)
      dualNormal = v.x_a/sum(v.x_a[:]**2)**.5
      HC.V.move(v, tuple(v.x_a + dualNormal*shift))
  return 

def AdamsBashforthProfile(Bo, RadTop):
#compute analytical interface shape according to eq 1 of Demirkir2024Langmuir
#Input the Bond number Bo, and the radius of curvature at bubble top; RadTop
#Return the volume of the bubble, radius of the contact patch, height of bubble 
#and height of the centre of mass.
  d=.0001
  psi=0
  r=0
  z=0
  Volume=0
  centroid=0
  fname='data/adams'+str(Bo)+'.txt'
  with open(fname, "w") as adams_txt:
    print('saving',fname)
    for i in range(int(10/d)):
      r += d * np.cos(psi)
      dz = d * np.sin(psi)
      z += dz
      Volume += np.pi*r**2*dz
      centroid += z*np.pi*r**2*dz
      #if i*d*100%1 == 0: print(r*RadTop, -z*RadTop, file=adams_txt)
      print(r*RadTop, -z*RadTop, file=adams_txt)
      psi += d * (2 - Bo*z - np.sin(psi)/r)
      #if 2 - Bo*z - np.sin(psi)/r < 0: break
      #if z < -.4: break
      #if psi > np.pi/2: break
      if psi > np.pi: break
      if psi < .4*np.pi and 2 - Bo*z - np.sin(psi)/r < 0: break
  centroid /= Volume
  return Volume*RadTop**3, r*RadTop, z*RadTop, (z-centroid)*RadTop
