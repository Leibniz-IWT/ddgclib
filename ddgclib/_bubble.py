#Ianto Cannon 2025 Fab 13. 
#Functions for calculating the interface shape of bubbles on surfaces
#Adapted from code written by Stefan Endres
import numpy as np
from ddgclib._curvatures import construct_HC#, HC_curvatures_sessile 
from ddgclib._curvatures_heron import hndA_i
from ddgclib.geometry._volume import triangle_prism_volume, cross_prod

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

def save_vert_positions(t, HC):
#Save the xyz position of all vertices in a text file.
  total_bubble_volume, total_bubble_area, bubble_centroid = triangle_prism_volume(HC)
  bubble_centroid[2] = 0
  fname='data/pos'+str(t)+'.txt'
  with open(fname, "w") as pos_txt:
    print('saving',fname)
    for v in HC.V:
      pos = v.x - bubble_centroid
      print(*pos,file=pos_txt)
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

def refine_edges(HC, bV, dist):
#Add vertices to edges which are longer than dist.
#Like fig 2a of Unverdi and Tryggvason J comp. phys. 1992.
#Neighbouring edges are not refined.
  to_split_1D=[]
  to_split=[]
  for i, v1 in enumerate(HC.V):
    if v1 in to_split_1D: continue
    #if v1 in bV: continue
    for v2 in v1.nn:
      if v2 in to_split_1D: continue
      sep = sum( (v1.x_a[:]-v2.x_a[:])**2 )
      #if v1 in bV or v2 in bV: sep*=16
      if sep > dist**2: 
        common_neigh = list(v1.nn.intersection(v2.nn))
        #Only refine if the vertices are in a planar region. 
        #This will exclude boundaries
        if len(common_neigh) > 2: continue
        if any(n in to_split_1D for n in common_neigh): continue
        to_split.append((v1,v2,*common_neigh))
        to_split_1D.extend((v1,v2,*common_neigh))
  if len(to_split)==0:return 0
  for (v1, v2, *common_neigh) in to_split:
    v1.disconnect(v2)
    # Compute vertex on centre of edge:
    v_pos = 0.5*v1.x_a + 0.5*v2.x_a
    if v1 in bV and v2 in bV: v_pos *= ( sum(v1.x_a**2) / sum(v_pos**2) )**.5
    v_new = HC.V[tuple(v_pos)]
    if v1 in bV and v2 in bV: 
      radPos = v_new.x_a*sum(v1.x_a**2)/sum(v_pos**2)
      bV.add(v_new)
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
      #if v2 not in bV: continue
      if (v2,v1) in to_split: continue
      if np.linalg.norm(v1.x_a-v2.x_a) > dist: 
        common_neigh = list(v1.nn.intersection(v2.nn))
        #Only refine if the vertices are in a planar region
        #if len(common_neigh) != 1: continue
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
    if v1 in bV and v2 in bV: bV.add(v_new)
  return len(to_split)

def reconnect_long_diagonals(HC, bV):
#Find bisected quadrilaterals and put the bisection between the closer pair of vertices.
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
    if v1 in bV and v2 in bV: 
      print('skip reconnecting boundary',v2.x)
      continue
    v1.disconnect(v2)
    v3.connect(v4)
  return len(to_reconnect)

def remesh(HC, minEdge, maxEdge, bV=set()):
#Remesh into roughly equilateral triangles with a size between minEdge and maxEdge
#Using the method in fig 2 of Unverdi and Tryggvason J comp. phys. 1992.
  bVneigh = list(bV)
  for v in bV:
    for vn in v.nn:
      bVneigh.append(vn)
  HC.V.merge_nn(cdist=minEdge, exclude=bVneigh)
  refine_edges(HC, bV, maxEdge)
  #refine_boundaries(HC, bV, .2*maxEdge)
  reconnect_long_diagonals(HC, bV)
  #Remove vertices that get disconnected due to merging
  for v in HC.V:
    if len(v.nn)<2: HC.V.remove(v)

def move(v, pos, HC, bV):
#Function wrapper which preserves the list of boundary vertices
  if v in bV: 
    bV.remove(v)
    HC.V.move(v, tuple(pos))
    bV.add(v)
  else:
    HC.V.move(v, tuple(pos))

def get_forces(HC, bV, t, params):
#get the surface tension and pressure forces on the vertices
  RadFoot=1
  # Compute interior curvatures
  #(HN_i, C_ij, K_H_i, HNdA_i_Cij, Theta_i,
  #  HNdA_i_cache, HN_i_cache, C_ij_cache, K_H_i_cache, HNdA_i_Cij_cache,
  #  Theta_i_cache) = HC_curvatures_sessile(HC, bV, RadFoot, params['contactAng'], printout=0)
  total_bubble_volume, total_bubble_area, bubble_centroid = triangle_prism_volume(HC)
  if total_bubble_volume != total_bubble_volume: raise ValueError('The bubble volume is not a number')
  gasPressure = params['targetPressure'] * (params['targetVol']/total_bubble_volume)**5
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
    #H = HNdA_i_cache[v.x]
    #interf_force = params['gamma'] * H
    #print('H',H)
    HNdA_i, C_i = hndA_i(v)#, n_i=n_i)
    #print('v.x',v.x)
    #print('HNdA_i',HNdA_i)
    #print('HNdA_i/H',HNdA_i/H)
    #print(' ')
    interf_force = - params['gamma'] * HNdA_i
    net_interf_force += interf_force
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
          liquidPressure = params['density'] * params['gravity'] * (height - centroid[2]) 
          if liquidPressure<0: raise ValueError('bubble is too tall, height =', centroid[2])
          liq_force -= triArea*liquidPressure /2 /3
    force = interf_force + liq_force + gas_force 
    net_gas_force += gas_force
    net_liq_force += liq_force
    #Compute force on boundary
    if v in bV:
      #force *= 0
      #get boundary sector length and normal, perhaps with b_curvatures
      for vn in v.nn:
        if vn in bV:
          common_neigh = list(v.nn.intersection(vn.nn))
          if len(common_neigh)!=1:
            print('boundary vertices have', len(common_neigh), 'common_neigh')
            continue
          localNormal = cross_prod(common_neigh[0].x_a - v.x_a, common_neigh[0].x_a - vn.x_a)
          localNormal /= sum(localNormal[:]**2)**.5
          #Interface force on the contact line, divide by 2 as two vertices per edge
          ContactForce = params['gamma'] * cross_prod(v.x_a-vn.x_a, localNormal) / 2
          net_solid_force += ContactForce
          force += ContactForce*[1,1,0]
          pullDir = cross_prod(v.x_a-vn.x_a, [0,0,1])
          if sum(pullDir*v.x_a)<0: pullDir *= -1
          #Solid force on the contact line, divide by 2 as two vertices per edge
          solidForce = params['gamma']*np.cos(params['contactAng']) * pullDir / 2
          force += solidForce
      force *= 0#[1,1,0]
    maxForce=max( maxForce, np.linalg.norm(force) ) 
    forceDict[v.x] = force
  with open('data/vol.txt', "a") as vol_txt:
    print(t,total_bubble_volume,gasPressure,maxForce,height,*bubble_centroid,*net_interf_force,*net_gas_force,*net_liq_force,*net_solid_force,file=vol_txt)
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
  idealGasEn = params['targetPressure']*params['targetVol']*np.log(params['targetVol']/total_bubble_volume)
  interfaceEn = params['gamma']*total_bubble_area
  gravityEn = - params['density']*params['gravity']*bubble_centroid[2]*total_bubble_volume
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
  idealGasEn = params['targetPressure']*params['targetVol']*np.log(params['targetVol']/total_bubble_volume)
  interfaceEn = params['gamma']*total_bubble_area
  gravityEn = - params['density']*params['gravity']*bubble_centroid[2]*total_bubble_volume
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

def AdamsBashforthProfile(Bo, RadTop, contactAng=-1, fname=None):
#compute analytical interface shape according to eq 1 of Demirkir2024Langmuir
#Input the Bond number Bo, and the radius of curvature at bubble top; RadTop
#Return the volume of the bubble, radius of the contact patch, height of bubble 
#and height of the centre of mass.
  d=.001*min(1./Bo,1)
  psi=0
  r=0
  z=0
  Volume=0
  centroid=0
  #fname='data/adams'+str(Bo)+'.txt'
  if fname: 
    adams_txt = open(fname, "w") 
    print('saving',fname)
  for i in range(int(1e5)):#10/d)):
    r += d * np.cos(psi)
    dz = d * np.sin(psi)
    z += dz
    Volume += np.pi*r**2*dz
    centroid += z*np.pi*r**2*dz
    if fname and not i%10: print(r*RadTop, -z*RadTop, psi, Bo, file=adams_txt)
    psi += d * (2 - Bo*z - np.sin(psi)/r)
    #if z < -.4: break
    if 2-Bo*z-np.sin(psi)/r < 0:
      if contactAng<0: break
      elif psi < contactAng: break
    #if contactAng>0 and 2-Bo*z-np.sin(psi)/r < 0 and psi < contactAng: 
    #  print('brea i',i, 'psi',psi)
    #  break
  if i>int(1e5-2): print('i',i, 'contactAng', contactAng)
  #if fname: close(adams_txt)
  centroid /= Volume
  return Volume*RadTop**3, r*RadTop, z*RadTop, (z-centroid)*RadTop, np.pi-psi
