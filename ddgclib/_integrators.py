from ddgclib._bubble import save_vert_positions, get_forces, correct_the_volume, grad_energy, get_energy_from_array, get_energy, remesh

def Euler(HC, bV, params, tInit, nSteps, stepSize, minEdge=-1, maxEdge=-1, implicitVolume=False, constMoveLen=False):
#Reduce the interface energy by an Eulerian method
#If implicitVolume, the volume is corrected at every timestep
#If constMoveLen, the step is adapted so that the maximum distance moved is equal to stepSize
  t = tInit
  while t <= tInit+nSteps:
    print('t',t)
    if t%10==0: save_vert_positions(t, HC)
    get_energy(HC, t, params)
    t+=1
    if minEdge>0: remesh(HC,minEdge,maxEdge)
    forceDict, maxForce = get_forces(HC, bV, t, params)
    if not constMoveLen: maxForce=1
    for v in HC.V:
      if v.x in forceDict:
        HC.V.move(v, tuple(v.x_a + stepSize*forceDict[v.x]/maxForce))
    if implicitVolume: correct_the_volume(HC, bV, params['initial_volume'])
  return t
  
def AdamsBashforth(HC, bV, params, tInit, nSteps, stepSize, minEdge=-1, maxEdge=-1, implicitVolume=False): 
#Reduce the interface energy by an Adams Bashforth method
#The first iteration is Eulerian
  forcePrev = {}
  t = tInit
  while t <= tInit+nSteps:
    print('t',t)
    if t%int(nSteps/10)==0: save_vert_positions(t, HC)
    get_energy(HC, t, params)
    t+=1
    if minEdge>0: remesh(HC,minEdge,maxEdge)
    forceDict, maxForce = get_forces(HC, bV, t, params)
    for v in HC.V:
      if v.x not in forceDict: continue
      if v.x in forcePrev: x_new = tuple(v.x_a + stepSize*(1.5*forceDict[v.x] - 0.5*forcePrev[v.x]))
      else: x_new = tuple(v.x_a + stepSize*forceDict[v.x])
      forcePrev[x_new] = forceDict[v.x]
      HC.V.move(v, tuple(x_new))
    if implicitVolume: correct_the_volume(HC, bV, params['initial_volume'])
  return t
  
def NewtonRaphson(HC, bV, params, tInit, nSteps, stepSize, minEdge=-1, maxEdge=-1, implicitVolume=False): 
#Find minimum interface energy by nSteps applications of the Newton Raphson method
#stepSize is used for the first iteration, which is Eulerian
  t = tInit
  forcePrev = {}
  posPrev = {}
  while t <= tInit+nSteps:
    print('t',t)
    if t%10==0: save_vert_positions(t, HC)
    get_energy(HC, t, params)
    t+=1
    if minEdge>0: remesh(HC,minEdge,maxEdge)
    forceDict, maxForce = get_forces(HC, bV, t, params)
    for v in HC.V:
      x_new = -1 
      if v.x in forcePrev:
        numer=0
        denom=0
        for i in range(3):
          numer += forceDict[v.x][i] * (v.x_a[i] - posPrev[v.x][i]) 
          denom += forceDict[v.x][i] * (forceDict[v.x][i] - forcePrev[v.x][i]) 
        if sum( forceDict[v.x][:]**2 ) * (numer/denom)**2 < stepSize**2: x_new = tuple(v.x_a - forceDict[v.x] * numer / denom)
      if v.x in forceDict and x_new==-1: x_new = tuple(v.x_a + stepSize*forceDict[v.x]/maxForce)
      if x_new==-1: continue
      posPrev[x_new] = v.x_a
      forcePrev[x_new] = forceDict[v.x]
      HC.V.move(v, tuple(x_new))
    if implicitVolume: correct_the_volume(HC, bV, params['initial_volume'])
  return t

def lineSearch(HC, bV, params, tInit, nSteps, stepSize, minEdge=-1, maxEdge=-1, implicitVolume=False):
#Reduce the interface energy by an Eulerian method, where the step size is chosen at each timestep using a line search
#If implicitVolume, the volume is corrected at every timestep
  from scipy.optimize import line_search
  import numpy as np
  t = tInit
  while t <= tInit+nSteps:
    print('t',t)
    if t%10==0: save_vert_positions(t, HC)
    t+=1
    if minEdge>0: remesh(HC,minEdge,maxEdge)
    posArray = np.array([x for v in HC.V for x in v.x])
    args=(HC, bV, t, params)
    gradArray = grad_energy(posArray, *args)
    #E0=get_energy_from_array(posArray, *args)
    #E1=get_energy_from_array(posArray-1e-10*gradArray, *args)
    #if E0<E1: 
    #  gradArray *= -1
    #  E1=get_energy_from_array(posArray-1e-10*gradArray, *args)
    #  print('E0-E1',E0-E1)
    ret = line_search(get_energy_from_array, grad_energy, posArray, -gradArray, args=args)#, amax=.1*minEdge)
    if ret[0] == None: alpha = stepSize
    else: 
      alpha = ret[0]
      print('alpha',alpha)
    for i, v in enumerate(HC.V):
      HC.V.move(v, tuple(posArray[3*i:3*(i+1)] - alpha*gradArray[3*i:3*(i+1)]))
    if implicitVolume: correct_the_volume(HC, bV, params['initial_volume'])
  return t
 
