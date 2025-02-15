from ddgclib._bubble import save_vert_positions, get_forces, correct_the_volume

def Euler(HC, bV, params, t, nSteps, stepSize, minEdge=-1, maxEdge=-1, implicitVolume=False, constMoveLen=False):
#Reduce the interface energy by an Eulerian method
#If implicitVolume, the volume is corrected at every timestep
#If constMoveLen, the step is adapted so that the maximum distance moved is equal to stepSize
  tInit=t
  while t <= tInit+nSteps:
    print('t',t)
    if t%10==0: save_vert_positions(t, HC)
    t+=1
    if minEdge>0: remesh(HC,minEdge,maxEdge)
    forceDict, maxForce = get_forces(HC, bV, t, params)
    if not constMoveLen: maxForce=1
    for v in HC.V:
      if v.x in forceDict:
        HC.V.move(v, tuple(v.x_a + stepSize*forceDict[v.x]/maxForce))
    if implicitVolume: correct_the_volume(HC, bV, params['initial_volume'])
  
def AdamsBashforth(HC, bV, params, t, nSteps, stepSize, minEdge=-1, maxEdge=-1, implicitVolume=False): 
#Reduce the interface energy by an Adams Bashforth method
#The first iteration is Eulerian
  tInit=t
  forcePrev = {}
  while t <= tInit+nSteps:
    print('t',t)
    if t%10==0: save_vert_positions(t, HC)
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
  
def NewtonRaphson(HC, bV, params, t, nSteps, stepSize, minEdge=-1, maxEdge=-1, implicitVolume=False): 
#Find minimum interface energy by nSteps applications of the Newton Raphson method
#stepSize is used for the first iteration, which is Eulerian
  tInit=t
  forcePrev = {}
  posPrev = {}
  while t <= tInit+nSteps:
    print('t',t)
    if t%10==0: save_vert_positions(t, HC)
    t+=1
    if minEdge>0: remesh(HC,minEdge,maxEdge)
    forceDict, maxForce = get_forces(HC, bV, t, params)
    for v in HC.V:
      if v.x in forcePrev:
        numer=0
        denom=0
        for i in range(3):
          numer += forceDict[v.x][i] * (v.x_a[i] - posPrev[v.x][i]) 
          denom += forceDict[v.x][i] * (forceDict[v.x][i] - forcePrev[v.x][i]) 
          x_new = tuple(v.x_a - forceDict[v.x] * numer / denom)
      elif v.x in forceDict: x_new = tuple(v.x_a + stepSize*forceDict[v.x])
      else: continue
      posPrev[x_new] = v.x_a
      forcePrev[x_new] = forceDict[v.x]
      HC.V.move(v, tuple(x_new))
    if implicitVolume: correct_the_volume(HC, bV, params['initial_volume'])

def LineSearch(HC, bV, params, t, nSteps, stepSize, minEdge=-1, maxEdge=-1, implicitVolume=False, constMoveLen=False):
#Reduce the interface energy by an Eulerian method
#If implicitVolume, the volume is corrected at every timestep
#If constMoveLen, the step is adapted so that the maximum distance moved is equal to stepSize
  tInit=t
  while t <= tInit+nSteps:
    print('t',t)
    if t%10==0: save_vert_positions(t, HC)
    t+=1
    if minEdge>0: remesh(HC,minEdge,maxEdge)
    forceDict, maxForce = get_forces(HC, bV, t, params)
    if not constMoveLen: maxForce=1
    for v in HC.V:
      if v.x in forceDict:
        HC.V.move(v, tuple(v.x_a + stepSize*forceDict[v.x]/maxForce))
    if implicitVolume: correct_the_volume(HC, bV, params['initial_volume'])
  
