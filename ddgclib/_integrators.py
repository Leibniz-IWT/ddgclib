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
  
def NowtonRhapson(HC, bV, params, t, nSteps, stepSize, minEdge=-1, maxEdge=-1, implicitVolume=False, constMoveLen=False):
  forcePrev = {}
  posPrev = {}
  x_new = 0
  tInit=t
  while t <= tInit+nSteps:
    print('t',t)
    if t%10==0: save_vert_positions(t, HC)
    t+=1
    if minEdge>0: remesh(HC,minEdge,maxEdge)
    forceDict, maxForce = get_forces(HC, bV, t, params)
    for v in HC.V:
      if v.x in forceDict:
        HC.V.move(v, tuple(v.x_a + stepSize*forceDict[v.x]/maxForce))
    if implicitVolume: correct_the_volume(HC, bV, params['initial_volume'])
  while t<=tInit+300:
    if t%10==0: save_vert_positions(t)
    E_0 = get_energy(HC)
    print('t',t,'nVerts',len(list(HC.V)),'maxMove',maxMove)
    t+=1
    if constMove:
      HC.V.merge_nn(cdist=minEdge, exclude=bV)
      refine_edges(HC, maxEdge)
      refine_boundaries(HC, bV, maxEdge)
      reconnect_long_diagonals(HC, bV)
      for v in HC.V:
        if len(v.nn)<2:  
          print('v.nn',v.nn)
          print('remove',v.x_a)
          HC.V.remove(v)
      #if t> 20 and sum(prevE)/len(prevE) < sum(pastE)/len(pastE): constMove=False
      if False:#sum(prevE)/len(prevE) < sum(pastE)/len(pastE): 
        maxMove*=.5
        E_0=np.nan
      pastE.append(E_0)
      prevE.append(pastE.pop(0))
      prevE.pop(0)
    forceDict, maxForce = get_forces(HC, bV)
    if constMove: alpha = maxMove/maxForce
    else: 
      alpha = 1e-1#/gamma
      maxMove = alpha*maxForce
    for v in HC.V:
      if v.x in forceDict:
        normFor = alpha*forceDict[v.x]
        HC.V.move(v, tuple(v.x_a + normFor))
    #for v in HC.V:
    #  if timeInt=='NewtonRapson' and v.x in forcePrev:
    #    #f_k = v.x_a - force * (v.x_a - posPrev[v.x]) / (force - forcePrev[v.x]) 
    #    numer=0
    #    denom=0
    #    for i in range(3):
    #      numer += forceDict[v.x][i] * (v.x_a[i] - posPrev[v.x][i]) 
    #      denom += forceDict[v.x][i] * (forceDict[v.x][i] - forcePrev[v.x][i]) 
    #    if np.linalg.norm(forceDict[v.x] * numer / denom) > maxMove: x_new = -1
    #    else: x_new = uple(v.x_a - forceDict[v.x] * numer / denom)
    #  if v.x in forceDict and (x_new==-1 or timeInt=='adaptiveEuler'):
    #    normFor = alpha*forceDict[v.x]
    #    x_new = tuple(v.x_a + normFor)
    #  else: continue
    #  posPrev[x_new] = v.x_a
    #  forcePrev[x_new] = forceDict[v.x]
    #  HC.V.move(v, tuple(x_new))
    #E_0 = get_energy(HC)
    if t%10==0: save_vert_positions(str(t)+'uncor')
    #if constMove: correct_the_volume(HC, bV)
