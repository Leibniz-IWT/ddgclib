def Euler(HC, tInit, nSteps, alpha, minEdge=-1, maxEdge=-1, targetVolume=-1)
  for t in range(tInit,tInit+nSteps+1):
    print('t',t,'nVerts',len(list(HC.V)),'maxMove',maxMove)
    if t%10==0: save_vert_positions(t)
    if minEdge>0: remesh(HC,minEdge,maxEdge)
    forceDict, maxForce = get_forces(HC, bV)
    if constMove: alpha = maxMove/maxForce
    for v in HC.V:
      if v.x in forceDict:
        HC.V.move(v, tuple(v.x_a + alpha*forceDict[v.x]))
    if targetVolume>0: correct_the_volume(HC, bV, targetVolume)
