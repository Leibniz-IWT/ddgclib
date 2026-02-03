



def dudt(v, dim=3, mu=8.90 * 1e-4):
    # Equal to the acceleration at a vertex (RHS of equation)
    #dudt = -dP(v, dim=dim) + mu * du(v, dim=dim)
    dudt = 0
    dudt = dudt/v.m  # normalize by mass
    return dudt