def solve12steps():
    p = solve_for_p(u, v, p)        # continuity equation (iterative method)
                                    # NOTE: pressure BCs applied at each step
    U = solve_for_u(u, v, p)        # momentum equation, x-component
    v = solve_for_v(u, v, p)        # momentum equation, y-component
    u = U.copy()                    
    u, v, = velocity_BCs(u, v)      # boundary conditions 


def solveJosStam():
    u, v = diffuse_velocity(u, v)    # diffusion
    u, v = enforce_continuity(u, v)  # continuity (with Helmholtz decomposition)
    u, v = advect_velocity(u, v)     # advection
    u, v = enforce_continuity(u, v)  # as before
    u, v = velocity_BCs(u, v)        # boundary conditions