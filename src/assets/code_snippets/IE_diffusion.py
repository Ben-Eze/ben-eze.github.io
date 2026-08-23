def diffuse_implicitEuler(d, nu, dx, dt, nit):
    """
    Diffuse the scalar field D with the Implicit Euler scheme
    Simplification: dx = dy
    """

    k = nu * dt / dx**2
    D = d.copy()            # initial guess 

    # iteratively progress D to satisfy the equation 
    for _ in range(nit):
        D[1:-1, 1:-1] = ((d[1:-1, 1:-1] + k*(
            D[2:, 1:-1] + D[:-2, 1:-1] + D[1:-1, 2:] + D[1:-1, :-2])) 
            / (1 + 4*k))
    return D