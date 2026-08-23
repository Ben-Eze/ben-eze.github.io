def advectionJos(D, u, v, dx, dy, IX, IY, dt):
    """
    Advect scalar field D in accordance with the velocity field (u, v).
    D can be the u, v components of the velocity field themselves, but it can 
    also be used for smoke visualisations, where D is the smoke density.
    """

    # get the (index) coordinates where we are advecting D from
    IX_prev = (IX - u * dt / dx)
    IY_prev = (IY - v * dt / dy)

    ny, nx = IX.shape
    
    # get the x and y coords of the corners of the cell we are advecting from
    # clipping is necessary to avoid indexing outside the domain
    x0 = np.clip(np.floor(IX_prev), 0, nx-1).astype(int)
    y0 = np.clip(np.floor(IY_prev), 0, ny-1).astype(int)
    x1 = np.clip(x0 + 1, 0, nx-1)
    y1 = np.clip(y0 + 1, 0, ny-1)
    
    # get the fractional part of the coordinates
    frac_x = IX_prev%1
    frac_y = IY_prev%1

    # get D at the corners of the cell we are advecting from
    D00 = D[y0, x0]
    D01 = D[y0, x1]
    D10 = D[y1, x0]
    D11 = D[y1, x1]

    # perform bilinear interpolation
    D0f = (1-frac_x)*D00 + frac_x*D01
    D1f = (1-frac_x)*D10 + frac_x*D11
    Dff = D.copy()
    Dff = ((1-frac_y)*D0f + frac_y*D1f)

    return Dff