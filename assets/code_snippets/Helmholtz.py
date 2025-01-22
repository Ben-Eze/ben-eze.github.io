def extract_divfree(u, v, phi, dx, dy, nit, div):
    div_v = div(u, v)

    # iterate to find phi
    for _ in range(nit):
        phi[1:-1, 1:-1] = ((phi[1:-1, 2:] + phi[1:-1, :-2]) * dy**2
                        + (phi[2:, 1:-1] + phi[:-2, 1:-1]) * dx**2
                        - dx**2 * dy**2 * div_v) / (2 * (dy**2 + dx**2))
    
    # curl-free component
    u_cf = (phi[1:-1, 2:] - phi[1:-1, :-2]) / (2 * dx)
    v_cf = (phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2 * dy)

    # divergence-free component
    u_df = u.copy()
    v_df = v.copy()
    u_df[1:-1, 1:-1] -= u_cf
    v_df[1:-1, 1:-1] -= v_cf

    return u_df, v_df, phi