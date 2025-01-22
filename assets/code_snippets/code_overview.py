initial_conditions()

# mainloop
for t in T:
    if method == "12-Steps":
        solve12steps(u, v, p)
    elif method == "Real-Time":
        solveJosStam(u, v)
    
    graphics()
    user_interface()