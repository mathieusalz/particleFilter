import numpy as np

def next_u(u_prev):
    noise = np.random.normal(0, 0.5, 1)
    return 0.9 * u_prev + noise

def curr_y(u):
    noise = np.random.normal(0, 0.1, 1)
    return 1.3 * u + noise

def gen_process(n = 2000):

    # Initial State
    u_0 = np.random.normal(0,1,1)

    # Initialize storage
    u = []
    y = []

    for i in range(n):
        u_prev = u[i] if i>0 else u_0
        u.append(next_u(u_prev))
        y.append(curr_y(u_prev))

    y.append(curr_y(u[-1]))

    iters = [i for i in range(len(y))]

    return iters, u, y
