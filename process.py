import numpy as np
from constants import A,C,H,T


def next_u(u_prev, n = 1):
    noise = np.random.normal(0, C, n)
    return A * u_prev + noise

def curr_y(u, n = 1):
    noise = np.random.normal(0, T, n)
    return H * u + noise

def gen_process(n = 2000):

    # Initial State
    u_0 = np.random.normal(0,1,1)

    # Initialize storage
    u = []
    y = []

    for i in range(n):
        u_prev = u[i-1] if i>0 else u_0
        u.append(next_u(u_prev))
        y.append(curr_y(u_prev))

    iters = [i+1 for i in range(len(y))]

    return iters, u, y
