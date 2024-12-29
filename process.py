import numpy as np
from constants import A,C,H,T
import pandas as pd
import math

def next_u_lin(u_prev, n = 1):
    noise = np.random.normal(0, C ** 0.5, n)
    return A * u_prev + noise

def curr_y_lin(u, n = 1):
    noise = np.random.normal(0, T ** 0.5, n)
    return H * u + noise

def next_u_nonlin(u_prev, n = 1, phi = 0.98, var = 0.16):
    return np.random.normal(phi * u_prev, var, n)

def curr_y_nonlin(u, n = 1, beta_sq = 0.7):
    return np.random.normal(0, beta_sq * np.exp(u) , n)

def gen_process(n = 2000, linear = True):

    # Initial State
    u_0 = np.random.normal(0,1,1)

    # Initialize storage
    u = []
    y = []

    for i in range(n):
        u_prev = u[i-1] if i>0 else u_0

        u_next = next_u_lin(u_prev) if linear else next_u_nonlin(u_prev)
        y_meas = curr_y_lin(u_next) if linear else curr_y_nonlin(u_next)
        y.append(y_meas)
        u.append(u_next)

    iters = [i+1 for i in range(len(y))]

    return iters, u, y
