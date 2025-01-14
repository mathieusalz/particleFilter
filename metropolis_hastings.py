import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import uniform

from filters import Bootstrap_PT

from scipy.stats import invgamma

import pandas as pd


def metropolis(n_parts = 10, iterations = 50):

    data = pd.read_csv('synthetic_data.csv')
    y_measurements = data['y'].to_numpy()

    converged = False

    inverse_gamma_dist = invgamma(0.01, scale=0.01)

    rvs  = inverse_gamma_dist.rvs(size=2)
    var, beta = rvs[0], rvs[1]
    p_var, p_beta = inverse_gamma_dist.pdf(var), inverse_gamma_dist.pdf(beta)
    var_hist, beta_hist = [var], [beta]

    bootstrap = Bootstrap_PT(n = n_parts, linear = False, var = var, beta = beta)
    [bootstrap.iterate(y_measurements[i]) for i in range(500)]

    likelihood = bootstrap.likelihood
    prev_prob = likelihood * p_var * p_beta

    for j in range(iterations):

        next_var = np.abs(np.random.normal(var, 0.01, 1))
        next_beta = np.abs(np.nandom.normal(beta, 0.01, 1))
        p_var, p_beta = inverse_gamma_dist.pdf(next_var), inverse_gamma_dist.pdf(next_beta)
        
        new_bs = Bootstrap_PT(n = n_parts, linear = False, var = next_var, beta = next_beta)
        [new_bs.iterate(y_measurements[i]) for i in range(500)]

        likelihood = new_bs.likelihood
        new_prob = likelihood * p_var * p_beta

        accept = np.min(1, new_prob/prev_prob)

        if (accept - 1 < 1e-12):
            prev_prob = new_prob
            var = next_var
            beta = next_beta
            var_hist.append(var)
            beta_hist.append(beta)
        else:
            u = uniform()
            if u < accept:
                prev_prob = new_prob
                var = next_var
                beta = next_beta
                var_hist.append(var)
                beta_hist.append(beta)






