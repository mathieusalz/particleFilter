import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod
from process import curr_y_lin, next_u_lin, curr_y_nonlin, next_u_nonlin
from constants import A,C,H,T

class Filter():

    def __init__(self, n = 1, linear = True):
        self.n = n
        self.parts = np.random.normal(0, 1, self.n)
        self.weights = np.ones(self.n)/self.n
        self.u_est_hist = []
        self.u_est = None
        self.expect = 0
        self.iters = 0
        self.curr_y = curr_y_lin if linear else curr_y_nonlin
        self.next_u = next_u_lin if linear else next_u_nonlin
        self.linear = linear

    @abstractmethod
    def predict(self):
        pass

    def update_expect(self, u_estimates, u_true):
        if u_true is not None:
            self.expect += (self.weights * (u_estimates - u_true)**2).sum()

    def RMSE(self):
        return (self.expect/self.iters) ** 0.5
    
    def update(self, new_u_est):
        self.iters += 1
        self.u_est_hist.append(new_u_est)
        self.u_est = new_u_est
    
    @abstractmethod
    def iterate(self, y_meas, u_true):
        pass

class Kalman(Filter):

    def __init__(self):
        super().__init__()

        self.sigma_a = 1
        self.sigma_p = 0
        self.mu_a = self.parts
        self.mu_p = None
        self.sigma_hist = []

    def predict(self):
        self.mu_p = next_u_lin(self.mu_a)
        self.sigma_p = A * self.sigma_a * A + C 
    
    def analyse(self, y_meas, u_true):
        d = y_meas - H * self.mu_p
        S = H * self.sigma_p * H + T
        K = self.sigma_p * H * (1/S)

        mu_a = self.mu_p + K * d
        self.sigma_a = (1 - K *H) * self.sigma_p

        self.update(mu_a)
        self.update_expect(mu_a, u_true)

    def update(self, mu_a):
        super().update(mu_a)
        self.sigma_hist.append(self.sigma_a)

    def iterate(self, y_meas, u_true = None):

        self.predict()
        self.analyse(y_meas, u_true)


class Naive(Filter):

    def __init__(self):
        super().__init__()

    def predict(self, y_meas, u_true = False):
        u_estimate = y_meas/H

        self.update_expect(u_estimate, u_true)
        self.update(u_estimate)
        
        return y_meas/H
    
    def iterate(self, y_meas, u_true):
        self.predict(y_meas, u_true)

class Kalman_Ensemble(Filter):

    def __init__(self, n, linear = True):
        super().__init__(n, linear)
        self.u_est = np.array(self.parts).mean()

    def predict(self):
        self.parts = self.next_u(self.parts, self.n)

    def analyse(self, y_measure, u_true):
        y_meas = self.curr_y(self.parts, self.n)
        u_preds = self.parts
        
        I = np.identity(self.n)
        sub_matrix = I - 1/self.n * np.ones((self.n, self.n))
        
        Y = np.array(y_meas).reshape(1,self.n) @ sub_matrix
        U = np.array(u_preds).reshape(1,self.n) @ sub_matrix

        K = ((U @ Y.T) / (Y @ Y.T)).flatten()

        u_estimates = []

        u_estimates = self.parts + K * (y_measure - y_meas)
        self.parts = u_estimates

        new_u_est = np.array(u_estimates).mean()

        if self.linear:
            self.update_expect(u_estimates, u_true)
        self.update(new_u_est)
        return new_u_est


    def iterate(self, y_meas, u_true = None):
        self.predict()
        self.analyse(y_meas, u_true)


class Bootstrap_PT(Filter):
    def __init__(self, n, linear = True, phi = 0.98, var = 0.16, beta = 0.7, no_resampling = False):
        super().__init__(n, linear)
        self.ESS_hist = []
        self.beta = beta
        self.var = var
        self.phi = phi
        self.loglikehood = 0
        self.likelihood = 1
        self.no_resampling = no_resampling

    def resample(self):

        if self.iters > 0:
            self.parts = np.random.choice(self.parts, size=self.n, p=self.weights)

        self.ESS_hist.append(1 / (self.n * (self.weights**2).sum()))

    def predict(self):
        self.parts = self.next_u(self.parts, self.n, self.phi, self.var)

    def analyse(self, y_meas, u_true):
        estimates = self.parts

        if self.linear:
            likelihoods = norm(H * estimates, T ** 0.5).pdf(y_meas)
        else:
            likelihoods = norm(0, (np.exp(estimates)**0.5) * self.beta).pdf(y_meas)
        like_exp = np.exp(likelihoods)
        self.weights = like_exp / np.sum(like_exp)

        u_est_b = np.dot(self.weights, estimates)
        
        self.update(u_est_b)

        if self.linear:
            self.update_expect(estimates, u_true)
        else:
            self.update_loglikelihood(likelihoods)

        return u_est_b

    def iterate(self,y_meas, u_true = None):

        if self.iters > 0 and self.no_resampling == False:
            self.resample()
        self.predict()
        self.analyse(y_meas, u_true)

    def update_loglikelihood(self, likelihoods):
        self.loglikehood += np.log(likelihoods.sum() * 1/self.n)
        self.likelihood *= likelihoods.sum() * 1/self.n
