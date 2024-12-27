import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod
from process import curr_y, next_u
from constants import A,C,H,T

class Filter():

    def __init__(self, n):
        self.n = n
        self.parts = np.random.normal(0, 1, self.n)
        self.weights = np.ones(self.n)/self.n
        self.u_est_hist = []
        self.u_est = None
        self.expect = 0
        self.iters = 0

    @abstractmethod
    def predict(self):
        pass

    def update_expect(self, u_estimates, u_true):
        if u_true is not None:
            self.expect += (self.weights * (u_estimates - u_true)**2).sum()

    def RMSE(self):
        return self.expect/self.iters
    
    def update(self, new_u_est):
        self.iters += 1
        self.u_est_hist.append(new_u_est)
        self.u_est = new_u_est
    
    @abstractmethod
    def iterate(self, y_meas, u_true):
        pass

class Kalman(Filter):

    def __init__(self):
        super().__init__(n = 1)

        self.var = 1
        self.var_pred = 0
        self.u_pred = None
        self.u_prev_est = self.parts

    def predict(self):
        u_pred = A * self.u_prev_est
        self.u_pred = u_pred

        self.var_pred = self.var * A + C 
        
    
    def analyse(self, y_meas, u_true):
        d = y_meas - H * self.u_pred
        S = H * self.var_pred * H + T
        K = self.var_pred * H * (1/S)

        u_pred_analysis = self.u_pred + K * d
        self.var = A * self.var_pred * A + C

        self.update(u_pred_analysis)
        self.update_expect(u_pred_analysis, u_true)
    
    def iterate(self, y_meas, u_true = None):

        self.predict()
        self.analyse(y_meas, u_true)


class Naive(Filter):

    def __init__(self):
        super().__init__(n = 1)

    def predict(self, y_meas, u_true):
        u_estimate = y_meas/H

        self.update_expect(u_estimate, u_true)
        self.update(u_estimate)
        
        return y_meas/H
    
    def iterate(self, y_meas, u_true):
        self.predict(y_meas, u_true)

class Kalman_Ensemble(Filter):

    def __init__(self, n):
        super().__init__(n)
        self.u_est = np.array(self.parts).mean()

    def predict(self):
        self.parts = next_u(self.parts, self.n)

    def analyse(self, y_measure, u_true):
        y_meas = curr_y(self.parts, self.n)
        u_preds = self.parts
        
        I = np.identity(self.n)
        sub_matrix = I - 1/self.n * np.ones((self.n, self.n))
        
        Y = np.array(y_meas).reshape(1,self.n) @ sub_matrix
        U = np.array(u_preds).reshape(1,self.n) @ sub_matrix

        K = ((U @ Y.T) / (Y @ Y.T)).flatten()

        u_estimates = []

        u_estimates = self.parts + K * y_measure - y_meas
        self.parts = u_estimates

        new_u_est = np.array(u_estimates).mean()

        self.update_expect(u_estimates, u_true)
        self.update(new_u_est)
        return new_u_est


    def iterate(self, y_meas, u_true = None):
        self.predict()
        self.analyse(y_meas, u_true)


class Bootstrap_PT(Filter):
    def __init__(self, n):
        super().__init__(n)
        self.ESS_hist = []

    def resample(self):

        if self.iters > 0:
            self.parts = np.random.choice(self.parts, size=self.n, p=self.weights)

        self.ESS_hist.append(1 / (self.n * (self.weights**2).sum()))

    def predict(self):
        self.parts = next_u(self.parts, self.n)

    def analyse(self, y_meas, u_true):
        estimates = self.parts
        likelihoods = norm(H * estimates, T).pdf(y_meas)
        self.weights = likelihoods / np.sum(likelihoods)

        u_est_b = np.dot(self.weights, estimates)
        
        self.update(u_est_b)
        self.update_expect(estimates, u_true)

        return u_est_b

    def iterate(self,y_meas, u_true = None):

        if self.iters > 0:
            self.resample()
        self.predict()
        self.analyse(y_meas, u_true)