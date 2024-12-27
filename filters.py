import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod
from process import curr_y, next_u

class Filter():

    def __init__(self, n):
        self.n = n
        self.parts = np.random.normal(0, 1, self.n)
        self.weights = np.ones(self.n)/self.n
        self.expect = 0
        self.iters = 0

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def analyse(self):
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
        self.A = 0.9
        self.C = 0.5
        self.H = 1.3
        self.T = 0.1

        self.var = 1
        self.var_pred = 0
        self.u_pred = None
        self.u_prev_est = np.random.normal(0,1,1)

    def predict(self):
        u_pred = self.A * self.u_prev_est
        self.u_pred = u_pred

        self.var_pred = self.var * self.A + self.C 
        
    
    def analyse(self, y_meas):
        d = y_meas - self.H * self.u_pred
        S = self.H * self.var_pred * self.H + self.T
        K = self.var_pred * self.H * (1/S)

        u_pred_analysis = self.u_pred + K * d
        self.var = self.A * self.var_pred * self.A + self.C

        self.update(u_pred_analysis)
    
    def iterate(self, y_meas, u_true = None):

        self.predict()
        self.analyse(y_meas, u_true)


class Naive(Filter):

    def __init__(self):
        super.__init__(n = 1)
        self.H = 1.3

    def predict(self, y_meas, u_true):
        u_estimate = y_meas/self.H
        self.update_expectation(u_estimate, u_true)
        self.update(u_estimate)
        return y_meas/self.H
    
    def iterate(self, y_meas, u_true):
        self.predict(y_meas, u_true)

class Kalman_Ensemble(Filter):

    def __init__(self, n):
        super().__init__(n)
        self.n = n
        self.weights = np.ones(self.n)/self.n
        self.u_est = np.array(self.parts).mean()

    def predict(self):
        for i in range(self.n):
            self.parts[i].predict()

    def analyse(self, y_measure, u_true):
        y_meas  = [curr_y(self.parts[i].u_pred) for i in range(self.n)]
        u_preds = [self.parts[i].u_pred for i in range(self.n)]
        
        I = np.identity(self.n)
        sub_matrix = I - 1/self.n * np.ones((self.n, self.n))
        
        Y = np.array(y_meas).reshape(1,self.n) @ sub_matrix
        U = np.array(u_preds).reshape(1,self.n) @ sub_matrix

        K = ((U @ Y.T) / (Y @ Y.T))[0][0]

        u_estimates = []

        for i in range(self.n):
            u_est = (self.parts[i].u_pred + K * (y_measure - y_meas[i]))[0]
            self.parts[i].update(u_est)
            u_estimates.append(u_est)

        new_u_est = np.array(u_estimates).mean()

        self.update_expectation(u_estimates, u_true)

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

        if self.parts is None:
            self.parts = np.random.normal(0, 1, self.n)
            self.weights = np.ones(self.n)/self.n
        else:
            self.parts = np.random.choice(self.parts, size=self.n, p=self.weights)

        self.ESS_hist.append(1 / (self.n * (self.weights**2).sum()))

    def predict(self):
        noise = np.random.normal(0, 0.5, self.n)
        self.parts = self.parts * 0.9 + noise

    def analyse(self, y_meas, u_true):
        estimates = self.parts

        likelihoods = norm(1.3 * estimates, 0.1).pdf(y_meas)

        self.weights = likelihoods / np.sum(likelihoods)

        u_est_b = np.dot(self.weights, estimates)
        self.update(u_est_b)

        return u_est_b

    def iterate(self,y_meas, u_true = None):

        if self.iters > 0:
            self.resample()
        self.predict()
        self.analyse(y_meas, u_true)