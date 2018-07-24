from scipy.stats import norm, bernoulli, halfnorm
import numpy as np
from bad.base_classes import Model


class Hyperbolic(Model):
    '''Hyperbolic risk discounting model
    The idea is that we hyperbolically discount ODDS AGAINST the reward'''

    prior = dict()
    prior['logh'] = norm(loc=np.log(1/365), scale=2)
    prior['α'] = halfnorm(loc=0, scale=3)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        # VA = data['RA'].values * self.time_discount_func(data['DA'].values, θ)
        # VB = data['RB'].values * self.time_discount_func(data['DB'].values, θ)
        return VB-VA
    
    @staticmethod
    def odds_discount_func(delay, θ):
        # # NOTE: we want k as a row matrix, and delays as a column matrix to do the
        # # appropriate array broadcasting.
        # k = np.exp(θ['logk'])
        # delay = delay[np.newaxis, :]
        # return 1 / (1 + k[:, np.newaxis] * delay)
        pass
        

class ProspectTheory(Model):
    '''Prospect Theory'''
    pass

