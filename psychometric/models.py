'''The classes in this file are domain specific, and therefore include specifics
about the design space and the model parameters.

The main jobs of the model classes are:
a) define priors over parameters - as scipy distribution objects
b) implement the `_calc_decision_variable` method. You can add
   whatever useful helper functions you wat in order to help with
   that job.

NOTE: There is some faff and checking required when we are doing
the numerical stuff. This might be my inexperience with Python, but
I think it comes down to annoyances in grabbing parameters and designs
out of a Pandas dataframe and getting that into useful Numpy arrays.
TODO: Can this be made easier/better?
'''


from scipy.stats import norm, bernoulli, halfnorm, uniform
import numpy as np
from bad.model import Model


def logistic(x, α, β):
    '''logistic function'''
    return 1 / (1 + np.exp(-β*(x-α)))  # TODO: impliment correctly


class Logistic(Model):
    '''Logistic function.

    FREE PARAMETERS
    β = slope
    α = shift along x axis

    FIXED PARAMETERS
    ϵ = epsilon = error or lapse rate. Typically around 0.01
    chance = chance performance
    '''

    # TODO: SORT OUT CORRECT PRIORS!!!
    prior = {'β': uniform(0, 1),
             'α': halfnorm(loc=0, scale=0.1)}
    θ_fixed = {'ϵ': 0.001,
               'chance': 0.5}

    def predictive_y(self, θ, data):
        '''
        chance + (1-chance-ϵ) * F(x, α, β)
        '''
        # TODO: impliment correctly
        return (self.θ_fixed['chance']+(1-self.θ_fixed['chance']-self.θ_fixed['ϵ'])
                * logistic(data['x'], self.θ['α'], self.θ['β']))
