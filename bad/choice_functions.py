'''
Choice functions.

We focus upon the cumulative normal
distribution as the choice function, but you could impliment your own, eg
Logistic, etc.
'''


from scipy.stats import norm
import numpy as np


def StandardCumulativeNormalChoiceFunc(decision_variable, θ, θ_fixed):
    '''Cumulative normal choice function, but no alpha parameter'''
    p_chose_B = θ_fixed['ϵ'] + (1 - 2 * θ_fixed['ϵ']) * _Phi(decision_variable)
    return p_chose_B


def CumulativeNormalChoiceFunc(decision_variable, θ, θ_fixed):
    '''Our default choice function'''
    α = θ['α'].values
    p_chose_B = (θ_fixed['ϵ'] + (1 - 2 * θ_fixed['ϵ'])
                 * _Phi(np.divide(decision_variable, α)))
    return p_chose_B


def _Phi(x):
    '''Cumulative normal distribution, provided here as a helper function'''
    # NOTE: because some of the data was from a pandas dataframe, the numpy
    # arrays are coming out as dtype = object. So we need to cooerce into
    # floats.
    return norm.cdf(x.astype('float'), loc=0, scale=1)
