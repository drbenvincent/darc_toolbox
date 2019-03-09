'''
Choice functions.

We focus upon the cumulative normal
distribution as the choice function, but you could impliment your own, eg
Logistic, etc.

These are used by concrete model classes. More specifically we are going to
bind the choice functions defined here to a model. Therefore we need to declare
these as static methods, hence the @staticmethod decorator. See
https://stackoverflow.com/questions/14645353/dynamically-added-static-methods-to-a-python-class
for more information. An alternative is to put an additional self argument as
an input to the choice function, but this is messy/clunky.
'''


from scipy.stats import norm
import numpy as np


@staticmethod
def StandardCumulativeNormalChoiceFunc(decision_variable, θ, θ_fixed):
    '''Cumulative normal choice function, but no alpha parameter'''
    p_chose_B = θ_fixed['ϵ'] + (1 - 2 * θ_fixed['ϵ']) * _Phi(decision_variable)
    return p_chose_B


@staticmethod
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
