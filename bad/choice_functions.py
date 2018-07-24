'''
Choice functions.

I will only provide the one choice function here, the cumulative normal with a fixed
baseline error rate.

These are being used by models. More specifically we are going to bind the choice
functions defined here to a model. Therefore we need to declare these as static methods,
hence the @staticmethod decorator. See https://stackoverflow.com/questions/14645353/dynamically-added-static-methods-to-a-python-class
for more information. An alternative is to put an additional self argument as an input
to the choice function, but this is messy/clunky.
'''


from scipy.stats import norm
import numpy as np


@staticmethod
def CumulativeNormalChoiceFunc(decision_variable, θ, θ_fixed):
    α = θ['α'][:, np.newaxis]
    return θ_fixed['ϵ'] + (1 - 2 * θ_fixed['ϵ']) * _Phi(decision_variable / α)


def _Phi(x):
    '''Cumulative normal distribution, provided here as a helper function'''
    # NOTE: because some of the data was from a padas dataframe, the numpy
    # arrays are coming out as dtype = object. So we need to cooerce into
    # floats. This seems like a lot of pain in the arse.
    return norm.cdf(x.astype('float'), loc=0, scale=1)


