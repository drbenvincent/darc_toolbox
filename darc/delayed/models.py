from scipy.stats import norm, bernoulli, halfnorm
import numpy as np
from bad.base_classes import Model


# The classes in this file are domain specific, and therefore include specifics
# about the design space and the model parameters.

# The main jobs of the model classes are:
# a) define priors over parameters - as scipy distribution objects
# b) implement the `calc_decision_variable` method


class Hyperbolic(Model):
    '''Hyperbolic time discounting model'''

    prior = dict()
    prior['logk'] = norm(loc=np.log(1/365), scale=2)
    prior['α'] = halfnorm(loc=0, scale=3)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        VA = data['RA'].values * self._time_discount_func(data['DA'].values, θ)
        VB = data['RB'].values * self._time_discount_func(data['DB'].values, θ)
        return VB-VA
    
    @staticmethod
    def _time_discount_func(delay, θ):
        k = np.exp(θ['logk'].values)
        return np.divide(1, (1 + k * delay))
        

class Exponential(Model):
    '''Exponential time discounting model'''

    prior = dict()
    prior['k'] = norm(loc=0.01, scale=0.1)
    prior['α'] = halfnorm(loc=0, scale=3)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        VA = data['RA'].values * self._time_discount_func(data['DA'].values, θ)
        VB = data['RB'].values * self._time_discount_func(data['DB'].values, θ)
        return VB - VA
    
    @staticmethod
    def _time_discount_func(delay, θ):
        # NOTE: we want k as a row matrix, and delays as a column matrix to do the
        # appropriate array broadcasting.
        k = θ['k'][:, np.newaxis]
        delay = delay[np.newaxis, :]
        return np.exp((-k * delay).astype('float'))


class HyperbolicMagnitudeEffect(Model):
    '''Hyperbolic time discounting model + magnitude effect
    Vincent, B. T. (2016). Hierarchical Bayesian estimation and hypothesis testing for
    delay discounting tasks. Behavior Research Methods, 48(4), 1608–1620.
    http://doi.org/10.3758/s13428-015-0672-2
    '''

    prior = dict()
    prior['m'] = norm(loc=-2.43, scale=2)
    prior['c'] = norm(loc=0, scale=100) # <------ TODO: improve this
    prior['α'] = halfnorm(loc=0, scale=3)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        VA = self._present_subjective_value(
            data['RA'].values, data['DA'].values, θ)
        VB = self._present_subjective_value(
            data['RB'].values, data['DA'].values, θ)
        return VB-VA

    @staticmethod
    def _present_subjective_value(reward, delay, θ):
        # process inputs
        delay = delay[np.newaxis, :].astype('float')
        reward = reward[np.newaxis, :].astype('float')
        m = θ['m'][:,np.newaxis]
        c = θ['c'][:, np.newaxis]
        # magnitude effect
        k = np.exp( m * np.log(reward) + c )
        # HYPERBOLIC discounting of time
        discount_fraction = 1 / (1 + k * delay)
        # subjective value function
        V = reward * discount_fraction
        return V


class ExponentialMagnitudeEffect(Model):
    '''Exponential time discounting model + magnitude effect
    Similar to...
    Vincent, B. T. (2016). Hierarchical Bayesian estimation and hypothesis testing for
    delay discounting tasks. Behavior Research Methods, 48(4), 1608–1620.
    http://doi.org/10.3758/s13428-015-0672-2
    '''

    prior = dict()
    prior['m'] = norm(loc=-2.43, scale=2) # <---- maybe need to update
    prior['c'] = norm(loc=0, scale=100)  # <------ TODO: improve this
    prior['α'] = halfnorm(loc=0, scale=3)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        VA = self._present_subjective_value(
            data['RA'].values, data['DA'].values, θ)
        VB = self._present_subjective_value(
            data['RB'].values, data['DB'].values, θ)
        return VB-VA

    @staticmethod
    def _present_subjective_value(reward, delay, θ):
        # process inputs
        delay = delay[np.newaxis, :].astype('float')
        reward = reward[np.newaxis, :].astype('float')
        m = θ['m'][:, np.newaxis]
        c = θ['c'][:, np.newaxis]
        # magnitude effect
        k = np.exp(m * np.log(reward) + c)
        # EXPONENTIAL discounting of time
        discount_fraction = np.exp((-k * delay)) #.astype('float'))
        # subjective value function
        V = reward * discount_fraction
        return V


class ConstantSensitivity(Model):
    '''The constant sensitivity time discounting model

    Ebert & Prelec (2007) The Fragility of Time: Time-Insensitivity and Valuation 
    of the Near and Far Future. Management Science, 53(9):1423–1438.
    '''

    prior = dict()
    prior['a'] = norm(loc=0.01, scale=0.1)
    prior['b'] = halfnorm(loc=0.001, scale=3) # TODO: Improve this prior! make it centered on 1, maybe lognormal
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        VA = data['RA'].values * self._time_discount_func(data['DA'].values, θ)
        VB = data['RB'].values * self._time_discount_func(data['DB'].values, θ)
        return VB-VA

    @ staticmethod
    def _time_discount_func(delay, θ):
        # NOTE: we want params as a row matrix, and delays as a column matrix to do the
        # appropriate array broadcasting.
        a = θ['a'][:, np.newaxis]
        b = θ['b'][:, np.newaxis]
        delay = delay[np.newaxis, :]
        temp = (a * delay)**b
        return np.exp(-temp.astype('float'))


class MyersonHyperboloid(Model):
    '''Myerson style hyperboloid
    '''

    prior = dict()
    prior['logk'] = norm(loc=np.log(1 / 365), scale=2)
    prior['s'] = halfnorm(loc=0, scale=2)
    prior['α'] = halfnorm(loc=0, scale=3)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        VA = data['RA'].values * self._time_discount_func(data['DA'].values, θ)
        VB = data['RB'].values * self._time_discount_func(data['DB'].values, θ)
        return VB-VA

    @staticmethod
    def _time_discount_func(delay, θ):
        # NOTE: we want k as a row matrix, and delays as a column matrix to do the
        # appropriate array broadcasting.
        k = np.exp(θ['logk'])
        s = θ['s'][:, np.newaxis]
        delay = delay[np.newaxis, :]
        return 1 / (1 + k[:, np.newaxis] * delay)**s


class ProportionalDifference(Model):
    '''Proportional difference model'''

    prior = dict()
    prior['δ'] = norm(loc=0, scale=10)
    prior['α'] = halfnorm(loc=0, scale=3)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        # organised so that higher values of the decision variable will 
        # mean higher probabability for the delayed option (prospect B)
        
        prop_reward = self._proportion(
            data['RA'].values, data['RB'].values)

        prop_delay = self._proportion(
            data['DA'].values, data['DB'].values)
        
        prop_difference = (prop_reward - prop_delay)[np.newaxis, :]
        decision_axis = prop_difference + θ['δ'][:, np.newaxis]
        return decision_axis

    @staticmethod
    def _max_abs(x, y):
        return np.max(np.array([np.absolute(x), np.absolute(y)]).astype('float'), axis=0).T

    @staticmethod
    def _min_abs(x, y):
        return np.min(np.array([np.absolute(x), np.absolute(y)]).astype('float'), axis=0).T

    def _proportion(self, x, y):
        diff = self._max_abs(x, y) - self._min_abs(x, y)
        return diff / self._max_abs(x, y)


class HyperbolicNonLinearUtility(Model):
    '''Hyperbolic time discounting + non-linear utility model.
    The a-model from ...
    Cheng, J., & González-Vallejo, C. (2014). Hyperbolic Discounting: Value and
    Time Processes of Substance Abusers and Non-Clinical Individuals in Intertemporal
    Choice. PLoS ONE, 9(11), e111378–18. http://doi.org/10.1371/journal.pone.0111378
    '''

    prior = dict()
    prior['a'] = norm(loc=1, scale=0.1) # TODO: must be positive!
    prior['logk'] = norm(loc=np.log(1/365), scale=2)
    prior['α'] = halfnorm(loc=0, scale=3)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        a = np.exp(θ['a'])
        VA = np.power(data['RA'].values,a) * self._time_discount_func(data['DA'].values, θ)
        VB = np.power(data['RB'].values,a) * self._time_discount_func(data['DB'].values, θ)
        return VB-VA

    @staticmethod
    def _time_discount_func(delay, θ):
        # NOTE: we want k as a row matrix, and delays as a column matrix to do the
        # appropriate array broadcasting.
        k = np.exp(θ['logk'])
        delay = delay[np.newaxis, :]
        return 1 / (1 + k[:, np.newaxis] * delay)
