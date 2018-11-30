'''The classes in this file are domain specific, and therefore include specifics
about the design space and the model parameters.

The main jobs of the model classes are:
a) define priors over parameters - as scipy distribution objects
b) implement the `calc_decision_variable` method. You can add
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


class DelaySlice(Model):
    '''This is an insane delay discounting model. It basically fits ONE indifference
    point. It amounts to fitting a psychometric function with the indifference point
    shifting the function and alpha determining the slope of the function.

    Note: the α parameter in this model is on a different scale to the same parameter
    in other models. Here we are doing inference over indifference points, so the whole
    range typically spans 0-1. So it makes sense for this model that our prior over
    α is more restricted to low values near zero
    '''

    prior = dict()
    prior['indiff'] = uniform(0, 1)
    prior['α'] = halfnorm(loc=0, scale=0.1)
    θ_fixed = {'ϵ': 0.001}

    def calc_decision_variable(self, θ, data):
        ''' The decision variable is difference between the indifference point and
        the 'stimulus intensity' which is RA/RB '''
        return θ['indiff'].values - (data['RA'].values / data['RB'].values)


# class DelaySlices(Model):
#     '''This is a 'non-parametric' model which estimates indifference points (A/B)
#     at a small number of specified delays. The parameters being inferred are the
#     indifference points for each delay level.
#     '''

#     prior = dict()
#     prior['indiff1'] = uniform(loc=0, scale=1)
#     prior['indiff2'] = uniform(loc=0, scale=1)
#     prior['indiff3'] = uniform(loc=0, scale=1)
#     prior['indiff4'] = uniform(loc=0, scale=1)
#     prior['α'] = halfnorm(loc=0, scale=2)
#     θ_fixed = {'ϵ': 0.01}

#     @staticmethod
#     def my_func(DB, RA, RB,  delays, θ):
#         if DB == delays[0]:
#             x = θ['indiff1'].values - (RA/RB)
#         elif DB == delays[1]:
#             x = θ['indiff2'].values - (RA/RB)
#         elif DB == delays[2]:
#             x = θ['indiff3'].values - (RA/RB)
#         elif DB == delays[3]:
#             x = θ['indiff4'].values - (RA/RB)
#         return x


#     def calc_decision_variable(self, θ, data):
#         ''' The decision variable is difference between the indifference point and
#         the 'stimulus intensity' which is RA/RB '''

#         v_my_func = np.vectorize(self.my_func, excluded=['delays', 'θ'])

#         x = v_my_func(data['DB'].values, data['RA'].values, data['RA'].values,  self.delays, θ)
#         return x
#         # print(self.delays[0])
#         # print(data['DB'].values)
#         # TODO: FIX THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         # if data['DB'].values == self.delays[0]:
#         #     x = θ['indiff1'].values - (data['RA'].values / data['RB'].values)
#         # elif data['DB'].values == self.delays[1]:
#         #     x = θ['indiff2'].values - (data['RA'].values / data['RB'].values)
#         # elif data['DB'].values == self.delays[2]:
#         #     x = θ['indiff3'].values - (data['RA'].values / data['RB'].values)
#         # elif data['DB'].values == self.delays[3]:
#         #     x = θ['indiff4'].values - (data['RA'].values / data['RB'].values)
#         # return x


class Hyperbolic(Model):
    '''Hyperbolic time discounting model

    Mazur, J. E. (1987). An adjusting procedure for studying delayed
    re-inforcement. In Commons, M. L., Mazur, J. E., Nevin, J. A., and
    Rachlin, H., editors, Quantitative Analyses of Behavior, pages 55–
    73. Erlbaum, Hillsdale, NJ.
    '''

    prior = dict()
    prior['logk'] = norm(loc=-4.5, scale=1)
    prior['α'] = halfnorm(loc=0, scale=2)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        VA = data['RA'].values * self._time_discount_func(data['DA'].values, np.exp(θ['logk'].values))
        VB = data['RB'].values * self._time_discount_func(data['DB'].values, np.exp(θ['logk'].values))
        return VB - VA

    @staticmethod
    def _time_discount_func(delay, k):
        return 1/(1 + k * delay)


class Exponential(Model):
    '''Exponential time discounting model'''

    prior = dict()
    prior['k'] = norm(loc=0.01, scale=0.1)
    prior['α'] = halfnorm(loc=0, scale=3)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        VA = data['RA'].values * self._time_discount_func(data['DA'].values, θ['k'].values)
        VB = data['RB'].values * self._time_discount_func(data['DB'].values, θ['k'].values)
        return VB - VA

    @staticmethod
    @np.vectorize
    def _time_discount_func(delay, k):
        return np.exp(-k * delay)

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
            data['RA'].values, data['DA'].values, θ['m'].values, θ['c'].values)
        VB = self._present_subjective_value(
            data['RB'].values, data['DB'].values, θ['m'].values, θ['c'].values)
        return VB-VA

    @staticmethod
    def _present_subjective_value(reward, delay, m, c):
        k = np.exp( m * np.log(reward) + c )
        discount_fraction = 1 / (1 + k * delay)
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
    prior['m'] = norm(loc=-2.43, scale=2) # <---- TODO: need to update
    prior['c'] = norm(loc=0, scale=100)  # <------ TODO: improve this
    prior['α'] = halfnorm(loc=0, scale=3)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        VA = self._present_subjective_value(
            data['RA'].values, data['DA'].values, θ['m'].values, θ['c'].values)
        VB = self._present_subjective_value(
            data['RB'].values, data['DB'].values, θ['m'].values, θ['c'].values)
        return VB-VA

    @staticmethod
    @np.vectorize
    def _present_subjective_value(reward, delay, m, c):
        k = np.exp(m * np.log(reward) + c)
        discount_fraction = np.exp(-k * delay)
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
    prior['α'] = halfnorm(loc=0, scale=3)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        VA = data['RA'].values * \
            self._time_discount_func(
                data['DA'].values, θ['a'].values, θ['b'].values)
        VB = data['RB'].values * self._time_discount_func(
                data['DB'].values, θ['a'].values, θ['b'].values)
        return VB-VA

    @ staticmethod
    def _time_discount_func(delay, a, b):
        # NOTE: we want params as a row matrix, and delays as a column matrix to do the appropriate array broadcasting.
        return np.exp(-np.power(a * delay, b))

class MyersonHyperboloid(Model):
    '''Myerson style hyperboloid
    '''

    prior = dict()
    prior['logk'] = norm(loc=np.log(1 / 365), scale=2)
    prior['s'] = halfnorm(loc=0, scale=2)
    prior['α'] = halfnorm(loc=0, scale=3)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        VA = data['RA'].values * self._time_discount_func(data['DA'].values, θ['logk'].values, θ['s'].values)
        VB = data['RB'].values * self._time_discount_func(data['DB'].values, θ['logk'].values, θ['s'].values)
        return VB-VA

    @staticmethod
    def _time_discount_func(delay, logk, s):
        # NOTE: we want logk as a row matrix, and delays as a column matrix to do the
        # appropriate array broadcasting.
        k = np.exp(logk)
        return 1 / np.power(1 + k * delay, s)


class ProportionalDifference(Model):
    '''Proportional difference model applied to delay discounting

    González-Vallejo, C. (2002). Making trade-offs: A probabilistic and
    context-sensitive model of choice behavior. Psychological Review, 109(1),
    137–155. http://doi.org/10.1037//0033-295X.109.1.137
    '''

    prior = dict()
    prior['δ'] = norm(loc=0, scale=10)
    prior['α'] = halfnorm(loc=0, scale=3)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        # organised so that higher values of the decision variable will
        # mean higher probabability for the delayed option (prospect B)
        prop_reward = self._proportional_difference(data['RA'].values, data['RB'].values)
        prop_delay = self._proportional_difference(data['DA'].values, data['DB'].values)
        decision_variable = prop_reward - prop_delay + θ['δ'].values
        return decision_variable

    @staticmethod
    def _max_abs(x, y):
        return np.max(np.array([np.absolute(x), np.absolute(y)]).astype('float'), axis=0).T

    @staticmethod
    def _min_abs(x, y):
        return np.min(np.array([np.absolute(x), np.absolute(y)]).astype('float'), axis=0).T

    def _proportional_difference(self, x, y):
        return (self._max_abs(x, y) - self._min_abs(x, y)) / self._max_abs(x, y)


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
        a = np.exp(θ['a'].values)
        VA = np.power(data['RA'].values,a) * self._time_discount_func(data['DA'].values, θ['logk'].values)
        VB = np.power(data['RB'].values,a) * self._time_discount_func(data['DB'].values, θ['logk'].values)
        return VB-VA

    @staticmethod
    def _time_discount_func(delay, logk):
        k = np.exp(logk)
        return 1/(1 + k * delay)
