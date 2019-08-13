'''The classes in this file are domain specific, and therefore include
specifics about the design space and the model parameters.

The main jobs of the model classes are:
a) define priors over parameters - as scipy distribution objects
b) implement the `predictive_y` method. You can add
   whatever useful helper functions you wat in order to help with
   that job.

NOTE: There is some faff and checking required when we are doing
the numerical stuff. This might be my inexperience with Python, but
I think it comes down to annoyances in grabbing parameters and designs
out of a Pandas dataframe and getting that into useful Numpy arrays.
TODO: Can this be made easier/better?
'''


from scipy.stats import norm, halfnorm, uniform
import numpy as np
from badapted.model import Model
from badapted.choice_functions import CumulativeNormalChoiceFunc, StandardCumulativeNormalChoiceFunc


class DelaySlice(Model):
    '''This is an insane delay discounting model. It basically fits ONE indifference
    point. It amounts to fitting a psychometric function with the indifference point
    shifting the function and alpha determining the slope of the function.

    Note: the α parameter in this model is on a different scale to the same parameter
    in other models. Here we are doing inference over indifference points, so the whole
    range typically spans 0-1. So it makes sense for this model that our prior over
    α is more restricted to low values near zero
    '''

    def __init__(self, n_particles,
                 prior={'indiff': uniform(0, 1),
                        'α': halfnorm(loc=0, scale=0.1)}):
        self.n_particles = int(n_particles)
        self.prior = prior
        self.θ_fixed = {'ϵ': 0.01}
        self.choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
        ''' The decision variable is difference between the indifference point and
        the 'stimulus intensity' which is RA/RB '''
        return θ['indiff'].values - (data['RA'].values / data['RB'].values)


class Hyperbolic(Model):
    '''Hyperbolic time discounting model

    Mazur, J. E. (1987). An adjusting procedure for studying delayed
    re-inforcement. In Commons, M. L., Mazur, J. E., Nevin, J. A., and
    Rachlin, H., editors, Quantitative Analyses of Behavior, pages 55–
    73. Erlbaum, Hillsdale, NJ.
    '''

    def __init__(self, n_particles,
                 prior={'logk': norm(loc=-4.5, scale=1),
                        'α': halfnorm(loc=0, scale=2)}):
        self.n_particles = int(n_particles)
        self.prior = prior
        self.θ_fixed = {'ϵ': 0.01}
        self.choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
        VA = (data['RA'].values
              * self._time_discount_func(data['DA'].values,
                                         np.exp(θ['logk'].values)))
        VB = (data['RB'].values
              * self._time_discount_func(data['DB'].values,
                                         np.exp(θ['logk'].values)))
        return VB - VA

    @staticmethod
    def _time_discount_func(delay, k):
        return 1/(1 + k * delay)


class Exponential(Model):
    '''Exponential time discounting model'''

    def __init__(self, n_particles,
                 prior={'k': norm(loc=0.01, scale=0.1),
                        'α': halfnorm(loc=0, scale=3)}):
        self.n_particles = int(n_particles)
        self.prior =prior
        self.θ_fixed = {'ϵ': 0.01}
        self.choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
        VA = data['RA'].values * self._time_discount_func(data['DA'].values,
                                                          θ['k'].values)
        VB = data['RB'].values * self._time_discount_func(data['DB'].values,
                                                          θ['k'].values)
        return VB - VA

    @staticmethod
    @np.vectorize
    def _time_discount_func(delay, k):
        return np.exp(-k * delay)


class HyperbolicMagnitudeEffect(Model):
    '''Hyperbolic time discounting model + magnitude effect

    Vincent, B. T. (2016). Hierarchical Bayesian estimation and hypothesis
    testing for delay discounting tasks. Behavior Research Methods, 48(4),
    1608–1620. http://doi.org/10.3758/s13428-015-0672-2
    '''

    def __init__(self, n_particles,
                 prior={'m': norm(loc=-2.43, scale=2),
                        'c': norm(loc=0, scale=100),
                        'α': halfnorm(loc=0, scale=3)}):
        self.n_particles = int(n_particles)
        self.prior = prior
        self.θ_fixed = {'ϵ': 0.01}
        self.choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
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
    Vincent, B. T. (2016). Hierarchical Bayesian estimation and hypothesis
    testing for delay discounting tasks. Behavior Research Methods, 48(4),
    1608–1620. http://doi.org/10.3758/s13428-015-0672-2
    '''

    def __init__(self, n_particles,
                 prior={'m': norm(loc=-2.43, scale=2),
                        'c': norm(loc=0, scale=100),
                        'α': halfnorm(loc=0, scale=3)}):
        self.n_particles = int(n_particles)
        self.prior = prior
        self.θ_fixed = {'ϵ': 0.01}
        self.choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
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

    def __init__(self, n_particles,
                 prior={'a': norm(loc=0.01, scale=0.1),
                        'b': halfnorm(loc=0.001, scale=3),
                        'α': halfnorm(loc=0, scale=3)}):
        self.n_particles = int(n_particles)
        self.prior = prior
        self.θ_fixed = {'ϵ': 0.01}
        self.choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
        VA = data['RA'].values * \
            self._time_discount_func(data['DA'].values,
                                     θ['a'].values,
                                     θ['b'].values)
        VB = data['RB'].values * self._time_discount_func(data['DB'].values,
                                                          θ['a'].values,
                                                          θ['b'].values)
        return VB-VA

    @ staticmethod
    def _time_discount_func(delay, a, b):
        # NOTE: we want params as a row matrix, and delays as a column matrix
        # to do the appropriate array broadcasting.
        return np.exp(-np.power(a * delay, b))


class MyersonHyperboloid(Model):
    '''Myerson style hyperboloid
    '''

    def __init__(self, n_particles,
                 prior={'logk': norm(loc=np.log(1 / 365), scale=2),
                        's': halfnorm(loc=0, scale=2),
                        'α': halfnorm(loc=0, scale=3)}):
        self.n_particles = int(n_particles)
        self.prior = prior
        self.θ_fixed = {'ϵ': 0.01}
        self.choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
        VA = data['RA'].values * self._time_discount_func(data['DA'].values,
                                                          θ['logk'].values,
                                                          θ['s'].values)
        VB = data['RB'].values * self._time_discount_func(data['DB'].values,
                                                          θ['logk'].values,
                                                          θ['s'].values)
        return VB-VA

    @staticmethod
    def _time_discount_func(delay, logk, s):
        # NOTE: we want logk as a row matrix, and delays as a column matrix to
        # do the appropriate array broadcasting.
        k = np.exp(logk)
        return 1 / np.power(1 + k * delay, s)


class ModifiedRachlin(Model):
    '''The Rachlin (2006) discount function, modified by Vincent &
    Stewart (2018). This has a better parameterisation.

    Rachlin, H. (2006, May). Notes on Discounting. Journal of the
        Experimental Analysis of Behavior, 85(3), 425–435.
    Vincent, B. T., & Stewart, N. (2018, October 16). The case of muddled
        units in temporal discounting.
        https://doi.org/10.31234/osf.io/29sgd
    '''

    def __init__(self, n_particles,
                 prior={'logk': norm(loc=np.log(1 / 365), scale=2),
                        's': halfnorm(loc=1, scale=2),
                        'α': halfnorm(loc=0, scale=3)}):
        self.n_particles = int(n_particles)
        self.prior = prior
        self.θ_fixed = {'ϵ': 0.01}
        self.choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
        VA = data['RA'].values * self._time_discount_func(
            data['DA'].values, θ['logk'].values, θ['s'].values)
        VB = data['RB'].values * self._time_discount_func(
            data['DB'].values, θ['logk'].values, θ['s'].values)
        return VB-VA

    @staticmethod
    @np.vectorize
    def _time_discount_func(delay, logk, s):
        # NOTE: we want logk as a row matrix, and delays as a column matrix to do the
        # appropriate array broadcasting.
        if delay == 0:
            return 1
        else:
            k = np.exp(logk)
            return 1 / (1 + np.power(k * delay, s))


class HyperbolicNonLinearUtility(Model):
    '''Hyperbolic time discounting + non-linear utility model.
    The a-model from ...
    Cheng, J., & González-Vallejo, C. (2014). Hyperbolic Discounting: Value and
    Time Processes of Substance Abusers and Non-Clinical Individuals in
    Intertemporal Choice. PLoS ONE, 9(11), e111378–18.
    http://doi.org/10.1371/journal.pone.0111378
    '''

    def __init__(self, n_particles,
                 prior={'a': norm(loc=1, scale=0.1),
                        'logk': norm(loc=np.log(1/365), scale=2),
                        'α': halfnorm(loc=0, scale=3)}):
        self.n_particles = int(n_particles)
        self.prior = prior
        self.θ_fixed = {'ϵ': 0.01}
        self.choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
        a = np.exp(θ['a'].values)
        VA = (np.power(data['RA'].values,a)
              * self._time_discount_func(data['DA'].values, θ['logk'].values))
        VB = (np.power(data['RB'].values,a)
              * self._time_discount_func(data['DB'].values, θ['logk'].values))
        return VB-VA

    @staticmethod
    def _time_discount_func(delay, logk):
        k = np.exp(logk)
        return 1/(1 + k * delay)


class ITCH(Model):
    '''ITCH model, as presented in:
    Ericson, K. M. M., White, J. M., Laibson, D., & Cohen, J. D. (2015). Money
    earlier or later? Simple heuristics explain intertemporal choices better
    than delay discounting does. Psychological Science, 26(6), 826–833.
    http://doi.org/10.1177/0956797615572232

    Note that we use a choice function _without_ a slope parameter.
    '''

    def __init__(self, n_particles,
                 prior={'β_I': norm(loc=0, scale=50),
                        'β_abs_reward': norm(loc=0, scale=50),
                        'β_rel_reward': norm(loc=0, scale=50),
                        'β_abs_delay': norm(loc=0, scale=50),
                        'β_rel_relay': norm(loc=0, scale=50),
                        'α': halfnorm(loc=0, scale=3)}):
        self.n_particles = int(n_particles)
        self.prior = prior
        self.θ_fixed = {'ϵ': 0.01}
        self.choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
        # organised so that higher values of the decision variable will
        # mean higher probabability for the delayed option (prospect B)

        reward_abs_diff = data['RB'].values - data['RA'].values
        reward_rel_diff = self._rel_diff(data['RB'].values, data['RA'].values)
        delay_abs_diff = data['DB'].values - data['DA'].values
        delay_rel_diff = self._rel_diff(data['DB'].values, data['DA'].values)

        decision_variable = (θ['β_I'].values
                             + θ['β_abs_reward'].values * reward_abs_diff
                             + θ['β_rel_reward'].values * reward_rel_diff
                             + θ['β_abs_delay'].values * delay_abs_diff
                             + θ['β_rel_relay'].values * delay_rel_diff)

        return decision_variable

    @staticmethod
    def _rel_diff(B, A):
        '''Calculate the difference between B and A, normalised by the mean
        of B and A'''
        return (B-A)/((B+A)/2)


class DRIFT(Model):
    '''DRIFT model, as presented in:
    Note that we use a choice function _without_ a slope parameter.

    Read, D., Frederick, S., & Scholten, M. (2013). DRIFT: an analysis of
    outcome framing in intertemporal choice. Journal of Experimental
    Psychology: Learning, Memory, and Cognition, 39(2), 573–588.
    http://doi.org/10.1037/a0029177
    '''

    def __init__(self, n_particles,
                 prior={'β0': norm(loc=0, scale=50),
                        'β1': norm(loc=0, scale=50),
                        'β2': norm(loc=0, scale=50),
                        'β3': norm(loc=0, scale=50),
                        'β4': norm(loc=0, scale=50),
                        'α': halfnorm(loc=0, scale=3)}):
        self.n_particles = int(n_particles)
        self.prior = prior
        self.θ_fixed = {'ϵ': 0.01}
        self.choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
        reward_abs_diff = data['RB'].values - data['RA'].values
        reward_diff = (data['RB'].values - data['RA'].values) / data['RA'].values
        delay_abs_diff = data['DB'].values - data['DA'].values
        delay_component = (data['RB'].values/data['RA'].values)**(1/(delay_abs_diff)) - 1

        decision_variable = (θ['β0'].values
                             + θ['β1'].values * reward_abs_diff
                             + θ['β2'].values * reward_diff
                             + θ['β3'].values * delay_component
                             + θ['β4'].values * delay_abs_diff)

        return decision_variable


class TradeOff(Model):
    '''Tradeoff model by Scholten & Read (2010). Model forumulation as defined
    in Ericson et al (2015).

    Scholten, M., & Read, D. (2010). The psychology of intertemporal tradeoffs.
    Psychological Review, 117(3), 925–944. http://doi.org/10.1037/a0019619
    '''

    def __init__(self, n_particles,
                 prior={'gamma_reward': halfnorm(loc=0, scale=10),
                        'gamma_delay': halfnorm(loc=0, scale=10),
                        'k': norm(loc=0, scale=2),
                        'α': halfnorm(loc=0, scale=3)}):
        self.n_particles = int(n_particles)
        self.prior = prior
        self.θ_fixed = {'ϵ': 0.01}
        self.choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
        return ((self._f(data['RB'].values, θ['gamma_reward'].values)
                 - self._f(data['RA'].values, θ['gamma_reward'].values))
                - θ['k'].values *
                (self._f(data['DB'].values, θ['gamma_delay'].values)
                 - self._f(data['DA'].values, θ['gamma_delay'].values)))

    @staticmethod
    def _f(x, gamma):
        return np.log(1.0 + gamma*x)/gamma
