from scipy.stats import norm, halfnorm
import numpy as np
from badapted.model import Model
from badapted.choice_functions import CumulativeNormalChoiceFunc


# TODO: THESE UTILITY FUNCTIONS ARE IN MULTIPLE PLACES !!!
def prob_to_odds_against(probabilities):
    '''convert probabilities of getting reward to odds against getting it'''
    odds_against = (1 - probabilities) / probabilities
    return odds_against


def odds_against_to_probs(odds):
    probabilities = 1 / (1+odds)
    return probabilities


class MultiplicativeHyperbolic(Model):
    '''Hyperbolic risk discounting model
    The idea is that we hyperbolically discount ODDS AGAINST the reward

    Vanderveldt, A., Green, L., & Myerson, J. (2015). Discounting of monetary
    rewards that are both delayed and probabilistic: delay and probability
    combine multiplicatively, not additively. Journal of Experimental
    Psychology: Learning, Memory, and Cognition, 41(1), 148–162.
    http://doi.org/10.1037/xlm0000029
    '''

    def __init__(self, n_particles,
                 prior={'logk': norm(loc=np.log(1/365), scale=2),
                        'logh': norm(loc=0, scale=1),
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
        VA = (data['RA'].values
              * self._time_discount_func(data['DA'].values, θ['logk'].values)
              * self._odds_discount_func(data['PA'].values, θ['logh'].values))
        VB = (data['RB'].values
              * self._time_discount_func(data['DB'].values, θ['logk'].values)
              * self._odds_discount_func(data['PB'].values, θ['logh'].values))
        return VB - VA

    @staticmethod
    def _time_discount_func(delay, logk):
        k = np.exp(logk)
        return 1/(1 + k * delay)

    @staticmethod
    def _odds_discount_func(probabilities, logh):
        # transform logh to h
        h = np.exp(logh)
        # convert probability to odds against
        odds_against = prob_to_odds_against(probabilities)
        return 1/(1 + h * odds_against)
