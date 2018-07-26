from scipy.stats import norm, bernoulli, halfnorm
import numpy as np
from bad.base_classes import Model


# TODO: THESE UTILITY FUNCTIONS ARE IN MULTIPLE PLACES !!!
def prob_to_odds_against(probabilities):
    '''convert probabilities of getting reward to odds against getting it'''
    odds_against = (1 - probabilities) / probabilities
    return odds_against

def odds_against_to_probs(odds):
    probabilities = 1 / (1+odds)
    return probabilities


class Hyperbolic(Model):
    '''Hyperbolic risk discounting model
    The idea is that we hyperbolically discount ODDS AGAINST the reward.
    '''

    prior = dict()
    # h=1 (ie logh=0) equates to risk neutral
    prior['logh'] = norm(loc=0, scale=1)
    prior['α'] = halfnorm(loc=0, scale=3)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        VA = data['RA'].values * self._odds_discount_func(data['PA'].values, θ)
        VB = data['RB'].values * self._odds_discount_func(data['PB'].values, θ)
        return VB - VA
    
    @staticmethod
    def _odds_discount_func(probabilities, θ):
        # transform logh to h
        h = np.exp(θ['logh'].values)
        # convert probability to odds against
        odds_against = prob_to_odds_against(probabilities)
        return np.divide(1, (1 + h * odds_against))


class ProportionalDifference(Model):
    '''Proportional difference model for risky rewards
    
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

        prop_reward = self._proportion(
            data['RA'].values, data['RB'].values)

        prop_risk = self._proportion(
            data['PA'].values, data['PB'].values)

        prop_difference = (prop_reward - prop_risk)
        decision_axis = prop_difference + θ['δ']
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


class ProspectTheory(Model):
    '''Prospect Theory'''
    pass

