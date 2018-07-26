from scipy.stats import norm, bernoulli, halfnorm
import numpy as np
from bad.base_classes import Model


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
    prior['logh'] = norm(loc=0, scale=1)  # h=1 (ie logh=0) equates to risk neutral
    prior['α'] = halfnorm(loc=0, scale=3)
    θ_fixed = {'ϵ': 0.01}

    def calc_decision_variable(self, θ, data):
        VA = data['RA'].values * self.odds_discount_func(data['PA'].values, θ)
        VB = data['RB'].values * self.odds_discount_func(data['PB'].values, θ)
        return VB - VA
    
    @staticmethod
    def odds_discount_func(probabilities, θ):
        # transform logh to h
        h = np.exp(θ['logh'].values)
        # convert probability to odds against
        odds_against = prob_to_odds_against(probabilities)
        return np.divide(1, (1 + h * odds_against))


class ProspectTheory(Model):
    '''Prospect Theory'''
    pass

