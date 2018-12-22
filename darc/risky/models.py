from scipy.stats import norm, bernoulli, halfnorm, beta
import numpy as np
from bad.model import Model
from bad.choice_functions import CumulativeNormalChoiceFunc


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
    choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
        VA = data['RA'].values * self._odds_discount_func(data['PA'].values, θ['logh'].values)
        VB = data['RB'].values * self._odds_discount_func(data['PB'].values, θ['logh'].values)
        return VB - VA

    @staticmethod
    def _odds_discount_func(probabilities, logh):
        # transform logh to h
        h = np.exp(logh)
        # convert probability to odds against
        odds_against = prob_to_odds_against(probabilities)
        return 1/(1 + h * odds_against)


class PrelecOneParameter(Model):
    '''Prelec (1998) one parameter probability bias model
    Prelec, D. (1998). The probability weighting function. Econometrica, 66, 497–527.
    '''

    prior = {'γ': beta(1,1),
             'α': halfnorm(loc=0, scale=3)}
    θ_fixed = {'ϵ': 0.01}
    choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
        VA = data['RA'].values * self._w(data['PA'].values, θ['γ'].values)
        VB = data['RB'].values * self._w(data['PB'].values, θ['γ'].values)
        return VB - VA

    @staticmethod
    def _w(p, γ):
        return np.exp(- (-np.log(p))**γ )


class LinearInLogOdds(Model):
    '''Prelec (1998) one parameter probability bias model.
    Gonzalez, R., & Wu, G. (1999). On the shape of the probability weighting
    function. Cognitive Psychology, 38(1), 129–166.
    http://doi.org/10.1006/cogp.1998.0710
    '''

    prior = {'β': beta(1, 1),  # TODO: What prior??????????????????????????????????
             's': beta(1, 1),  # TODO: What prior??????????????????????????????????
             'α': halfnorm(loc=0, scale=3)}
    θ_fixed = {'ϵ': 0.01}
    choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
        VA = data['RA'].values * self._w(data['PA'].values, θ['s'].values, θ['β'].values)
        VB = data['RB'].values * self._w(data['PB'].values, θ['s'].values, θ['β'].values)
        return VB - VA

    @staticmethod
    def _w(p, s, β):
        return s * p**β / (s * p**β + (1-p)**β)


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
    choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        decision_variable = self._calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B

    def _calc_decision_variable(self, θ, data):
        # organised so that higher values of the decision variable will
        # mean higher probabability for the delayed option (prospect B)

        prop_reward = self._proportion(
            data['RA'].values, data['RB'].values)

        prop_risk = self._proportion(
            data['PA'].values, data['PB'].values)

        prop_difference = prop_reward - prop_risk
        decision_variable = prop_difference + θ['δ'].values
        return decision_variable

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

