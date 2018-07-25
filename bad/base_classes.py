'''
Provides base classes to be used by _any_ domain specific use of this Bayesian 
Adaptive Design package.

'''


from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import copy
from bad.inference import update_beliefs
from scipy.stats import norm, bernoulli
from random import random
from bad.choice_functions import CumulativeNormalChoiceFunc


# DESIGN RELATED ===================================================================

class DesignABC(ABC):
    '''
    The top level Abstract Base class for designs.
    '''

    # NOTE: these should probably not be class attributes, but declared in the __init__
    trial = 0
    #all_data = None
    last_response = None

    @abstractmethod
    def get_next_design(self, model):
        ''' This method must be implemented in concrete classes. It should
        output either a Design (a named tuple we are using), or a None when 
        there are no more designs left.
        '''
        pass

    def enter_trial_design_and_response(self, design, response):
        self.update_all_data(design, response)
        self.trial += 1
        # we potentially manually call model to update beliefs here. But so far
        # this is done manually in PsychoPy
        return

    @abstractmethod
    def update_all_data(self, design, response):
        pass


class BayesianAdaptiveDesign(ABC):
    '''An abstract base class for Bayesian Adaptive Design'''

    heuristic_order = None
    all_possible_designs = None

    @abstractmethod
    def generate_all_possible_designs(self):
        pass


    @abstractmethod
    def refine_design_space(self):
        ''' In theory we could do design optimisation on ALL possible designs.
        However, this is complex. You can imagine some sets of designs will map on to
        very distinct values of the decision variable, but that there are many designs
        which will be almost invariant to the decision variable. But we would like to
        explore the design space sensibly. Our general approach is (on any given trial)
        to take a subset of the possible designs and conduct design optimisation on this
        reduced subset. Importantly, over different trials, we are chosing this subset
        intelligently to maximise exploration over the design space.'''
        pass


# MODEL RELATED ====================================================================

# This is meant to be a very general model class. Therefore we have NOTHING specific
# about the design space or the parameters. The only thing we are assuming is that
# we have binary responses R

# NOTE: we are dealing with particles differently from the Matlab version of the code. We are
# only ever representing particles for the free parameters.
# Fixed parameters are just build in to the model class as scalars and are used by model
# functions when needed... they are never converted into a series of particles.


class Model(ABC):

    prior = None
    θ_fixed = None

    # Decide on the choice function we are using. I am going to focus on 
    # `CumulativeNormalChoiceFunc`, but if you want to use another choice function
    # for whatever reason, then it should be pretty obvious how to do this, using
    # `CumulativeNormalChoiceFunc` as an example. You obviously have to update
    # to any new or renamed parameters and ensure you provide either fixed values
    # or a prior over non-fixed parameters.
    choiceFunction = CumulativeNormalChoiceFunc

    def __init__(self, n_particles):
        self.n_particles = n_particles

        # FINISHING UP STUFF ==================================
        # NOTE `prior` and `θ_fixed` must be defined in the concrete model class before
        # we call this. I've not figures out how to demand these exist in this ABC yet
        self.parameter_names = self.prior.keys()
        self.θ = self._θ_initial()

    @abstractmethod
    def calc_decision_variable(self, θ, data):
        pass

    def update_beliefs(self, data):
        '''simply call the low-level `update_beliefs` function'''
        self.θ, _ = update_beliefs(self.p_log_pdf, self.θ, data, display=False)
        return self

    def p_log_pdf(self, θ, data):
        """unnormalized posterior log( p(data|θ)p(θ) )
        θ: pd dataframe
        """
        return self.log_likelihood(θ, data) + self.log_prior_pdf(θ)

    def log_likelihood(self, θ, data):
        """
        Calculate the log liklihood of the data for given theta parameters.
        Σ log(p(data|θ))
        """
        # past_responses: vector of binary responses
        # p_chose_delayed: model prediction(0-1) of response
        
        # past_responses = data['R'].astype('int')
        # p_chose_delayed = self.predictive_y(θ, data)
        # log_liklihoods = bernoulli.logpmf(past_responses, p_chose_delayed)
        # return np.sum(log_liklihoods, axis=1)

        n_trials, _ = data.shape
        n_particles, _ = θ.shape
        # if no data, return ll = 0

        # # iterate over trials, calculating p_chose_delayed
        # p_chose_delayed = []
        # for trial in range(n_trials):
        #     trial_data = data.take([trial])
        #     p_chose_delayed[trial] = self.predictive_y(θ, trial_data)
        
        # this creates a list of lists
        # p_chose_delayed = [self.predictive_y(θ, data.take([trial])) for trial in range(n_trials)]

        # # deal with the chose immediate trials
        # for trial in range(n_trials):
        #     trialdata = data.take([trial])
        #     R = trialdata.R.values[0]
        #     if R is 0:
        #         p_chose_delayed[trial] = 1 - p_chose_delayed[trial]

        # # then calculate log likelihood... sum(log(p_chose_delayed))
        # ll = [sum(np.log(p_chose_delayed[trial])) for trial in range(n_trials)]
        # #ll = sum(np.log(p_chose_delayed))
        # return ll

        p_chose_delayed = np.zeros((n_particles, n_trials))
        R = data.R.values

        for trial in range(n_trials):
            trial_data = data.take([trial])
            if R[trial] is 1:
                # chose delayed trial
                p_chose_delayed[:, trial] = self.predictive_y(
                    θ, trial_data)
            elif R[trial] is 0:
                # choise immediate trial
                p_chose_delayed[:, trial] = 1 - self.predictive_y(
                    θ, trial_data)

        ll = np.sum(np.log(p_chose_delayed), axis=1)
        return ll
        

    def log_prior_pdf(self, θ):
        """Evaluate the log prior density, log(p(θ)), for the values θ
        θ: dictionary, each key is a parameter name
        """
        log_prior = copy.copy(θ)
        for key in self.parameter_names:
            log_prior[key] = self.prior[key].logpdf(x=θ[key])

        log_prior = np.sum(log_prior, axis=1)  # sum over columns (parameters)
        return log_prior

    def _θ_initial(self):
        """Generate initial θ particles. Might as well sample from the prior"""
        # dictionary comprehension... iterate through the parameter and sample values from it
        particles_dict = {key: self.prior[key].rvs(size=self.n_particles)
            for key in self.parameter_names}
        return pd.DataFrame.from_dict(particles_dict)

    def predictive_y(self, θ, data):
        ''' Calculate the probability of chosing delayed 

        We need this to work in multiple contexts:

        INFERENCE CONTEXT
        input: θ has P rows, for example P = 5000 particles
        input: data has T rows, equal to number of trials we've run
        DESIRED output: p_chose_delayed is a P x 1 array

        OPTIMISATION CONTEXT
        input: θ has N rows (eg N=500)
        input: data has N rows
        DESIRED output: p_chose_delayed is a N x 1 array

        TODO: do some assertions here to catch errors
        '''
        print(f'θ:    {θ.shape}')
        print(f'data: {data.shape}')
        decision_variable = self.calc_decision_variable(θ, data)
        p_chose_delayed = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_delayed


    def _get_simulated_response(self, data, θtrue):
        '''
        Get simulated response for a given set of true parameter
        ONLY NEEDED FOR SIMILATED EXPERIMENTS?
        '''
        assert θtrue.shape[0] == 1, 'Only one true set of parameters expected'
        assert data.shape[0] == 1, 'Only one design expected'

        p_chose_delayed = self.predictive_y(θtrue, data)
        print(p_chose_delayed)
        # TODO: this assumes only one data (ie trial) is coming in
        chose_delayed = random() < p_chose_delayed
        return chose_delayed
