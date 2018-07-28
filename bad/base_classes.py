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
import matplotlib.pyplot as plt
import logging


# DESIGN RELATED ===================================================================

class DesignABC(ABC):
    '''
    The top level Abstract Base class for designs. It is not functional in itself, 
    but provides the template for handling designs for given contexts, such as DARC.

    Core functionality is:
    a) It pumps out experimental designs with the get_next_design() method.
    b) It recieves the resulting (design, response) pair and stores a history of
       designs and responses.
    '''

    # NOTE: these should probably not be class attributes, but declared in the __init__
    trial = 0
    all_data = None
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

    def get_last_response_chose_B(self):
        '''return True if the last response was for the option B'''
        if self.all_data.size == 0:
            # no previous responses
            return None

        if list(self.all_data.R)[-1] == 1:  # TODO: do this better?
            return True
        else:
            return False


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
    '''
    Model abstract base class. It does nothing on it's own, but it sketches out the core
    elements of _any_ model which we could use. To be clear, a model could be pretty much
    any computational/mathematical model which relates inputs (ie experimental designs) 
    and model parameters to a behavioural response:
        response = f(design, parameters)
    
    We are only considering experimental paradigms where we have two possible responses.
    This is a simplification, but it also covers a very wide range of experiment classes
    including:
    a) psychophysics such as yes/no or 2AFC paradigms, 
    b) decision making experiments with choices between 2 prospects

    I also impose that all of the models will involve a single decision variable. A choice
    function then operates on this decision variable in order to produce a probability of
    responding one way of the other.
    '''

    prior = None
    θ_fixed = None
    θ_true = None

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
        We are going to iterate over trials. For each one, we take the trial
        data and calculate the predictive_y. This gives us many values 
        (correspoding to particles). We deal with these appropriately for 
        'chose B' and 'chose A' trials. Then calculate the log
        likelihood, which involves summing the ll over trials so that we end
        up with a log likelihood value for all the particles.
        """

        n_trials, _ = data.shape
        n_particles, _ = θ.shape
        
        # TODO safety check... if no data, return ll = 0

        p_chose_B = np.zeros((n_particles, n_trials))
        R = data.R.values

        for trial in range(n_trials):
            trial_data = data.take([trial])
            if R[trial] is 1:  # meaning they chose option B
                p_chose_B[:, trial] = self.predictive_y( θ, trial_data)
            elif R[trial] is 0:  # meaning they chose option A
                p_chose_B[:, trial] = 1 - self.predictive_y(θ, trial_data)
            else:
                raise ValueError('Failing to identify response')

        ll = np.sum(np.log(p_chose_B), axis=1)
        return ll
        
    def log_prior_pdf(self, θ):
        """Evaluate the log prior density, log(p(θ)), for the values θ
        θ: dictionary, each key is a parameter name
        """
        # NOTE: avoid tears by copying θ. If we don't do this then we unintentionally
        # and undesirably update θ itself.
        log_prior = copy.copy(θ)
        for key in self.parameter_names:
            log_prior[key] = self.prior[key].logpdf(x=θ[key])

        log_prior = np.sum(log_prior, axis=1)  # sum over columns (parameters)
        return log_prior

    def _θ_initial(self):
        """Generate initial θ particles, by sampling from the prior"""
        particles_dict = {key: self.prior[key].rvs(size=self.n_particles)
            for key in self.parameter_names}
        return pd.DataFrame.from_dict(particles_dict)

    def predictive_y(self, θ, data):
        ''' 
        Calculate the probability of chosing B. We need this to work in multiple
        contexts:

        INFERENCE CONTEXT
        input: θ has P rows, for example P = 5000 particles
        input: data has T rows, equal to number of trials we've run
        DESIRED output: p_chose_B is a P x 1 array

        OPTIMISATION CONTEXT
        input: θ has N rows (eg N=500)
        input: data has N rows
        DESIRED output: p_chose_B is a N x 1 array

        TODO: do some assertions on sizes of inputs/outputs here to catch errors
        '''
        decision_variable = self.calc_decision_variable(θ, data)
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B


    def _get_simulated_response(self, design_tuple):
        '''
        Get simulated response for a given set of true parameter.
        This functionality is only needed when we are simulating experiment. It is not
        needed when we just want to run experiments on real participants.
        '''

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO: this being here violates bad not knowing about darc
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        ''' Convert the named tuple into a 1-row pandas dataframe'''
        trial_data = {'RA': [design_tuple.ProspectA.reward],
                      'DA': [design_tuple.ProspectA.delay],
                      'PA': [design_tuple.ProspectA.prob],
                      'RB': [design_tuple.ProspectB.reward],
                      'DB': [design_tuple.ProspectB.delay],
                      'PB': [design_tuple.ProspectB.prob]}
        design_df = pd.DataFrame.from_dict(trial_data)

        p_chose_B = self.predictive_y(self.θ_true, design_df)
        chose_B = random() < p_chose_B[0]
        return chose_B

    def export_posterior_histograms(self, filename):
        '''Export pdf of marginal posteriors
        filename: expecting this to be a string of filename and experiment date & time.
        '''
        n_params = len(self.prior)
        fig, axes = plt.subplots(1, n_params, figsize=(9, 4), tight_layout=True)
        for (axis, key) in zip(axes, self.θ.keys()):
            axis.hist(self.θ[key], 100, density=1, facecolor='green', alpha=0.75)
            axis.set_xlabel(key)
            if self.θ_true is not None:
                axis.axvline(x=self.θ_true[key][0],
                             color='red', linestyle='--')
        savename = filename + '_parameter_plot.pdf'
        plt.savefig(savename)
        logging.info(f'📊 Posterior histograms exported: {savename}')

    def get_θ_point_estimate(self):
        '''return a point estimate (posterior median) for the model parameters'''
        median_series = self.θ.median(axis=0)
        return median_series.to_frame().T
        
