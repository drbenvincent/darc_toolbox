from abc import ABC, abstractmethod
from collections import namedtuple
from bad.base_classes import DesignABC, BayesianAdaptiveDesign
import pandas as pd
import numpy as np
import itertools
from bad.optimisation import design_optimisation
import matplotlib.pyplot as plt
import copy
import logging
import time
from darc.data_plotting import all_data_plotter


DEFAULT_DB = np.concatenate([
    np.array([1, 2, 5, 10, 15, 30, 45])/24/60,
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 12])/24,
    np.array([1, 2, 3, 4, 5, 6, 7]),
    np.array([2, 3, 4])*7,
    np.array([3, 4, 5, 6, 8, 9])*30,
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25])*365]).tolist()
       
# define useful data structures
Prospect = namedtuple('Prospect', ['reward', 'delay', 'prob'])
Design = namedtuple('Design', ['ProspectA', 'ProspectB'])

# helper functions
def design_tuple_to_df(design):
    ''' Convert the named tuple into a 1-row pandas dataframe'''
    trial_data = {'RA': design.ProspectA.reward,
                  'DA': design.ProspectA.delay,
                  'PA': design.ProspectA.prob,
                  'RB': design.ProspectB.reward,
                  'DB': design.ProspectB.delay,
                  'PB': design.ProspectB.prob}
    return pd.DataFrame(trial_data)

def df_to_design_tuple(df):
    ''' Convert 1-row pandas dataframe into named tuple'''
    RA = df.RA.values[0]
    DA = df.DA.values[0]
    PA = df.PA.values[0]
    RB = df.RB.values[0]
    DB = df.DB.values[0]
    PB = df.PB.values[0]
    chosen_design = Design(ProspectA=Prospect(reward=RA, delay=DA, prob=PA),
                           ProspectB=Prospect(reward=RB, delay=DB, prob=PB))
    return chosen_design


## ANOTHER BASE CLASS: Users not to change this

class DARCDesign(DesignABC):
    '''
    Another abstract base class which extends the basic design class, adding
    specialisations for our DARC domain. This includes:
    - the design space variables
    - how to update (design, response) pairs from an experimental trial
    - some basic plotting of the raw data
    '''

    RA, DA, PA = list(), list(), list()
    RB, DB, PB = list(), list(), list()

    def __init__(self):
        # generate empty `all_data`
        data_columns = ['RA', 'DA', 'PA', 'RB', 'DB', 'PB', 'R']
        self.all_data = pd.DataFrame(columns=data_columns)

    def update_all_data(self, design, response):
        # TODO: need to specify types here I think... then life might be 
        # easier to decant the data out at another point
        # trial_df = design_to_df(design)
        # self.all_data = self.all_data.append(trial_df)
        
        trial_data = {'RA': design.ProspectA.reward,
                      'DA': design.ProspectA.delay,
                      'PA': design.ProspectA.prob,
                      'RB': design.ProspectB.reward,
                      'DB': design.ProspectB.delay,
                      'PB': design.ProspectB.prob,
                      'R': [int(response)]}
        self.all_data = self.all_data.append(pd.DataFrame(trial_data))
        return

    def plot_all_data(self, filename):
        '''Visualise data'''
        all_data_plotter(self.all_data, filename) 

    def generate_all_possible_designs(self, assume_discounting=True):
        '''Create a dataframe of all possible designs (one design is one row) based upon
        the set of design variables (RA, DA, PA, RB, DB, PB) provided.
        '''
        # NOTE: the order of the two lists below HAVE to be the same
        column_list = ['RA', 'DA', 'PA', 'RB', 'DB', 'PB']
        list_of_lists = [self.RA, self.DA, self.PA, self.RB, self.DB, self.PB]

        # Log the raw values to help with debugging
        logging.debug(f'provided RA = {self.RA}')
        logging.debug(f'provided DA = {self.DA}')
        logging.debug(f'provided PA = {self.PA}')
        logging.debug(f'provided RB = {self.RB}')
        logging.debug(f'provided DB = {self.DB}')
        logging.debug(f'provided PB = {self.PB}')

        # NOTE: list_of_lists must actually be a list of lists... even if there is only one
        # value being considered for a particular design variable (DA=0) for example, should dbe DA=[0]
        all_combinations = list(itertools.product(*list_of_lists))
        D = pd.DataFrame(all_combinations, columns=column_list)
        logging.debug(
            f'{D.shape[0]} designs generated initially')

        # eliminate any designs where DA>DB, because by convention ProspectB is our more delayed reward
        D.drop(D[D.DA > D.DB].index, inplace=True)
        logging.debug(f'{D.shape[0]} left after dropping DA>DB')
        
        if assume_discounting:
            D.drop(D[D.RB < D.RA].index, inplace=True)
            logging.debug(f'{D.shape[0]} left after dropping RB<RA')

        # NOTE: we may want to do further trimming and refining of the possible
        # set of designs, based upon domain knowledge etc.

        # check we actually have some designs!
        if D.shape[0] == 0:
            logging.error(f'No ({D.shape[0]}) designs generated!')

        # set the values
        self.all_possible_designs = D


# CONCRETE DESIGN CLASSES BELOW ======================================================

class Kirby2009(DARCDesign):
    '''
    A class to provide designs from the Kirby (2009) delay discounting task.

    Kirby, K. N. (2009). One-year temporal stability of delay-discount rates. 
    Psychonomic Bulletin & Review, 16(3):457–462.
    '''

    # NOTE: these should probably not be class attributes, but declared in the __init__ 
    # only likely to be a problem if we have mulitple Kirby2009 object instances. We'd
    # also have to explicitly call the superclass constructor at that point, I believe.
    max_trials = 27
    RA = [80, 34, 25, 11, 49, 41, 34, 31, 19, 22, 55, 28, 47,
          14, 54, 69, 54, 25, 27, 40, 54, 15, 33, 24, 78, 67, 20]
    DA = 0
    RB = [85, 50, 60, 30, 60, 75, 35, 85, 25, 25, 75, 30, 50,
          25, 80, 85, 55, 30, 50, 55, 60, 35, 80, 35, 80, 75, 55]
    DB = [157, 30, 14, 7, 89, 20, 186, 7, 53, 136, 61, 179, 160, 19,
          30, 91, 117, 80, 21, 62, 111, 13, 14, 29, 162, 119, 7]
    PA, PB = 1, 1

    def get_next_design(self, _):
        # NOTE: This is un-Pythonic as we are asking permission... we should just do it, and have a catch ??
        if self.trial < self.max_trials - 1:
            logging.info(f'Getting design for trial {self.trial}')
            design = Design(ProspectA=Prospect(reward=self.RA[self.trial], delay=self.DA, prob=self.PA),
                            ProspectB=Prospect(reward=self.RB[self.trial], delay=self.DB[self.trial], prob=self.PB))
            return design
        else:
            return None


class Frye(DARCDesign):
    '''
    A class to provide designs based on the Frye et al (2016) protocol.

    Frye, C. C. J., Galizio, A., Friedel, J. E., DeHart, W. B., & Odum, A. L.
    (2016). Measuring Delay Discounting in Humans Using an Adjusting Amount
    Task. Journal of Visualized Experiments, (107), 1-8.
    http://doi.org/10.3791/53584
    '''
    
    def __init__(self, DB=[7, 30, 365], RB=100., trials_per_delay=5):
        self.DA = 0
        self.DB = DB
        self.RB = RB
        self.R_A = RB * 0.5
        self.post_choice_adjustment = 0.25
        self.trials_per_delay = trials_per_delay
        self.trial_per_delay_counter = 0
        self.delay_counter = 0
        self.PA = 1
        self.PB = 1
        # call the superclass constructor
        super().__init__()

    def get_next_design(self, _):
        """return the next design as a tuple of prospects"""
        
        if self.delay_counter == len(self.DB):
            return None

        logging.info(f'Getting design for trial {self.trial}')
        last_response_chose_B = self.get_last_response_chose_B()

        if self.trial_per_delay_counter is 0:
            self.RA = self.RB * 0.5
        else:
            self._update_RA_given_last_response(last_response_chose_B)
            self.post_choice_adjustment *= 0.5

        design = Design(ProspectA=Prospect(reward=self.RA, delay=self.DA, prob=self.PA),
                        ProspectB=Prospect(reward=self.RB, delay=self.DB[self.delay_counter], prob=self.PB))
        self._increment_counter()
        return design

    # Below are helper functions

    def _increment_counter(self):
        """Increment trial counter, and increment delay counter if we have done all the trials per delay"""
        self.trial_per_delay_counter += 1
        # reset trial_per_delay_counter if need be
        if self.trial_per_delay_counter > self.trials_per_delay-1:
            self._increment_delay()


    def _increment_delay(self):
        """ Done trials_per_delay trials for this delay, so we'll move on to the next delay level now"""
        self.delay_counter += 1
        self.trial_per_delay_counter = 0
        self.post_choice_adjustment = 0.25

    def _update_RA_given_last_response(self, last_response_chose_B):
        # change things depending upon last response
        if last_response_chose_B:
            self.RA = self.RA + (self.RB * self.post_choice_adjustment)
        else:
            self.RA = self.RA - (self.RB * self.post_choice_adjustment)




# CONCRETE BAD CLASSES BELOW -----------------------------------------------------------------

class DARC_Designs(DARCDesign, BayesianAdaptiveDesign):
    '''
    A class for running DARC choice tasks with Bayesian Adaptive Design.
    '''

    def __init__(self, DA=[0], DB=DEFAULT_DB, RA=list(), RB=[100], 
                 PA=[1], PB=[1], max_trials=20):
        super().__init__()

        
        self._input_type_validation(RA, DA, PA, RB, DB, PB)
        self._input_value_validation(PA, PB, DA, DB)
        RA, DA, PA, RB, DB, PB = self._apply_logic_to_design_space_spec(RA, DA, PA, RB, DB, PB)  

        self.DA = DA
        self.DB = DB
        self.RA = RA
        self.RB = RB
        self.PA = PA
        self.PB = PB
        self.max_trials = max_trials

        self.generate_all_possible_designs()

    def _apply_logic_to_design_space_spec(self, RA, DA, PA, RB, DB, PB):
        '''Apply our domain specific logic to the design space specs provided'''
        # go through logic based on inputs
        if len(RA) == 0:
            # we have vector RB
            RA_over_RB = np.linspace(0.05, 0.95, 19)  # [5, 10, 15, .... 95]
            RA = list(np.array(RB) * RA_over_RB)  
        return RA, DA, PA, RB, DB, PB

    def _input_type_validation(self, RA, DA, PA, RB, DB, PB):
        # NOTE: possibly not very Pythonic
        assert isinstance(RA, list), "RA should be a list"
        assert isinstance(DA, list), "DA should be a list"
        assert isinstance(PA, list), "PA should be a list"
        assert isinstance(RB, list), "RB should be a list"
        assert isinstance(DB, list), "DB should be a list"
        assert isinstance(PB, list), "PB should be a list"
        
    def _input_value_validation(self, PA, PB, DA, DB):
        '''Confirm values of provided design space specs are valid'''
        if np.any((np.array(PA) < 0) | (np.array(PA) > 1)):
            raise ValueError('Expect all values of PA to be between 0-1')

        if np.any((np.array(PB) < 0) | (np.array(PB) > 1)):
            raise ValueError('Expect all values of PB to be between 0-1')

        if np.any(np.array(DA) < 0):
            raise ValueError('Expecting all values of DA to be >= 0')

        if np.any(np.array(DB) < 0):
            raise ValueError('Expecting all values of DB to be >= 0')

    def get_next_design(self, model):

        if self.trial > self.max_trials - 1:
            return None
        start_time = time.time()
        logging.info(f'Getting design for trial {self.trial}')
        allowable_designs = self.refine_design_space(model)
        # BAYESIAN DESIGN OPTIMISATION here... calling optimisation.py
        chosen_design, _ = design_optimisation(allowable_designs, model.predictive_y, model.θ)
        # convert from a 1-row pandas dataframe to a Design named tuple
        chosen_design = df_to_design_tuple(chosen_design)
        logging.debug(f'chosen design is: {chosen_design}')
        logging.info(f'get_next_design() took: {time.time()-start_time:1.3f} seconds')
        return chosen_design

    def refine_design_space(self, model, NO_REPEATS=True):
        '''
        This will implement something very simular to our initial Matlab implementation written 
        by Tom Rainforth.
        https://github.com/drbenvincent/darc-experiments-matlab/blob/master/darc-experiments/response_error_types/%40ChoiceFuncPsychometric/generate_designs.m
        '''
        
        allowable_designs = copy.copy(self.all_possible_designs)
        logging.debug(f'{allowable_designs.shape[0]} designs initially')

        if NO_REPEATS and self.trial>1:
            allowable_designs = remove_trials_already_run(
                allowable_designs, self.all_data.drop(columns=['R']))

        allowable_designs = remove_highly_predictable_designs(
            allowable_designs, model)

        if allowable_designs.shape[0] == 0:
            logging.error(f'No ({allowable_designs.shape[0]}) designs left')

        if allowable_designs.shape[0] < 10:
            logging.warning(f'Very few ({allowable_designs.shape[0]}) designs left')

        return allowable_designs


def remove_trials_already_run(design_set, exclude_these):
    '''Take in a set of designs (design_set) and remove aleady run trials (exclude_these)
    Dropping duplicates will work in this situation because `exclude_these` is going to 
    be a subset of `design_set`'''
    # see https://stackoverflow.com/a/40209800/5172570
    allowable_designs = pd.concat([design_set, exclude_these]).drop_duplicates(keep=False)
    logging.debug(f'{allowable_designs.shape[0]} designs after removing prior designs')
    return allowable_designs


def remove_highly_predictable_designs(allowable_designs, model):
    ''' Eliminate designs which are highly predictable as these will not be very informative '''
    θ_point_estimate = model.get_θ_point_estimate()

    # TODO: CHECK WE CAN EPSILON TO 0
    p_chose_B = model.predictive_y(θ_point_estimate, allowable_designs)
    # add p_chose_B as a column to allowable_designs
    allowable_designs['p_chose_B'] = pd.Series(p_chose_B)
    # label rows which are highly predictable
    threshold = 0.011  # TODO: Tom used a lower threshold of 0.005, but that was with epsilon=0
    highly_predictable = (allowable_designs['p_chose_B'] < threshold) | (
        allowable_designs['p_chose_B'] > 1 - threshold)
    allowable_designs['highly_predictable'] = pd.Series(highly_predictable)

    n_not_predictable = allowable_designs.size - sum(allowable_designs.highly_predictable)
    if n_not_predictable > 10:
        # drop the offending designs (rows)
        allowable_designs = allowable_designs.drop(
            allowable_designs[allowable_designs.p_chose_B < threshold].index)
        allowable_designs = allowable_designs.drop(
            allowable_designs[allowable_designs.p_chose_B > 1 - threshold].index)
    else:
        # take the 10 designs closest to p_chose_B=0.5
        # NOTE: This is not exactly the same as Tom's implementation which examines
        # VB-VA (which is the design variable axis) and also VA+VB (orthogonal to
        # the design variable axis)
        logging.warning('not many unpredictable designs, so taking the 10 closest to unpredictable')
        allowable_designs['badness'] = np.abs(0.5- allowable_designs.p_chose_B)
        allowable_designs.sort_values(by=['badness'], inplace=True)
        allowable_designs = allowable_designs[:10]

    allowable_designs.drop(columns=['p_chose_B'])
    logging.debug(f'{allowable_designs.shape[0]} designs after removing highly predicted designs')
    return allowable_designs
