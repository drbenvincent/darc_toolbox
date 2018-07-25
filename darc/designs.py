from abc import ABC, abstractmethod
from collections import namedtuple
from bad.base_classes import DesignABC, BayesianAdaptiveDesign
import pandas as pd
import numpy as np
import itertools
from bad.optimisation import design_optimisation


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

    trial = 0
    RA, DA, PA = None, None, None
    RB, DB, PB = None, None, None

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

    def get_last_response_chose_delayed(self):
        '''return True if the last response was for the delayed option'''
        if self.all_data.size == 0:
            # no previous responses
            return None

        if list(self.all_data.R)[-1] == 1: # TODO: do this better?
            return True
        else:
            return False


# CONCRETE DESIGN CLASSES BELOW ======================================================

class Kirby2009(DARCDesign):
    '''
    *** KIRBY REFERENCE HERE ***
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
            design = Design(ProspectA=Prospect(reward=self.RA[self.trial], delay=self.DA, prob=self.PA),
                            ProspectB=Prospect(reward=self.RB[self.trial], delay=self.DB[self.trial], prob=self.PB))
            return design
        else:
            return None


class Frye(DARCDesign):
    '''
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

        last_response_chose_delayed = self.get_last_response_chose_delayed()

        if self.trial_per_delay_counter is 0:
            self.RA = self.RB * 0.5
        else:
            self._update_RA_given_last_response(last_response_chose_delayed)
            self.post_choice_adjustment *= 0.5

        design = Design(ProspectA=Prospect(reward=self.RA, delay=self.DA, prob=self.PA),
                        ProspectB=Prospect(reward=self.RB, delay=self.DB[self.delay_counter], prob=self.PB))
        self._increment_counter()
        return design


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

    def _update_RA_given_last_response(self, last_response_chose_delayed):
        # change things depending upon last response
        if last_response_chose_delayed:
            self.RA = self.RA + (self.RB * self.post_choice_adjustment)
        else:
            self.RA = self.RA - (self.RB * self.post_choice_adjustment)




# # CONCRETE BAD CLASSES BELOW -----------------------------------------------------------------

class BAD_delayed_choices(DARCDesign, BayesianAdaptiveDesign):
    '''
    An actual concrete class for doing BAD. 
    Inherit from both DARCDesign and BayesianAdaptiveDesign
    '''

    def __init__(self, DA=[0],
                       DB=np.array([1, 2, 3, 4, 5, 6, 7, 14, 30, 30*6, 365, 365*2, 365*5, 365*10]),
                       RA=None,
                       RB=np.array([100]),
                       max_trials=20):
        super().__init__()
        self.DA = DA
        self.DB = DB
        self.RA = np.linspace(5, RB, num=20)
        #self.RA = RB * 0.5
        self.RB = RB
        self.PA = [1]
        self.PB = [1]
        self.generate_all_possible_designs()
        self.max_trials = max_trials

    def generate_all_possible_designs(self):
        '''Create a dataframe of all possible designs (one design is one row) based upon
        the set of design variables (RA, DA, PA, RB, DB, PB) provided.
        '''
        # NOTE: the order of the two lists below HAVE to be the same
        column_list = ['RA', 'DA', 'PA', 'RB', 'DB', 'PB']
        list_of_lists = [self.RA, self.DA, self.PA, self.RB, self.DB, self.PB]

        # NOTE: list_of_lists must actually be a list of lists... even if there is only one 
        # value being considered for a particular design variable (DA=0) for example, should dbe DA=[0]
        all_combinations = list(itertools.product(*list_of_lists))
        # TODO: we may want to do further trimming and refining of the possible
        # set of designs, based upon domain knowledge etc.
        self.all_possible_designs = pd.DataFrame(
            all_combinations, columns=column_list)

    def get_next_design(self, model):

        if self.trial > self.max_trials - 1:
            return None

        allowable_designs = self.refine_design_space()
        # BAYESIAN DESIGN OPTIMISATION here... calling optimisation.py
        chosen_design, _ = design_optimisation(allowable_designs, model.predictive_y, model.θ)
        # convert from a 1-row pandas dataframe to a Design named tuple
        chosen_design = df_to_design_tuple(chosen_design)
        return chosen_design

    def refine_design_space(self):
        # TODO KLUDGE ALERT. We need to implement the heuristic shrinking of the `all_possible_designs`
        # down to the ones which we will allow on a given trial. For the moment, we are simply
        # by-passing this step, so doing design optimisation over the entire design space (which
        # is problematic).
        allowable_designs = self.all_possible_designs
        return allowable_designs
