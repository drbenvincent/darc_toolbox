'''
Provides base classes related to experimental designs to be used by _any_ 
domain specific use of this Bayesian Adaptive Design package.
'''


from abc import ABC, abstractmethod
import pandas as pd
import logging


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
    trial = int(0)
    all_data = pd.DataFrame()

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
    all_possible_designs = pd.DataFrame()

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


