'''
Provides base classes related to experimental designs to be used by _any_
domain specific use of this Bayesian Adaptive Design package.
'''


from abc import ABC, abstractmethod
import pandas as pd
import logging


class TrialData():
    '''A class to hold trial data from real (or simulated) experiments'''

    def __init__(self):
        # generate empty dataframe
        data_columns = ['RA', 'DA', 'PA', 'RB', 'DB', 'PB', 'R']
        self.df = pd.DataFrame(columns=data_columns)

    def get_last_response_chose_B(self):
        '''return True if the last response was for the option B'''
        if self.df.size == 0:
            # no previous responses
            return None

        if list(self.df.R)[-1] == 1:  # TODO: do this better?
            return True
        else:
            return False

    def update_data(self, design, response):
        # TODO: need to specify types here I think... then life might be
        # easier to decant the data out at another point
        # trial_df = design_to_df(design)
        # self.data = self.data.append(trial_df)

        trial_data = {'RA': design.ProspectA.reward,
                    'DA': design.ProspectA.delay,
                    'PA': design.ProspectA.prob,
                    'RB': design.ProspectB.reward,
                    'DB': design.ProspectB.delay,
                    'PB': design.ProspectB.prob,
                    'R': [int(response)]}
        self.df = self.df.append(pd.DataFrame(trial_data))

        # a bit clumsy but...
        self.df['R'] = self.df['R'].astype('int64')

        self.df = self.df.reset_index(drop=True)
        return

    # TODO: look up how to do getters in a Pythonic way.
    def get_df(self):
        '''return dataframe of data'''
        return self.df


class DesignGeneratorABC(ABC):
    '''
    The top level Abstract Base class for designs. It is not functional in itself,
    but provides the template for handling designs for given contexts, such as DARC.

    Core functionality is:
    a) It pumps out experimental designs with the get_next_design() method.
    b) It recieves the resulting (design, response) pair and stores a history of
       designs and responses.
    '''


    def __init__(self):
        self.trial = int(0)
        self.data = TrialData()

    @abstractmethod
    def get_next_design(self, model):
        ''' This method must be implemented in concrete classes. It should
        output either a Design (a named tuple we are using), or a None when
        there are no more designs left.
        '''
        pass

    # MIDDLE MAN METHODS ===========================================================
    def enter_trial_design_and_response(self, design, response):
        '''middle-man method'''
        self.data.update_data(design, response)
        self.trial += 1
        # we potentially manually call model to update beliefs here. But so far
        # this is done manually in PsychoPy
        return

    def get_last_response_chose_B(self):
        return self.data.get_last_response_chose_B()

    def get_df(self):
        return self.data.get_df()
