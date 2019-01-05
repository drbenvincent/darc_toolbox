from abc import ABC, abstractmethod
from bad.designs import DesignGeneratorABC
from darc import Prospect, Design
import pandas as pd
import numpy as np
from bad.optimisation import design_optimisation
import matplotlib.pyplot as plt
import copy
import logging
import itertools
import time
import random


DEFAULT_DB = np.concatenate([
    np.array([1, 2, 5, 10, 15, 30, 45])/24/60,
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 12])/24,
    np.array([1, 2, 3, 4, 5, 6, 7]),
    np.array([2, 3, 4])*7,
    np.array([3, 4, 5, 6, 8, 9])*30,
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25])*365]).tolist()

# helper functions

# def design_tuple_to_df(design):
#     ''' Convert the named tuple into a 1-row pandas dataframe'''
#     trial_data = {'RA': design.ProspectA.reward,
#                   'DA': design.ProspectA.delay,
#                   'PA': design.ProspectA.prob,
#                   'RB': design.ProspectB.reward,
#                   'DB': design.ProspectB.delay,
#                   'PB': design.ProspectB.prob}
#     return pd.DataFrame(trial_data)

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


# CONCRETE BAD CLASSES BELOW -----------------------------------------------------------------

class BayesianAdaptiveDesignGeneratorDARC(DesignGeneratorABC):
    '''
    This class selects the next design to run, based on a provided design
    space, a model, and a design/response history.
    '''

    def __init__(self, design_space,
                 max_trials=20,
                 NO_REPEATS=False):
        super().__init__()

        self.all_possible_designs = design_space
        self.max_trials = max_trials
        self.NO_REPEATS = NO_REPEATS


    def get_next_design(self, model):

        if self.trial > self.max_trials - 1:
            return None
        start_time = time.time()
        logging.info(f'Getting design for trial {self.trial}')

        allowable_designs = copy.copy(self.all_possible_designs)
        logging.debug(f'{allowable_designs.shape[0]} designs initially')

        allowable_designs = _remove_highly_predictable_designs(allowable_designs,
                                                               model)

        allowable_designs = self._refine_design_space(model,
                                                      allowable_designs)

        chosen_design_df, _ = design_optimisation(allowable_designs,
                                                  model.predictive_y,
                                                  model.θ)

        chosen_design_named_tuple = df_to_design_tuple(chosen_design_df)

        logging.debug(f'chosen design is: {chosen_design_named_tuple}')
        logging.info(
            f'get_next_design() took: {time.time()-start_time:1.3f} seconds')
        return chosen_design_named_tuple


    def _refine_design_space(self, model, allowable_designs):
        '''A series of filter operations to refine down the space of designs which we
        do design optimisations on.'''

        # allowable_designs = copy.copy(self.all_possible_designs)
        # logging.debug(f'{allowable_designs.shape[0]} designs initially')

        if self.NO_REPEATS and self.trial>1:
            allowable_designs = _remove_trials_already_run(
                allowable_designs, self.data.df.drop(columns=['R'])) # TODO: resolve this

        if allowable_designs.shape[0] == 0:
            logging.error(f'No ({allowable_designs.shape[0]}) designs left')

        if allowable_designs.shape[0] < 10:
            logging.warning(f'Very few ({allowable_designs.shape[0]}) designs left')

        return allowable_designs


def _remove_trials_already_run(design_set, exclude_these):
    '''Take in a set of designs (design_set) and remove aleady run trials (exclude_these)
    Dropping duplicates will work in this situation because `exclude_these` is going to
    be a subset of `design_set`'''
    # see https://stackoverflow.com/a/40209800/5172570
    allowable_designs = pd.concat([design_set, exclude_these]).drop_duplicates(keep=False)
    logging.debug(f'{allowable_designs.shape[0]} designs after removing prior designs')
    return allowable_designs


def _choose_one_along_design_dimension(allowable_designs, design_dim_name):
    '''We are going to take one design dimension given by `design_dim_name` and randomly
    pick one of it's values and hold it constant by removing all others from the list of
    allowable_designs.
    The purpose of this is to promote variation along the chosen design dimension.
    Cutting down the set of allowable_designs which we do design optimisation on is a
    nice side-effect rather than a direct goal.
    '''
    unique_values = allowable_designs[design_dim_name].unique()
    chosen_value = random.choice(unique_values)
    # filter by chosen value of this dimension
    allowable_designs = allowable_designs.loc[allowable_designs[design_dim_name] == chosen_value]
    return allowable_designs


def _remove_highly_predictable_designs(allowable_designs, model):
    ''' Eliminate designs which are highly predictable as these will not be very informative '''
    θ_point_estimate = model.get_θ_point_estimate()

    # TODO: CHECK WE CAN EPSILON TO 0
    p_chose_B = model.predictive_y(θ_point_estimate, allowable_designs)
    # add p_chose_B as a column to allowable_designs
    allowable_designs['p_chose_B'] = pd.Series(p_chose_B)
    # label rows which are highly predictable
    threshold = 0.01  # TODO: Tom used a lower threshold of 0.005, but that was with epsilon=0
    highly_predictable = (allowable_designs['p_chose_B'] < threshold) | (
        allowable_designs['p_chose_B'] > 1 - threshold)
    allowable_designs['highly_predictable'] = pd.Series(highly_predictable)

    n_not_predictable = allowable_designs.size - \
        sum(allowable_designs.highly_predictable)
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
        logging.warning(
            'not many unpredictable designs, so taking the 10 closest to unpredictable')
        allowable_designs['badness'] = np.abs(
            0.5 - allowable_designs.p_chose_B)
        allowable_designs.sort_values(by=['badness'], inplace=True)
        allowable_designs = allowable_designs[:10]

    allowable_designs.drop(columns=['p_chose_B'])
    logging.debug(
        f'{allowable_designs.shape[0]} designs after removing highly predicted designs')
    return allowable_designs


class DesignSpaceBuilder():
    '''
    A class to generate a design space.
    '''

    def __init__(self,
                 DA=[0.],
                 DB=DEFAULT_DB,
                 RA=list(),
                 RB=[100.],
                 RA_over_RB=list(),
                 IRI=list(),
                 PA=[1.],
                 PB=[1.]):

        self.DA = DA
        self.RA = RA
        self.PA = PA
        self.DB = DB
        self.RB = RB
        self.PB = PB
        self.RA_over_RB = RA_over_RB
        self.IRI = IRI

        self._input_type_validation()
        self._input_value_validation()

    def _input_type_validation(self):
        # NOTE: possibly not very Pythonic
        assert isinstance(self.RA, list), "RA should be a list"
        assert isinstance(self.DA, list), "DA should be a list"
        assert isinstance(self.PA, list), "PA should be a list"
        assert isinstance(self.RB, list), "RB should be a list"
        assert isinstance(self.DB, list), "DB should be a list"
        assert isinstance(self.PB, list), "PB should be a list"
        assert isinstance(self.RA_over_RB, list), "RA_over_RB should be a list"
        assert isinstance(self.IRI,
                          list), "IRI should be a list"

        # we expect EITHER values in RA OR values in RA_over_RB
        # assert (not RA) ^ (not RA_over_RB), "Expecting EITHER RA OR RA_over_RB as an"
        if not self.RA:
            assert not self.RA_over_RB is False, "If not providing list for RA, we expect a list for RA_over_RB"

        if not self.RA_over_RB:
            assert not self.RA is False, "If not providing list for RA_over_RB, we expect a list for RA"

    def _input_value_validation(self):
        '''Confirm values of provided design space specs are valid'''
        if np.any((np.array(self.PA) < 0) | (np.array(self.PA) > 1)):
            raise ValueError('Expect all values of PA to be between 0-1')

        if np.any((np.array(self.PB) < 0) | (np.array(self.PB) > 1)):
            raise ValueError('Expect all values of PB to be between 0-1')

        if np.any(np.array(self.DA) < 0):
            raise ValueError('Expecting all values of DA to be >= 0')

        if np.any(np.array(self.DB) < 0):
            raise ValueError('Expecting all values of DB to be >= 0')

        if np.any(np.array(self.IRI) < 0):
            raise ValueError(
                'Expecting all values of IRI to be >= 0')

        if np.any((np.array(self.RA_over_RB) < 0) | (np.array(self.RA_over_RB) > 1)):
            raise ValueError(
                'Expect all values of RA_over_RB to be between 0-1')

    def build(self, assume_discounting=True):
        '''Create a dataframe of all possible designs (one design is one row)
        based upon the set of design variables (RA, DA, PA, RB, DB, PB)
        provided. We do this generation process ONCE. There may be additional
        trial-level processes which choose subsets of all of the possible
        designs. But here, we generate the largest set of designs that we
        will ever consider
        '''

        # Log the raw values to help with debugging
        logging.debug(f'provided RA = {self.RA}')
        logging.debug(f'provided DA = {self.DA}')
        logging.debug(f'provided PA = {self.PA}')
        logging.debug(f'provided RB = {self.RB}')
        logging.debug(f'provided DB = {self.DB}')
        logging.debug(f'provided PB = {self.PB}')
        logging.debug(f'provided RA_over_RB = {self.RA_over_RB}')
        logging.debug(f'provided IRI = {self.IRI}')

        if len(self.IRI) > 1:
            '''
            We have been given IRI values. We want to
            set DB values based on all combinations of DA and
            IRI.

            Example: if we have DA = [7, 14, 30] and IRI
            = [1, 2, 3] then we want all combinations of DA + IRI
            which results in DB = [8, 9, 10, 15, 16, 17, 31, 32, 33]
            '''

            if len(self.DB)>0:
                print('We are expecting values in EITHER IRI OR RB')

            if len(self.RA)>1 or len(self.RB)>1:
                print('You might know what you are doing, but a fixed reward ratio is recommended when using front-end delays')

            column_list = ['RA', 'DA', 'PA', 'RB', 'IRI', 'PB']
            list_of_lists = [self.RA, self.DA, self.PA,
                             self.RB, self.IRI, self.PB]
            all_combinations = list(itertools.product(*list_of_lists))
            D = pd.DataFrame(all_combinations, columns=column_list)
            D['DB'] = D['DA'] + D['IRI']
            D = D.drop(columns=['IRI'])

        elif not self.RA_over_RB:
            '''assuming we are not doing magnitude effect, as this is
            when we normally would be providing RA_over_RB values'''

            # NOTE: the order of the two lists below HAVE to be the same
            column_list = ['RA', 'DA', 'PA', 'RB', 'DB', 'PB']
            list_of_lists = [self.RA, self.DA,
                             self.PA, self.RB, self.DB, self.PB]
            all_combinations = list(itertools.product(*list_of_lists))
            D = pd.DataFrame(all_combinations, columns=column_list)

        elif not self.RA:
            '''now assume we are dealing with magnitude effect'''

            # create all designs, but using RA_over_RB
            # NOTE: the order of the two lists below HAVE to be the same
            column_list = ['RA_over_RB', 'DA', 'PA', 'RB', 'DB', 'PB']
            list_of_lists = [self.RA_over_RB, self.DA,
                             self.PA, self.RB, self.DB, self.PB]
            all_combinations = list(itertools.product(*list_of_lists))
            D = pd.DataFrame(all_combinations, columns=column_list)

            # now we will convert RA_over_RB to RA for each design then remove it
            D['RA'] = D['RB'] * D['RA_over_RB']
            D = D.drop(columns=['RA_over_RB'])

        else:
            logging.error(
                'Failed to work out what we want. Confusion over RA and RA_over_RB')

        logging.debug(f'{D.shape[0]} designs generated initially')

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

        # convert all columns to float64.
        for col_name in D.columns:
            D[col_name] = D[col_name].astype('float64')

        return D


    ''' Define alternate constructors here
    These methods are convenient in order to set up design spaces without
    having to define all the design dimensions individually.
    '''

    @classmethod
    def delay_magnitude_effect(cls):
        return cls(RB=[100, 500, 1_000],
                   RA_over_RB=np.linspace(0.05, 0.95, 19).tolist())

    @classmethod
    def delayed_and_risky(cls):
        return cls(DA=[0.], DB=DEFAULT_DB,
                   PA=[1.], PB=[0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99],
                   RA=list(100*np.linspace(0.05, 0.95, 91)), RB=[100.])

    @classmethod
    def delayed(cls):
        return cls(RA=list(100*np.linspace(0.05, 0.95, 91)))

    @classmethod
    def frontend_delay(cls):
        '''Defaults for a front-end delay experiment. These typically use a
        fixed reward ratio.
        - IRI = RA+RB'''
        return cls(RA=[50.], RB=[100.],
                   DA=[0., 7, 30, 30*3, 30*6, 365, 365*5],
                   DB=[],
                   IRI=[1, 7, 14, 30, 30*3, 30*6])

    @classmethod
    def risky(cls):
        prob_list = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9]
        return cls(DA=[0], DB=[0], PA=[1], PB=prob_list,
                   RA=list(100*np.linspace(0.05, 0.95, 91)), RB=[100])
