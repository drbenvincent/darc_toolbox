from abc import ABC, abstractmethod
from collections import namedtuple
from bad.designs import DesignGeneratorABC
import pandas as pd
import numpy as np
import itertools
from bad.optimisation import design_optimisation
import matplotlib.pyplot as plt
import copy
import logging
import time
import random


DEFAULT_X = np.array([1, 2, 5, 10, 15, 30, 45]).tolist()

# define useful data structures
Design = namedtuple('Design', ['x'])

# helper functions

def design_tuple_to_df(design):
    ''' Convert the named tuple into a 1-row pandas dataframe'''
    trial_data = {'x': design.x}
    return pd.DataFrame(trial_data)


def df_to_design_tuple(df):
    ''' Convert 1-row pandas dataframe into named tuple'''
    x = df.x.values[0]
    chosen_design = Design(x=x)
    return chosen_design


# CONCRETE BAD CLASSES BELOW -----------------------------------------------------------------
class BayesianAdaptiveDesignGeneratorPsychometric(DesignGeneratorABC):
    '''
    A Bayesian Adaptive Design class for estimation of Psychometric functions
    '''

    def __init__(self, x=DEFAULT_X, max_trials=50, allow_repeats=True):
        super().__init__()

        self._input_type_validation(x)

        self._x = x
        self.max_trials = max_trials
        self.allow_repeats = allow_repeats


    def get_next_design(self, model):

        if self.trial > self.max_trials - 1:
            return None
        start_time = time.time()
        logging.info(f'Getting design for trial {self.trial}')

        allowable_designs = copy.copy(self._x)
        logging.debug(f'{allowable_designs.shape[0]} designs initially')

        allowable_designs = _remove_highly_predictable_designs(allowable_designs,
                                                               model)

        chosen_design_df, _ = design_optimisation(allowable_designs,
                                                  model.predictive_y,
                                                  model.θ)
        chosen_design_named_tuple = df_to_design_tuple(chosen_design_df)

        logging.debug(f'chosen design is: {chosen_design_named_tuple}')
        logging.info(
            f'get_next_design() took: {time.time()-start_time:1.3f} seconds')
        return chosen_design_named_tuple

    def _input_type_validation(self, x):
        # NOTE: possibly not very Pythonic
        assert isinstance(x, list), "x should be a list"
        return

    def _refine_design_space(self, model, allowable_designs):
        '''A series of filter operations to refine down the space of designs which we
        do design optimisations on.'''

        if not self.allow_repeats and self.trial > 1:
            allowable_designs = _remove_trials_already_run(
                allowable_designs, self.data.df.drop(columns=['R']))  # TODO: resolve this

        if allowable_designs.shape[0] == 0:
            logging.error(f'No ({allowable_designs.shape[0]}) designs left')

        if allowable_designs.shape[0] < 10:
            logging.warning(
                f'Very few ({allowable_designs.shape[0]}) designs left')

        return allowable_designs


def _remove_trials_already_run(design_set, exclude_these):
    '''Take in a set of designs (design_set) and remove aleady run trials (exclude_these)
    Dropping duplicates will work in this situation because `exclude_these` is going to
    be a subset of `design_set`'''
    # see https://stackoverflow.com/a/40209800/5172570
    allowable_designs = pd.concat(
        [design_set, exclude_these]).drop_duplicates(keep=False)
    logging.debug(
        f'{allowable_designs.shape[0]} designs after removing prior designs')
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




# TODO: Impliment heuristic design generators here, such as staircase methods.
