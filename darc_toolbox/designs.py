from badapted.designs import DesignGeneratorABC
from darc_toolbox import Prospect, Design
import pandas as pd
import numpy as np
from badapted.optimisation import design_optimisation
import copy
import logging
import itertools
import time
import random
from scipy.stats import multivariate_normal


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


# CONCRETE BAD CLASSES BELOW --------------------------------------------------

class BayesianAdaptiveDesignGeneratorDARC(DesignGeneratorABC):
    '''
    This class selects the next design to run, based on a provided design
    space, a model, and a design/response history.
    '''

    def __init__(self, design_space,
                 max_trials=20,
                 allow_repeats=True,
                 penalty_function_option='default',
                 λ=2):
        super().__init__()

        self.all_possible_designs = design_space
        self.max_trials = max_trials
        self.allow_repeats = allow_repeats
        self.penalty_function_option = penalty_function_option
        # extract design variables as a list
        self.design_variables = list(design_space.columns.values)
        self.λ = λ  # penalty factor for _default_penalty_func()

    def get_next_design(self, model):

        if self.trial > self.max_trials - 1:
            return None
        start_time = time.time()
        logging.info(f'Getting design for trial {self.trial}')

        allowable_designs = copy.copy(self.all_possible_designs)
        logging.debug(f'{allowable_designs.shape[0]} designs initially')

        # Refine the design space
        allowable_designs = self._refine_design_space(model, allowable_designs)

        # Some checks to see if we have been way too aggressive
        # in refining the design space
        if allowable_designs.shape[0] == 0:
            logging.error(f'No ({allowable_designs.shape[0]}) designs left')

        if allowable_designs.shape[0] < 10:
            logging.warning(
                f'Very few ({allowable_designs.shape[0]}) designs left')

        # process penalty_function_option provided by user upon object
        # construction
        if self.penalty_function_option is 'default':
            # penalty_func = lambda d: self._default_penalty_func(d)
            def penalty_func(d): return self._default_penalty_func(d, λ=self.λ)
        elif self.penalty_function_option is None:
            penalty_func = None

        # Run the low level design optimisation code ~~~~~~~~~~~~~~~~~~~~~~~~~~
        chosen_design_df, _ = design_optimisation(allowable_designs,
                                                  model.predictive_y,
                                                  model.θ,
                                                  n_steps=50,
                                                  penalty_func=penalty_func)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        chosen_design_named_tuple = df_to_design_tuple(chosen_design_df)

        logging.debug(f'chosen design is: {chosen_design_named_tuple}')
        logging.info(
            f'get_next_design() took: {time.time()-start_time:1.3f} seconds')
        return chosen_design_named_tuple

    def _refine_design_space(self, model, allowable_designs):
        '''A series of operations to refine down the space of designs which we
        do design optimisations on.'''

        # Remove already run designs, if appropriate
        if not self.allow_repeats and self.trial > 1:
            # strip the responses off of the stored data
            designs_to_exclude = self.data.df.drop(columns=['R'])

            # there is no convenient set difference function for pandas, but
            # this achieves what we want
            allowable_designs = allowable_designs.loc[~allowable_designs.isin(
                designs_to_exclude.to_dict(orient="list")).all(axis=1), :]

        # Remove highly preductable designs
        allowable_designs = _remove_highly_predictable_designs(allowable_designs,
                                                               model)

        return allowable_designs

    def _get_sorted_unique_design_vals(self, design_variable):
        '''return sorted set of unique design values for the given design
        variable
        TODO: refactor this so it only gets called ONCE
        '''
        all_values = self.all_possible_designs[design_variable].values
        return np.sort(np.unique(all_values))

    def _convert_to_ranks(self, input_designs):
        '''Convert each design variable to a quantile in the set of allowed
        values for that variable, with the minimum and maximum
        taken to be 0 and 1 respectively.

        INPUT:
        - input_designs is a subset of rows of self.all_possible_designs
        OUTPUT:
        - design_ranks = ?????
        '''

        design_variable_names = list(input_designs.columns.values)

        # convert input_designs to martrix in the right dtype -----------------
        # input_designs = input_designs.to_numpy(dtype='float64')  # desired
        input_designs = input_designs.values  # for backward compatibility

        design_ranks = np.full_like(input_designs, np.nan)

        for n, design_variable in enumerate(design_variable_names):
            # Get set of instances this design variable can take
            all_designs_this_variable = self._get_sorted_unique_design_vals(design_variable)
            N_this = len(all_designs_this_variable)

            if N_this is 1:
                design_ranks[:, n] = 0.5
            else:
                # Interpolate input_designs so they are a value between 0 and 1
                # based on a linear regression from possible values of design
                # variables to their scaled ranks (where the later is just
                # (0:(N_this-1))'/(N_this-1))
                design_ranks[:, n] = np.interp(input_designs[:, n],
                                               all_designs_this_variable,
                                               np.linspace(0., 1., num=N_this))

        return design_ranks

    def _default_penalty_func(self, candidate_designs, base_sigma=1, λ=2):
        # get previous designs -----------------------------------
        # TODO: refactor so there is a simple getter for just the designs,
        # not the responses
        # This will get previous designs AND responses, then just get design
        # variables
        previous_designs = self.get_df()[self.design_variables]
        # --------------------------------------------------------

        n_previous_designs = previous_designs.shape[0]
        if n_previous_designs is 0:
            # If there are no previous designs, we shouldn't apply any factors
            penalty_factors = np.ones(candidate_designs.shape[0])
            return penalty_factors

        # To keep problem self similarity, we apply the kernel in the space of
        # ranks.
        # Note: output to self._convert_to_ranks() is DataFrame, output is
        # numerical array
        candidate_designs = self._convert_to_ranks(candidate_designs)
        previous_designs = self._convert_to_ranks(previous_designs)

        # Though will be eliminated later, this is useful for doing the
        # rescaling of p without having to write everything twice.
        candidate_designs = np.concatenate((candidate_designs,
                                            previous_designs),
                                           axis=0)

        # Calculate density of each candidate point under a Gaussian
        # distribution centered on each previous design.
        nD, dD = candidate_designs.shape
        # Reduce standard deviation as we get more designs
        sigma = base_sigma/n_previous_designs
        if np.isscalar(sigma):
            sigma = sigma**2*np.eye(dD)
        else:
            sigma = np.diagflat(sigma**2)

        # Calculate the density of each candidate design using a Gaussian
        # centered at each previous point with covariance sigma
        densities = np.empty((nD, n_previous_designs)) # <----- IS THIS CORRECT SIZE?

        for n in range(n_previous_designs):
            densities[:, n] = multivariate_normal.pdf(candidate_designs,
                                                      mean=previous_designs[n, :],
                                                      cov=sigma)

        # p is a vector, one entry for each designs. It represents kernel
        # density estimate of previous design points by averaging over a
        # mixture of Gaussians centered at each previous design
        p = np.mean(densities, axis=1)
        p = p/np.max(p)  # Rescale so all p are between 0 and 1

        # Remove previous designs
        p = p[:-n_previous_designs]

        # Calculate final penalty factors
        penalty_factors = 1.0/(1.0+λ*p)

        # penalty_factors should be a 1D vector
        penalty_factors = np.squeeze(penalty_factors)
        return penalty_factors


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
    ''' Eliminate designs which are highly predictable as these will not be
    very informative '''

    # grab the design variables. We are going to add some additional columns
    # to the DataFrame in order to do the job, but we want to just grab the
    # design variables after that
    design_variables = allowable_designs.columns.values

    θ_point_estimate = model.get_θ_point_estimate()

    # TODO: CHECK WE CAN EPSILON TO 0
    p_chose_B = model.predictive_y(θ_point_estimate, allowable_designs)
    # add p_chose_B as a column to allowable_designs
    allowable_designs['p_chose_B'] = pd.Series(p_chose_B)

    # Decide which designs (rows) correspond to highly predictable responses
    # threshold = 0.05 means we drop designs with 0>P(y)<0.05 and 0.95<P(y)<1
    threshold = 0.05
    max_threshold = 0.25
    n_not_predictable = 201
    # n_designs_provided = allowable_designs.shape[0]

    while n_not_predictable > 200 and threshold < max_threshold:
        threshold *= 1.05
        highly_predictable, n_predictable, n_not_predictable = _calc_predictability(allowable_designs, threshold)

    # add this as a column in the design DataFrame
    allowable_designs['highly_predictable'] = pd.Series(highly_predictable)

    if n_not_predictable > 10:
        # drop the offending designs (rows)
        allowable_designs = allowable_designs.drop(
            allowable_designs[allowable_designs.p_chose_B < threshold].index)
        allowable_designs = allowable_designs.drop(
            allowable_designs[allowable_designs.p_chose_B > 1 - threshold].index)
        if n_not_predictable > 200:
            allowable_designs = allowable_designs.sample(n=200)
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

    # Grab just the design variables... we don't want any intermediate columns
    # that we added above
    allowable_designs = allowable_designs[design_variables]

    logging.debug(
        f'{allowable_designs.shape[0]} designs after removing highly predicted designs')
    return allowable_designs


def _calc_predictability(allowable_designs, threshold):
    '''calculate how many designes are classified as predictable for a given
    threshold on 'p_chose_B' '''
    highly_predictable = (allowable_designs['p_chose_B'] < threshold) | (
        allowable_designs['p_chose_B'] > 1 - threshold)
    n_predictable = sum(highly_predictable)
    n_not_predictable = allowable_designs.shape[0] - n_predictable
    return (highly_predictable, n_predictable, n_not_predictable)


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

        # eliminate any designs where DA>DB, because by convention ProspectB is
        # our more delayed reward
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
        inter-reward interval = IRI = DA+DB
        DA and IRI values taken from:
        Green, L., Fristoe, N., & Myerson, J. (1994). Temporal discounting
        and preference reversals in choice between delayed outcomes.
        Psychonomic Bulletin & Review, 1(3), 383–389.
        http://doi.org/10.3758/BF03213979
        '''
        return cls(RA=[100.], RB=[250.],
                   DA=[0, 7, 7*2, 30, 30*6, 365, 365*2, 365*3, 365*5,
                       365*7, 365*10, 365*12, 365*15, 365*17, 365*20],
                   DB=[],
                   IRI=[7, 30, 30*3, 30*6, 365, 365*3, 365*5, 365*7,
                        365*10, 365*15, 365*20])

    @classmethod
    def risky(cls):
        prob_list = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9]
        return cls(DA=[0], DB=[0], PA=[1], PB=prob_list,
                   RA=list(100*np.linspace(0.05, 0.95, 91)), RB=[100])
