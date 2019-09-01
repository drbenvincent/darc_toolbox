from badapted.designs import DesignGeneratorABC, BayesianAdaptiveDesignGenerator
from darc_toolbox import Prospect, Design
import pandas as pd
import numpy as np
import logging
import itertools


DEFAULT_DB = np.concatenate(
    [
        np.array([1, 2, 5, 10, 15, 30, 45]) / 24 / 60,
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 12]) / 24,
        np.array([1, 2, 3, 4, 5, 6, 7]),
        np.array([2, 3, 4]) * 7,
        np.array([3, 4, 5, 6, 8, 9]) * 30,
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25]) * 365,
    ]
).tolist()

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


class DARCDesignGenerator(DesignGeneratorABC):
    """This adds DARC specific functionality to the design generator"""

    def __init__(self):
        # super().__init__()
        DesignGeneratorABC.__init__(self)

        # generate empty dataframe
        data_columns = ["RA", "DA", "PA", "RB", "DB", "PB", "R"]
        self.data = pd.DataFrame(columns=data_columns)

    def add_design_response_to_dataframe(self, design, response):
        """
        This method must take in `design` and `reward` from the current trial
        and store this as a new row in self.data which is a pandas data frame.
        """

        # TODO: need to specify types here I think... then life might be
        # easier to decant the data out at another point
        # trial_df = design_to_df(design)
        # self.data = self.data.append(trial_df)

        trial_data = {
            "RA": design.ProspectA.reward,
            "DA": design.ProspectA.delay,
            "PA": design.ProspectA.prob,
            "RB": design.ProspectB.reward,
            "DB": design.ProspectB.delay,
            "PB": design.ProspectB.prob,
            "R": [int(response)],
        }
        self.data = self.data.append(pd.DataFrame(trial_data))
        # a bit clumsy but...
        self.data["R"] = self.data["R"].astype("int64")
        self.data = self.data.reset_index(drop=True)

        # we potentially manually call model to update beliefs here. But so far
        # this is done manually in PsychoPy
        return

    @staticmethod
    def df_to_design_tuple(df):
        """User must impliment this method. It takes in a design in the form of a
        single row of pandas dataframe, and it must return the chosen design as a
        named tuple.
        Convert 1-row pandas dataframe into named tuple"""
        RA = df.RA.values[0]
        DA = df.DA.values[0]
        PA = df.PA.values[0]
        RB = df.RB.values[0]
        DB = df.DB.values[0]
        PB = df.PB.values[0]
        chosen_design = Design(
            ProspectA=Prospect(reward=RA, delay=DA, prob=PA),
            ProspectB=Prospect(reward=RB, delay=DB, prob=PB),
        )
        return chosen_design


class BayesianAdaptiveDesignGeneratorDARC(
    DARCDesignGenerator, BayesianAdaptiveDesignGenerator
):
    """This will be the concrete class for doing Bayesian adaptive design
    in the DARC experiment domain."""

    def __init__(
        self,
        design_space,
        max_trials=20,
        allow_repeats=True,
        penalty_function_option="default",
        λ=2,
    ):

        # call superclass constructors - note that the order of calling these is important
        BayesianAdaptiveDesignGenerator.__init__(
            self,
            design_space,
            max_trials=max_trials,
            allow_repeats=allow_repeats,
            penalty_function_option=penalty_function_option,
            λ=λ,
        )

        DARCDesignGenerator.__init__(self)


class DesignSpaceBuilder:
    """
    A class to generate a design space.
    """

    def __init__(
        self,
        DA=[0.0],
        DB=DEFAULT_DB,
        RA=list(),
        RB=[100.0],
        RA_over_RB=list(),
        IRI=list(),
        PA=[1.0],
        PB=[1.0],
    ):

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
        assert isinstance(self.IRI, list), "IRI should be a list"

        # we expect EITHER values in RA OR values in RA_over_RB
        # assert (not RA) ^ (not RA_over_RB), "Expecting EITHER RA OR RA_over_RB as an"
        if not self.RA:
            assert (
                not self.RA_over_RB is False
            ), "If not providing list for RA, we expect a list for RA_over_RB"

        if not self.RA_over_RB:
            assert (
                not self.RA is False
            ), "If not providing list for RA_over_RB, we expect a list for RA"

    def _input_value_validation(self):
        """Confirm values of provided design space specs are valid"""
        if np.any((np.array(self.PA) < 0) | (np.array(self.PA) > 1)):
            raise ValueError("Expect all values of PA to be between 0-1")

        if np.any((np.array(self.PB) < 0) | (np.array(self.PB) > 1)):
            raise ValueError("Expect all values of PB to be between 0-1")

        if np.any(np.array(self.DA) < 0):
            raise ValueError("Expecting all values of DA to be >= 0")

        if np.any(np.array(self.DB) < 0):
            raise ValueError("Expecting all values of DB to be >= 0")

        if np.any(np.array(self.IRI) < 0):
            raise ValueError("Expecting all values of IRI to be >= 0")

        if np.any((np.array(self.RA_over_RB) < 0) | (np.array(self.RA_over_RB) > 1)):
            raise ValueError("Expect all values of RA_over_RB to be between 0-1")

    def build(self, assume_discounting=True):
        """Create a dataframe of all possible designs (one design is one row)
        based upon the set of design variables (RA, DA, PA, RB, DB, PB)
        provided. We do this generation process ONCE. There may be additional
        trial-level processes which choose subsets of all of the possible
        designs. But here, we generate the largest set of designs that we
        will ever consider
        """

        # Log the raw values to help with debugging
        logging.debug(f"provided RA = {self.RA}")
        logging.debug(f"provided DA = {self.DA}")
        logging.debug(f"provided PA = {self.PA}")
        logging.debug(f"provided RB = {self.RB}")
        logging.debug(f"provided DB = {self.DB}")
        logging.debug(f"provided PB = {self.PB}")
        logging.debug(f"provided RA_over_RB = {self.RA_over_RB}")
        logging.debug(f"provided IRI = {self.IRI}")

        if len(self.IRI) > 1:
            """
            We have been given IRI values. We want to
            set DB values based on all combinations of DA and
            IRI.

            Example: if we have DA = [7, 14, 30] and IRI
            = [1, 2, 3] then we want all combinations of DA + IRI
            which results in DB = [8, 9, 10, 15, 16, 17, 31, 32, 33]
            """

            if len(self.DB) > 0:
                print("We are expecting values in EITHER IRI OR RB")

            if len(self.RA) > 1 or len(self.RB) > 1:
                print(
                    "You might know what you are doing, but a fixed reward ratio is recommended when using front-end delays"
                )

            column_list = ["RA", "DA", "PA", "RB", "IRI", "PB"]
            list_of_lists = [self.RA, self.DA, self.PA, self.RB, self.IRI, self.PB]
            all_combinations = list(itertools.product(*list_of_lists))
            D = pd.DataFrame(all_combinations, columns=column_list)
            D["DB"] = D["DA"] + D["IRI"]
            D = D.drop(columns=["IRI"])

        elif not self.RA_over_RB:
            """assuming we are not doing magnitude effect, as this is
            when we normally would be providing RA_over_RB values"""

            # NOTE: the order of the two lists below HAVE to be the same
            column_list = ["RA", "DA", "PA", "RB", "DB", "PB"]
            list_of_lists = [self.RA, self.DA, self.PA, self.RB, self.DB, self.PB]
            all_combinations = list(itertools.product(*list_of_lists))
            D = pd.DataFrame(all_combinations, columns=column_list)

        elif not self.RA:
            """now assume we are dealing with magnitude effect"""

            # create all designs, but using RA_over_RB
            # NOTE: the order of the two lists below HAVE to be the same
            column_list = ["RA_over_RB", "DA", "PA", "RB", "DB", "PB"]
            list_of_lists = [
                self.RA_over_RB,
                self.DA,
                self.PA,
                self.RB,
                self.DB,
                self.PB,
            ]
            all_combinations = list(itertools.product(*list_of_lists))
            D = pd.DataFrame(all_combinations, columns=column_list)

            # now we will convert RA_over_RB to RA for each design then remove it
            D["RA"] = D["RB"] * D["RA_over_RB"]
            D = D.drop(columns=["RA_over_RB"])

        else:
            logging.error(
                "Failed to work out what we want. Confusion over RA and RA_over_RB"
            )

        logging.debug(f"{D.shape[0]} designs generated initially")

        # eliminate any designs where DA>DB, because by convention ProspectB is
        # our more delayed reward
        D.drop(D[D.DA > D.DB].index, inplace=True)
        logging.debug(f"{D.shape[0]} left after dropping DA>DB")

        if assume_discounting:
            D.drop(D[D.RB < D.RA].index, inplace=True)
            logging.debug(f"{D.shape[0]} left after dropping RB<RA")

        # NOTE: we may want to do further trimming and refining of the possible
        # set of designs, based upon domain knowledge etc.

        # check we actually have some designs!
        if D.shape[0] == 0:
            logging.error(f"No ({D.shape[0]}) designs generated!")

        # convert all columns to float64.
        for col_name in D.columns:
            D[col_name] = D[col_name].astype("float64")

        return D

    """ Define alternate constructors here
    These methods are convenient in order to set up design spaces without
    having to define all the design dimensions individually.
    """

    @classmethod
    def delay_magnitude_effect(cls):
        return cls(
            RB=[100, 500, 1_000], RA_over_RB=np.linspace(0.05, 0.95, 19).tolist()
        )

    @classmethod
    def delayed_and_risky(cls):
        return cls(
            DA=[0.0],
            DB=DEFAULT_DB,
            PA=[1.0],
            PB=[0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99],
            RA=list(100 * np.linspace(0.05, 0.95, 91)),
            RB=[100.0],
        )

    @classmethod
    def delayed(cls):
        return cls(RA=list(100 * np.linspace(0.05, 0.95, 91)))

    @classmethod
    def frontend_delay(cls):
        """Defaults for a front-end delay experiment. These typically use a
        fixed reward ratio.
        inter-reward interval = IRI = DA+DB
        DA and IRI values taken from:
        Green, L., Fristoe, N., & Myerson, J. (1994). Temporal discounting
        and preference reversals in choice between delayed outcomes.
        Psychonomic Bulletin & Review, 1(3), 383–389.
        http://doi.org/10.3758/BF03213979
        """
        return cls(
            RA=[100.0],
            RB=[250.0],
            DA=[
                0,
                7,
                7 * 2,
                30,
                30 * 6,
                365,
                365 * 2,
                365 * 3,
                365 * 5,
                365 * 7,
                365 * 10,
                365 * 12,
                365 * 15,
                365 * 17,
                365 * 20,
            ],
            DB=[],
            IRI=[
                7,
                30,
                30 * 3,
                30 * 6,
                365,
                365 * 3,
                365 * 5,
                365 * 7,
                365 * 10,
                365 * 15,
                365 * 20,
            ],
        )

    @classmethod
    def risky(cls):
        prob_list = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9]
        return cls(
            DA=[0],
            DB=[0],
            PA=[1],
            PB=prob_list,
            RA=list(100 * np.linspace(0.05, 0.95, 91)),
            RB=[100],
        )

