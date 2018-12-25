import logging
from darc.designs import DesignGeneratorABC
from darc import Prospect, Design


class Griskevicius2011(DesignGeneratorABC):
    '''
    A class to provide designs from the Griskevicius et al (2011) risky
    choice task.

    Griskevicius, V., Tybur, J. M., Delton, A. W., & Robertson, T. E. (2011).
    The influence of mortality and socioeconomic status on risk and delayed
    rewards: A life history theory approach. Journal of Personality and
    Social Psychology, 100(6), 1015–26. http://doi.org/10.1037/a0022403
    '''

    # NOTE: these should probably not be class attributes, but declared in the __init__
    # only likely to be a problem if we have mulitple instances. We'd
    # also have to explicitly call the superclass constructor at that point, I believe.
    max_trials = 7
    _RA = [100, 200, 300, 400, 500, 600, 700]
    _DA = 0
    _PA = 1
    _RB = 800
    _DB = 0
    _PB = 0.5

    def get_next_design(self, _):
        # NOTE: This is un-Pythonic as we are asking permission... we should just do it, and have a catch ??
        if self.trial < self.max_trials - 1:
            logging.info(f'Getting design for trial {self.trial}')
            design = Design(ProspectA=Prospect(reward=self._RA[self.trial], delay=self._DA, prob=self._PA),
                            ProspectB=Prospect(reward=self._RB, delay=self._DB, prob=self._PB))
            return design
        else:
            return None


class DuGreenMyerson2002(DesignGeneratorABC):
    '''
    A class to provide designs based on the Du et al (2002) protocol.

    Du, W., Green, L., & Myerson, J. (2002). Cross-cultural comparisons
    of discounting delayed and probabilistic rewards. The Psychological
    Record.
    '''

    def __init__(self, PB=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95], RB=100., trials_per_delay=7):
        self._DA = 0
        self._DB = 0
        self._RB = RB
        self._R_A = RB * 0.5
        self._trials_per_delay = trials_per_delay
        self._trial_per_delay_counter = 0
        self._delay_counter = 0
        self._PA = 1
        self._PB = PB
        self.max_trials = len(self._DB) * self._trials_per_delay
        # call the superclass constructor
        super().__init__()

    def get_next_design(self, _):
        """return the next design as a tuple of prospects"""

        if self._delay_counter == len(self._DB):
            return None

        logging.info(f'Getting design for trial {self.trial}')
        last_response_chose_B = self.get_last_response_chose_B()

        if self._trial_per_delay_counter is 0:
            self._RA = self._RB * 0.5
        else:
            # update RA depending upon last response
            if last_response_chose_B:
                self._RA = self._RA + (self._RB-self._RA)/2
            else:
                self._RA = self._RA - (self._RB-self._RA)/2

        design = Design(ProspectA=Prospect(reward=self._RA, delay=self._DA, prob=self._PA),
                        ProspectB=Prospect(reward=self._RB, delay=self._DB, prob=self._PB[self._delay_counter]))

        self._trial_per_delay_counter += 1
        if self._trial_per_delay_counter > self._trials_per_delay-1:
            # done trials for this delay, so move on to next
            self._delay_counter += 1
            self._trial_per_delay_counter = 0
            self._post_choice_adjustment = 0.25
        return design
