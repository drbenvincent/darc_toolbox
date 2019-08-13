import logging
from darc_toolbox.designs import DesignGeneratorABC
from darc_toolbox import Prospect, Design


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

    def __init__(self, PB=[0.95, 0.9, 0.7, 0.5, 0.3, 0.1, 0.05], RB=100.):
        self._DA = 0
        self._DB = 0
        self._RB = RB
        self._RA = self._RB * 0.5
        # self._post_choice_adjustment = (self._RB - self._RA) * 0.5
        self._trials_per_prob = 6
        self._trial_per_prob_counter = 0
        self._prob_counter = 0
        self._PA = 1
        self._PB = PB
        self.max_trials = len(self._PB) * self._trials_per_prob
        # call the superclass constructor
        super().__init__()

    def get_next_design(self, _):
        """return the next design as a tuple of prospects"""

        if self._prob_counter == len(self._PB):
            return None

        logging.info(f'Getting design for trial {self.trial}')
        last_response_chose_B = self.get_last_response_chose_B()

        if self._trial_per_prob_counter is 0:
            self._RA = self._RB * 0.5
            self._post_choice_adjustment = (self._RB - self._RA) * 0.5
        else:
            # update RA depending upon last response
            if last_response_chose_B:
                self._RA += self._post_choice_adjustment
            else:
                self._RA -= self._post_choice_adjustment

            self._post_choice_adjustment *= 0.5

        design = Design(ProspectA=Prospect(reward=self._RA, delay=self._DA, prob=self._PA),
                        ProspectB=Prospect(reward=self._RB, delay=self._DB, prob=self._PB[self._prob_counter]))

        self._trial_per_prob_counter += 1
        if self._trial_per_prob_counter > self._trials_per_prob-1:
            # done trials for this prob level, so move on to next
            self._prob_counter += 1
            self._trial_per_prob_counter = 0
        return design
