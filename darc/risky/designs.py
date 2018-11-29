from collections import namedtuple
import logging
from darc.designs import DARCDesignABC


# define useful data structures
Prospect = namedtuple('Prospect', ['reward', 'delay', 'prob'])
Design = namedtuple('Design', ['ProspectA', 'ProspectB'])


class Griskevicius2011(DARCDesignABC):
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
    RA = [100, 200, 300, 400, 500, 600, 700]
    DA = 0
    PA = 1
    RB = 800
    DB = 0
    PB = 0.5
    
    def get_next_design(self, _):
        # NOTE: This is un-Pythonic as we are asking permission... we should just do it, and have a catch ??
        if self.trial < self.max_trials - 1:
            logging.info(f'Getting design for trial {self.trial}')
            design = Design(ProspectA=Prospect(reward=self.RA[self.trial], delay=self.DA, prob=self.PA),
                            ProspectB=Prospect(reward=self.RB, delay=self.DB, prob=self.PB))
            return design
        else:
            return None
