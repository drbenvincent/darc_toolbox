import sys
sys.path.insert(0, '/Users/btvincent/git-local/darc-experiments-python')

import numpy as np
from darc.designs import Kirby2009, Frye, DARC_Designs
import pytest


def test_Kirby_default_instantiation():
    design_thing = Kirby2009()
    assert isinstance(design_thing, Kirby2009)


def test_Frye_default_instantiation():
    design_thing = Frye()
    assert isinstance(design_thing, Frye)


def test_DARC_Designs_default_instantiation():
    design_thing = DARC_Designs()
    assert isinstance(design_thing, DARC_Designs)


# below we test our ability to create design objects with various
# options

def test_Frye_custom1_instantiation():
    design_thing = Frye(DB=[7, 30, 30*6, 365], trials_per_delay=7)
    assert isinstance(design_thing, Frye)


def test_DARC_Designs_delay_instantiation():    
    design_thing = DARC_Designs(max_trials=3,
                                RA=list(100*np.linspace(0.05, 0.95, 91)),
                                RB=[100])
    assert isinstance(design_thing, DARC_Designs)


def test_DARC_Designs_delay_magnitude_effect_instantiation():
    # we want more RB values for magnitude effects
    design_thing = DARC_Designs(max_trials=3,
                                RB=[70, 80, 90, 100, 110, 120, 130])
    assert isinstance(design_thing, DARC_Designs)


def test_DARC_Designs_risky_instantiation():
    design_thing = DARC_Designs(max_trials=3,
                                DA=[0], DB=[0], PA=[1],
                                PB=[0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99],
                                RA=list(100*np.linspace(0.05, 0.95, 91)),
                                RB=[100])
    assert isinstance(design_thing, DARC_Designs)
