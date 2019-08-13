import sys
sys.path.insert(0, '/Users/btvincent/git-local/darc-experiments-python')

import numpy as np
import darc_toolbox.delayed.designs as delayed_designs
import darc_toolbox.risky.designs as risky_designs
from darc_toolbox.designs import BayesianAdaptiveDesignGeneratorDARC, DesignSpaceBuilder
import pytest


def test_Kirby_default_instantiation():
    design_thing = delayed_designs.Kirby2009()
    assert isinstance(design_thing, delayed_designs.Kirby2009)

def test_Koffarnus_Bickel_default_instantiation():
    design_thing = delayed_designs.Koffarnus_Bickel()
    assert isinstance(design_thing, delayed_designs.Koffarnus_Bickel)


def test_Griskevicius2011_default_instantiation():
    design_thing = delayed_designs.Griskevicius2011()
    assert isinstance(design_thing, delayed_designs.Griskevicius2011)


def test_Griskevicius2011risky_default_instantiation():
    design_thing = risky_designs.Griskevicius2011()
    assert isinstance(design_thing, risky_designs.Griskevicius2011)


def test_Frye_default_instantiation():
    design_thing = delayed_designs.Frye()
    assert isinstance(design_thing, delayed_designs.Frye)


def test_DuGreenMyerson2002_delay_default_instantiation():
    design_thing = delayed_designs.DuGreenMyerson2002()
    assert isinstance(design_thing, delayed_designs.DuGreenMyerson2002)


def test_DuGreenMyerson2002_risky_default_instantiation():
    design_thing = risky_designs.DuGreenMyerson2002()
    assert isinstance(design_thing, risky_designs.DuGreenMyerson2002)


def test_DARCDesign_default_instantiation():
    D = DesignSpaceBuilder(RA=list(100*np.linspace(0.05, 0.95, 91))).build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D)
    assert isinstance(design_thing, BayesianAdaptiveDesignGeneratorDARC)


# below we test our ability to create design objects with various
# options

def test_Frye_custom1_instantiation():
    design_thing = delayed_designs.Frye(DB=[7., 30, 30*6, 365], trials_per_delay=7)
    assert isinstance(design_thing, delayed_designs.Frye)


def test_DARCDesign_delay_instantiation():
    D = DesignSpaceBuilder(RA=list(100*np.linspace(0.05, 0.95, 91)),
                           RB=[100.]).build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D, max_trials=3)
    assert isinstance(design_thing, BayesianAdaptiveDesignGeneratorDARC)


def test_DARCDesign_delay_magnitude_effect_instantiation():
    '''When we are investigating the magnitide effect, we want to ask for a
    reasonable range of DB values. When we do this, we are going to provide
    a vector of proportions (RA_over_RB) which will be translated into
    actual RA values. '''
    D = DesignSpaceBuilder(RB=[10., 100., 1_000.],
                           RA_over_RB=np.linspace(0.05, 0.95, 19).tolist()).build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D, max_trials=3)
    assert isinstance(design_thing, BayesianAdaptiveDesignGeneratorDARC)


def test_DARCDesign_risky_instantiation():
    D = DesignSpaceBuilder(DA=[0.], DB=[0.], PA=[1.],
                           PB=[0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99],
                           RA=list(100*np.linspace(0.05, 0.95, 91)),
                           RB=[100.]).build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D, max_trials=3)
    assert isinstance(design_thing, BayesianAdaptiveDesignGeneratorDARC)

def test_DARCDesign_delayed_and_risky_instantiation():
    D = DesignSpaceBuilder(DA=[0.], DB=[7., 30, 30*6, 365], PA=[1.],
                           PB=[0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99],
                           RA=list(100*np.linspace(0.05, 0.95, 91)),
                           RB=[100.]).build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D, max_trials=3)
    assert isinstance(design_thing, BayesianAdaptiveDesignGeneratorDARC)

# test using the alternate constructors
def test_DARC_BAD_alt_delaymag():
    D = DesignSpaceBuilder.delay_magnitude_effect().build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D, max_trials=3)
    assert isinstance(design_thing, BayesianAdaptiveDesignGeneratorDARC)


def test_DARC_BAD_alt_delayandrisky():
    D = DesignSpaceBuilder.delayed_and_risky().build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D, max_trials=3)
    assert isinstance(design_thing, BayesianAdaptiveDesignGeneratorDARC)


def test_DARC_BAD_alt_delayed():
    D = DesignSpaceBuilder.delayed().build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D, max_trials=3)
    assert isinstance(design_thing, BayesianAdaptiveDesignGeneratorDARC)


def test_DARC_BAD_alt_risky():
    D = DesignSpaceBuilder.risky().build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D, max_trials=3)
    assert isinstance(design_thing, BayesianAdaptiveDesignGeneratorDARC)


def test_DARC_BAD_alt_frontenddelay():
    D = DesignSpaceBuilder.frontend_delay().build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D, max_trials=3)
    assert isinstance(design_thing, BayesianAdaptiveDesignGeneratorDARC)




# a similar set of tests to above, but testing we have some designs


def test_DARCDesign_delay_initial_design_space():
    D = DesignSpaceBuilder(RA=list(100*np.linspace(0.05, 0.95, 91))).build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D)
    n_designs = design_thing.all_possible_designs.shape[0]
    assert n_designs > 10

def test_DARCDesign_delay_magnitude_effect_initial_design_space():
    '''When we are investigating the magnitide effect, we want to ask for a
    reasonable range of DB values. When we do this, we are going to provide
    a vector of proportions (RA_over_RB) which will be translated into
    actual RA values. '''
    D = DesignSpaceBuilder(RB=[10, 100, 1_000],
                           RA_over_RB=np.linspace(0.05, 0.95, 19).tolist()).build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D, max_trials=3)
    n_designs = design_thing.all_possible_designs.shape[0]
    assert n_designs > 10


def test_DARCDesign_risky_initial_design_space():
    D = DesignSpaceBuilder(DA=[0], DB=[0], PA=[1],
                           PB=[0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99],
                           RA=list(100*np.linspace(0.05, 0.95, 91)),
                           RB=[100]).build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D, max_trials=3)
    n_designs = design_thing.all_possible_designs.shape[0]
    assert n_designs>10
