import sys
sys.path.insert(0, '/Users/btvincent/git-local/darc-experiments-python')

from collections import namedtuple
import pandas as pd
import numpy as np
import pytest


# # test model creation: just one
# def test_model_creation():
#     model = models.Hyperbolic(n_particles=10)
#     assert isinstance(model, models.Hyperbolic)
#     return model


from darc.delayed import models as delayed_models
from darc.risky import models as risky_models
from darc.delayed_and_risky import models as delayed_and_risky_models


# define useful data structures
Prospect = namedtuple('Prospect', ['reward', 'delay', 'prob'])
Design = namedtuple('Design', ['ProspectA', 'ProspectB'])

delayed_models_list = [
    delayed_models.Hyperbolic,
    delayed_models.Exponential,
    delayed_models.HyperbolicMagnitudeEffect,
    delayed_models.ExponentialMagnitudeEffect,
    delayed_models.ConstantSensitivity,
    delayed_models.MyersonHyperboloid,
    delayed_models.ProportionalDifference,
    delayed_models.HyperbolicNonLinearUtility
]

risky_models_list = [
    risky_models.Hyperbolic,
    risky_models.ProportionalDifference,
    risky_models.ProspectTheory
]

delayed_and_risky_models_list = [
    delayed_and_risky_models.MultiplicativeHyperbolic
]

# TODO: do we need to separate out the experiment types here? Why not one big list of models?


@pytest.mark.parametrize("model", delayed_models_list)
def test_delayed_model_creation(model):
    n_particles = 10
    model_instance = model(n_particles=n_particles)
    assert isinstance(model_instance, model)


@pytest.mark.parametrize("model", risky_models_list)
def test_risky_model_creation(model):
    n_particles = 10
    model_instance = model(n_particles=n_particles)
    assert isinstance(model_instance, model)


@pytest.mark.parametrize("model", delayed_and_risky_models_list)
def test_delayed_and_risky_model_creation(model):
    n_particles = 10
    model_instance = model(n_particles=n_particles)
    assert isinstance(model_instance, model)




@pytest.mark.parametrize("model", delayed_models_list)
def test_delayed_calc_decision_variable(model):
    n_particles = 10
    model_instance = model(n_particles=n_particles)

    faux_data = pd.DataFrame({'RA': [100], 'DA': [0], 'PA': [1],
                              'RB': [150], 'DB': [14], 'PB': [1]})
    dv = model_instance.calc_decision_variable(model_instance.θ, faux_data)
    assert isinstance(dv, np.ndarray)


@pytest.mark.parametrize("model", risky_models_list)
def test_risky_calc_decision_variable(model):
    n_particles = 10
    model_instance = model(n_particles=n_particles)

    faux_data = pd.DataFrame({'RA': [100], 'DA': [0], 'PA': [1],
                              'RB': [150], 'DB': [14], 'PB': [1]})
    dv = model_instance.calc_decision_variable(model_instance.θ, faux_data)
    assert isinstance(dv, np.ndarray)


@pytest.mark.parametrize("model", delayed_and_risky_models_list)
def test_delayed_and_risky_calc_decision_variable(model):
    n_particles = 10
    model_instance = model(n_particles=n_particles)

    faux_data = pd.DataFrame({'RA': [100], 'DA': [0], 'PA': [1],
                              'RB': [150], 'DB': [14], 'PB': [1]})
    dv = model_instance.calc_decision_variable(model_instance.θ, faux_data)
    assert isinstance(dv, np.ndarray)
