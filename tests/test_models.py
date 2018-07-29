import sys
sys.path.insert(0, '/Users/btvincent/git-local/darc-experiments-python')

import pytest


# # test model creation: just one
# def test_model_creation():
#     model = models.Hyperbolic(n_particles=10)
#     assert isinstance(model, models.Hyperbolic)
#     return model


from darc.delayed import models as delayed_models
from darc.risky import models as risky_models
from darc.delayed_and_risky import models as delayed_and_risky_models

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


@pytest.mark.parametrize("model", delayed_models_list)
def test_deayed_model_creation(model):
    model_instance = model(n_particles=10)
    assert isinstance(model_instance, model)


@pytest.mark.parametrize("model", risky_models_list)
def test_risky_model_creation(model):
    model_instance = model(n_particles=10)
    assert isinstance(model_instance, model)


@pytest.mark.parametrize("model", delayed_and_risky_models_list)
def test_delayed_and_risky_model_creation(model):
    model_instance = model(n_particles=10)
    assert isinstance(model_instance, model)
