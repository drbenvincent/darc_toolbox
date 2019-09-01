import sys

sys.path.insert(0, "/Users/benjamv/git-local/badapted")

import pandas as pd
import numpy as np
import pytest
from darc_toolbox.delayed import models as delayed_models
from darc_toolbox.risky import models as risky_models
from darc_toolbox.delayed_and_risky import models as delayed_and_risky_models
from darc_toolbox import Prospect, Design
from scipy.stats import norm, expon


delayed_models_list = [
    delayed_models.Hyperbolic,
    delayed_models.Exponential,
    delayed_models.HyperbolicMagnitudeEffect,
    delayed_models.ExponentialMagnitudeEffect,
    delayed_models.ModifiedRachlin,
    delayed_models.MyersonHyperboloid,
    # delayed_models.HyperbolicNonLinearUtility
]

risky_models_list = [
    risky_models.Hyperbolic,
    risky_models.ProportionalDifference,
    risky_models.LinearInLogOdds,
]

delayed_and_risky_models_list = [delayed_and_risky_models.MultiplicativeHyperbolic]


# test model instantiation


@pytest.mark.parametrize(
    "model", delayed_models_list + risky_models_list + delayed_and_risky_models_list
)
def test_model_creation(model):
    n_particles = 10
    model_instance = model(n_particles=n_particles)
    assert isinstance(model_instance, model)


def test_model_creation_custom_prior():
    # just a test of one model as each model has different parameter names
    n_particles = 10
    prior = {"logk": norm(loc=1, scale=1), "α": expon(loc=0, scale=0.04)}
    model_instance = delayed_models.Hyperbolic(n_particles=n_particles, prior=prior)
    assert isinstance(model_instance, delayed_models.Hyperbolic)


# test predictive_y() method of model classes ==========


@pytest.mark.parametrize(
    "model", delayed_models_list + risky_models_list + delayed_and_risky_models_list
)
def test_predictive_y(model):
    n_particles = 10
    model_instance = model(n_particles=n_particles)

    faux_design = pd.DataFrame(
        {
            "RA": [100.0],
            "DA": [0.0],
            "PA": [1.0],
            "RB": [150.0],
            "DB": [14.0],
            "PB": [1.0],
        }
    )
    dv = model_instance._calc_decision_variable(model_instance.θ, faux_design)
    assert isinstance(dv, np.ndarray)


# tests to confirm that we can update beliefs

# THIS IS NO LONGER HOW UPDATING OF data WORKS: NEED TO UPDATE THIS TEST
# @pytest.mark.parametrize("model", delayed_models_list + risky_models_list + delayed_and_risky_models_list)
# def test_update_beliefs(model):
#     # set up model
#     n_particles = 100
#     model_instance = model(n_particles=n_particles)
#     # set up faux data
#     data_columns = ['RA', 'DA', 'PA', 'RB', 'DB', 'PB', 'R']
#     data = pd.DataFrame(columns=data_columns)
#     faux_trial_data = {'RA': [100.], 'DA': [0.], 'PA': [1.],
#                        'RB': [160.], 'DB': [60.], 'PB': [1.],
#                        'R': [int(False)]}
#     data = data.append(pd.DataFrame(faux_trial_data))
#     model_instance.update_beliefs(data)
#     # basically checking model_instance is a model and that we've not errored by this point
#     assert isinstance(model_instance, model)


@pytest.mark.parametrize(
    "model", delayed_models_list + risky_models_list + delayed_and_risky_models_list
)
def test_generate_faux_true_params(model):
    model_instance = model(n_particles=30)
    model_instance = model_instance.generate_faux_true_params()
    isinstance(model_instance.θ_true, dict)


@pytest.mark.parametrize(
    "model", delayed_models_list + risky_models_list + delayed_and_risky_models_list
)
def test_simulate_y(model):
    # set up model
    n_particles = 100
    model_instance = model(n_particles=n_particles)
    model_instance = model_instance.generate_faux_true_params()

    faux_design = pd.DataFrame(
        {
            "RA": [100.0],
            "DA": [0.0],
            "PA": [1.0],
            "RB": [150.0],
            "DB": [14.0],
            "PB": [1.0],
        }
    )

    response = model_instance.simulate_y(faux_design)
    isinstance(response, bool)

