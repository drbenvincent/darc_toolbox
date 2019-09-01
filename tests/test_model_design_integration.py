import sys

sys.path.insert(0, "/Users/benjamv/git-local/badapted")

from darc_toolbox.delayed import models
from darc_toolbox.designs import BayesianAdaptiveDesignGeneratorDARC, DesignSpaceBuilder
import numpy as np
import pandas as pd
import logging
import darc_toolbox
from darc_toolbox.delayed import models as delayed_models
from darc_toolbox.risky import models as risky_models
from darc_toolbox.delayed_and_risky import models as delayed_and_risky_models
import pytest


logging.basicConfig(
    filename="test.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(funcName)s:%(message)s",
)


delayed_models_list = [
    delayed_models.Hyperbolic,
    delayed_models.Exponential,
    delayed_models.MyersonHyperboloid,
]

delayed_models_list_ME = [
    delayed_models.HyperbolicMagnitudeEffect,
    delayed_models.ExponentialMagnitudeEffect,
]

risky_models_list = [
    risky_models.Hyperbolic,
    risky_models.ProportionalDifference,
    risky_models.LinearInLogOdds,
]

delayed_and_risky_models_list = [delayed_and_risky_models.MultiplicativeHyperbolic]

max_trials = 2
n_particles = 2_000  # needs to be highish to ensure reliable test outcome


# HELPER FUNCTION -------------------------------------------------------


def simulated_experiment_trial_loop(design_thing, model):
    """run a simulated experiment trial loop"""
    for trial in range(666):
        design = design_thing.get_next_design(model)

        if design is None:
            break

        design_df = darc_toolbox.single_design_tuple_to_df(design)
        response = model.simulate_y(design_df)
        design_thing.enter_trial_design_and_response(design, response)

        model.update_beliefs(design_thing.data)

        logging.info(f"Trial {trial} complete")


# TESTS ------------------------------------------------------------------


@pytest.mark.parametrize("model", delayed_models_list)
def test_model_design_integration_delayed(model):
    """Tests integration of model and design. Basically conducts Parameter
    Estimation"""

    D = DesignSpaceBuilder(RA=list(100 * np.linspace(0.05, 0.95, 19))).build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D, max_trials=max_trials)

    model = model(n_particles=n_particles)
    model = model.generate_faux_true_params()

    simulated_experiment_trial_loop(design_thing, model)


@pytest.mark.parametrize("model", delayed_models_list_ME)
def test_model_design_integration_delayed_ME(model):
    """Tests integration of model and design. Basically conducts Parameter
    Estimation"""

    D = DesignSpaceBuilder(
        RB=[100.0, 500.0, 1_000.0], RA_over_RB=np.linspace(0.05, 0.95, 19).tolist()
    ).build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D, max_trials=max_trials)

    model = model(n_particles=n_particles)
    model = model.generate_faux_true_params()

    simulated_experiment_trial_loop(design_thing, model)


@pytest.mark.parametrize("model", risky_models_list)
def test_model_design_integration_risky(model):
    """Tests integration of model and design. Basically conducts Parameter
    Estimation"""

    D = DesignSpaceBuilder(
        DA=[0.0],
        DB=[0.0],
        PA=[1.0],
        PB=list(np.linspace(0.01, 0.99, 91)),
        RA=list(100 * np.linspace(0.05, 0.95, 19)),
        RB=[100.0],
    ).build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D, max_trials=max_trials)

    model = model(n_particles=n_particles)
    model = model.generate_faux_true_params()

    simulated_experiment_trial_loop(design_thing, model)


@pytest.mark.parametrize("model", delayed_and_risky_models_list)
def test_model_design_integration_delayed_and_risky(model):
    """Tests integration of model and design. Basically conducts Parameter
    Estimation"""

    D = DesignSpaceBuilder(
        RA=list(100 * np.linspace(0.05, 0.95, 91)), PB=list(np.linspace(0.01, 0.99, 19))
    ).build()
    design_thing = BayesianAdaptiveDesignGeneratorDARC(D, max_trials=max_trials)

    model = model(n_particles=n_particles)
    model = model.generate_faux_true_params()

    simulated_experiment_trial_loop(design_thing, model)
