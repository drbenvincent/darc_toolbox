# Point Python to the path where we have installed the bad and darc packages
import sys
sys.path.insert(0, '/Users/btvincent/git-local/darc-experiments-python')


from darc.delayed import models
from darc.designs import DARCDesign
import numpy as np
import pandas as pd
import logging
import darc
import pytest


logging.basicConfig(filename='test.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')

# HELPER FUNCTION -------------------------------------------------------
def simulated_experiment_trial_loop(design_thing, model):
    '''run a simulated experiment trial loop'''
    for trial in range(666):
        design = design_thing.get_next_design(model, random_choice_dimension='DB')

        if design is None:
            break
        
        design_df = darc.single_design_tuple_to_df(design)
        response = model.get_simulated_response(design_df)
        design_thing.enter_trial_design_and_response(design, response)

        model.update_beliefs(design_thing.all_data)

        logging.info(f'Trial {trial} complete')


def test_model_design_integration():

    design_thing = DARCDesign(max_trials=5,
                              RA=list(100*np.linspace(0.05, 0.95, 10)))

    model = models.Hyperbolic(n_particles=100) 
    model.θ_true = pd.DataFrame.from_dict({'logk': [np.log(1/365)], 'α': [2]})

    simulated_experiment_trial_loop(design_thing, model)
