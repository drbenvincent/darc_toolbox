# Point Python to the path where we have installed the bad and darc packages
import sys
sys.path.insert(0, '/Users/btvincent/git-local/darc-experiments-python')


import darc
from darc.delayed import models
from darc.designs import Kirby2009, Frye, DARCDesign
import pandas as pd
import numpy as np
from copy import copy


def parameter_recovery_sweep(sweep_θ_true, model, design_thing):
    print('starting parameter recovery sweep')
    
    # TODO: iterate over rows of pandas df
    rows, _ = sweep_θ_true.shape

    θ_estimated_list = []
    
    for row in range(rows):
        
        # for every true set of parameters, make a local copy of the model and design thing
        local_model = copy(model)
        local_design_thing = copy(design_thing)

        # set true parameter values
        local_model.θ_true = sweep_θ_true.loc[[row]]

        # print(f'true parameters: {local_model.θ_true}')
              
        # TODO: append θ_estimated to a new row in a dataframe
        θ_estimated = parameter_recovery(local_design_thing, local_model)
        
        # print(f'estimated parameters: {θ_estimated}')
        
        θ_estimated_list.append(θ_estimated)

    return pd.concat(θ_estimated_list)


def parameter_recovery(design_thing, model):
    model = simulated_experiment_trial_loop(design_thing, model)
    return model.get_θ_point_estimate()


def simulated_experiment_trial_loop(design_thing, model):
    '''run a simulated experiment trial loop'''
    for trial in range(666):
        design = design_thing.get_next_design(model)

        if design is None:
            break
        
        design_df = darc.single_design_tuple_to_df(design)
        response = model.get_simulated_response(design_df)
        design_thing.enter_trial_design_and_response(design, response)

        model.update_beliefs(design_thing.all_data)
    
    return model


# DOWN TO BUSINESS =======================================================
# do param recovery for hyperbolic time discounting

max_trials = 20

# create a design thing
design_thing = DARCDesign(max_trials=max_trials,
                          RA=list(100*np.linspace(0.05, 0.95, 19)),
                          random_choice_dimension='DB')

# create a model
model = models.Hyperbolic(n_particles=5000) 

# set up a dataframe of true parameter combinations to run through
θsweep = pd.DataFrame.from_dict({'logk': [-2, -3, -4], 
                                 'α': [3, 3, 3]})
                                
# do the parameter sweep
θ_estimated = parameter_recovery_sweep(θsweep, model, design_thing)

print(θ_estimated)