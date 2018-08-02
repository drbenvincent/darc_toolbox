# Point Python to the path where we have installed the bad and darc packages
import sys
sys.path.insert(0, '/Users/btvincent/git-local/darc-experiments-python')


import darc
from darc.delayed import models
from darc.designs import Kirby2009, Frye, DARCDesign
import pandas as pd
import numpy as np


def parameter_recovery(θ_true, design_method, model_type, n_particles=10_000, n_trials=20):

    print('parameter recovery')
    design_thing = design_method()
    model = model_type(n_particles=n_particles)
    # set true model parameters, as a dataframe
    model.θ_true = θ_true

    for _ in range(n_trials):
        design = design_thing.get_next_design(model)
        response = model.get_simulated_response( darc.single_design_tuple_to_df(design))
        design_thing.enter_trial_design_and_response(design, response)
        model.update_beliefs(design_thing.all_data)

    return model.get_θ_point_estimate()


def parameter_recovery_sweep(sweep_θ_true, model_type):

    print('starting parameter recovery sweep')

    # TODO: iterate over rows of pandas df
    rows, _ = sweep_θ_true.shape

    for row in range(rows):
        θ_true = sweep_θ_true.loc[[row]]

        # TODO: repeat the parameter recovery a number of times
        # TODO: provide as a generic function to call for the given inputs
        θ_estimated = parameter_recovery(θ_true, 
                                         design_method,
                                         model_type,
                                         n_particles,
                                         n_trials)

        print(θ_true - θ_estimated)



# Run a single parameter recovery
θ_true = pd.DataFrame.from_dict({'logk': [-7], 'α': [3]})

θ_estimated = parameter_recovery(θ_true,
                                 model_type=models.Hyperbolic,
                                 n_trials=10)

print(θ_estimated)
print(θ_true - θ_estimated)

# TODO: define sweep as a list of dicts so we can do list comprehensions

# Run a parameter sweep
# First we create a pandas table, each row is one set of true params for the sweep
sweep_θ_true = pd.DataFrame.from_dict(
    {'logk': np.linspace(-7, -2, 5), 'α': [3, 3, 3, 3, 3]})

parameter_recovery_sweep(sweep_θ_true)






# NOTE: So this is ok, but it requires knowledge of the parameters and their order 
# within the parameter sweep.
# It might be best to set this up ONCE as a dataframe then just process it.

# define params I want
logk_list = np.linspace(-7, -2, 5)
alpha_list = [3, 3, 3, 3, 3]

# do a parameter sweep
for logk, alpha in zip(logk_list, alpha_list):
    param_dict = {'logk': [logk], 'α': [alpha]}
    print(param_dict)
    
    param_df = pd.DataFrame.from_dict(param_dict)
    print(param_df)
