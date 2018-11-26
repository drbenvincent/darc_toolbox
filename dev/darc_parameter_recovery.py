import darc
from darc.delayed import models
from darc.designs import Kirby2009, Frye, DARCDesign
import pandas as pd
import numpy as np
from copy import copy


def parameter_recovery_sweep(sweep_θ_true, model, design_thing, target_param_name):
    print('starting parameter recovery sweep')
    rows, _ = sweep_θ_true.shape
    θ_estimated_list = []
    for row in range(rows):
        # make local copies
        local_model = copy(model)
        local_design_thing = copy(design_thing)
        # set true parameter values
        local_model.θ_true = sweep_θ_true.loc[[row]]  

        fitted_model = simulated_experiment_trial_loop(local_design_thing, local_model)
        
        # This gets point estimate of all parameters
        #θ_estimated = fitted_model.get_θ_point_estimate()

        # get summary stats for the parameter of interest
        θ_estimated = fitted_model.get_θ_summary_stats(target_param_name)

        θ_estimated_list.append(θ_estimated)

    return pd.concat(θ_estimated_list)



def simulated_experiment_trial_loop(design_thing, model):
    '''run a simulated experiment trial loop'''

    # TODO: Add a check here to confirm that the model has some θ_true

    for trial in range(666):
        design = design_thing.get_next_design(model)

        if design is None:
            break
        
        design_df = darc.single_design_tuple_to_df(design)
        response = model.get_simulated_response(design_df)

        # TODO: DOUBLE CHECK THIS IS ACTUALLY UPDATING STUFF!! OR DO WE NEED TO 
        # SPECIFICALLY UPATE THE design_thing AND model ????
        design_thing.enter_trial_design_and_response(design, response)

        model.update_beliefs(design_thing.all_data)
    
    return model
