import darc
import pandas as pd
import numpy as np
import copy
from scipy.stats import norm
import random


def parameter_recovery_sweep(sweep_θ_true, model, design_thing, target_param_name):
    print('starting parameter recovery sweep')
    rows, _ = sweep_θ_true.shape
    summary_stats_final_trial = []
    summary_stats_all_trials = []
    for row in range(rows):
        # make local copies
        local_model = copy.copy(model)
        local_design_thing = copy.deepcopy(design_thing)
        # set true parameter values
        local_model.θ_true = sweep_θ_true.loc[[row]]

        fitted_model, summary_stats = simulated_experiment_trial_loop(local_design_thing, local_model)

        # get summary stats for the parameter of interest
        θ_estimated = fitted_model.get_θ_summary_stats(target_param_name)
        summary_stats_final_trial.append(θ_estimated)

        summary_stats_all_trials.append(summary_stats)

    return (pd.concat(summary_stats_final_trial), summary_stats_all_trials)



def simulated_experiment_trial_loop(design_thing, fitted_model, response_model=None, track_this_parameter='logk'):
    '''run a simulated experiment trial loop
    If we provide an optional response_model then we use that in order to generate
    response data. This allows responses to come from one model and the fitted model to
    be another type of model. This allows us to examine model misspecification.
    However, if there is no response_model provided, then we generate data and fit with
    the same model.'''

    if response_model is None:
        response_model = fitted_model

    if response_model.θ_true is None:
        raise ValueError('response_model must have θ_true values set')

    if track_this_parameter is not None:
        # first row of summary_stats will represent the prior
        summary_stats = fitted_model.get_θ_summary_stats(track_this_parameter)

    for trial in range(666):

        design = design_thing.get_next_design(fitted_model)

        if design is None:
            break

        design_df = darc.single_design_tuple_to_df(design)
        response = response_model.simulate_y(design_df)

        design_thing.enter_trial_design_and_response(design, response)

        fitted_model.update_beliefs(design_thing.get_df())

        if track_this_parameter is not None:
            # add another row to summary_stats
            summary_stats = summary_stats.append(fitted_model.get_θ_summary_stats(track_this_parameter),
                                                ignore_index=True)
    if track_this_parameter is None:
        summary_stats = None

    return fitted_model, summary_stats


def simulated_multi_experiment(design_thing, models_to_fit, response_model):
    '''Simulate an experiment where we have one response model, but have mulitple models
    which we do parameter estimation with and get designs from.'''

    if response_model.θ_true is None:
        raise ValueError('response_model must have θ_true values set')

    n_models = len(models_to_fit)

    for trial in range(666):

        # get the design from a random model
        m = random.randint(0,n_models-1)
        design = design_thing.get_next_design(models_to_fit[m])
        if design is None:
            break

        print(f'trial {trial}, design from model: {m}')

        # get response from response model
        design_df = darc.single_design_tuple_to_df(design)
        response = response_model.simulate_y(design_df)
        design_thing.enter_trial_design_and_response(design, response)

        # update beliefs of all models
        [model.update_beliefs(design_thing.get_df()) for model in models_to_fit]

    return models_to_fit, design_thing
