from darc.delayed import models
from darc.designs import Kirby2009, Frye, DARC_Designs

def parameter_recovery(θ_true,
                       n_trials=10,
                       design_method=DARC_Designs,
                       model_type=model.Hyperbolic,
                       n_particles=10_000):

    design_thing = design_method()
    model = model_type(n_particles=n_particles)
    # set true model parameters, as a dataframe
    model.θ_true = θ_true

    for trial in range(n_trials):
        design = design_thing.get_next_design(model)
        response = model.get_simulated_response(design)
        design_thing.enter_trial_design_and_response(design, response)
        model.update_beliefs(design_thing.all_data)

    # TODO: return point estimate, or summary of param estimates
    return model


θ_true = pd.DataFrame.from_dict({'logk': [-7], 'α': [3]})

θ_estimated = parameter_recovery(θ_true)

# TODO: add ability to run a parameter sweep

# TODO: add ability to 