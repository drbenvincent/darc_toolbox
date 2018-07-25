from darc.delayed import models
from darc.designs import Kirby2009, Frye, BAD_delayed_choices
import numpy as np

# CHOSE THE DESIGN METHOD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
design_thing = BAD_delayed_choices(max_trials=20)
#design_thing = Frye(DB=[7, 30, 30*6, 365], trials_per_delay=7)

model = models.Hyperbolic(n_particles=10_000)  # was 50_000

# set true model parameters, as a dataframe
import pandas as pd
model.θ_true = pd.DataFrame.from_dict({'logk': [np.log(1/100)], 'α': [2]})

for trial in range(666):
    design = design_thing.get_next_design(model)
    
    if design is None:
        break

    # simulated response
    response = model._get_simulated_response(design)

    design_thing.enter_trial_design_and_response(design, response)

    # update beliefs
    model.update_beliefs(design_thing.all_data)

    print(f'trial {trial} complete 🙂')


print('Parameter recovery completed: 😀 ✅')

model.export_posterior_histograms('zzz')
design_thing.plot_all_data('zzz')

