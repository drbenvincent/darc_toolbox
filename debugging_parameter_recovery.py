from darc.delayed import models
from darc.designs import Kirby2009, Frye, BAD_delayed_choices
import numpy as np


design_thing = BAD_delayed_choices()
model = models.Hyperbolic(n_particles=10_000)  # was 50_000

# set true model parameters, as a dataframe
import pandas as pd
model.θ_true = pd.DataFrame.from_dict({'logk': [np.log(1/100)], 'α': [2]})

for trial in range(20):
    design = design_thing.get_next_design(model)

    # simulated response
    response = model._get_simulated_response(design)

    design_thing.enter_trial_design_and_response(design, response)

    # update beliefs
    model.update_beliefs(design_thing.all_data)

    print(f'trial {trial} complete 🙂')


model.export_posterior_histograms('zzz')
print('Parameter recovery completed: 😀 ✅')
