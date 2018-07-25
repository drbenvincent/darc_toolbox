from darc.delayed import models
from darc.designs import Kirby2009, Frye, BAD_delayed_choices


design_thing = BAD_delayed_choices()
model = models.Hyperbolic(n_particles=5_000)  # was 50_000

# set true model parameters, as a dataframe
import pandas as pd
model.θ_true = pd.DataFrame.from_dict({'logk': [-3], 'α': [3]})

for trial in range(3):
    design = design_thing.get_next_design(model)

    # simulated response
    response = model._get_simulated_response(design)

    design_thing.enter_trial_design_and_response(design, response)

    # update beliefs
    model.update_beliefs(design_thing.all_data)

    print(f'trial {trial} complete 🙂')


print('Parameter recovery completed: 😀 ✅')
