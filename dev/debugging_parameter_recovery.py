# Point Python to the path where we have installed the bad and darc packages
import sys
sys.path.insert(0, '/Users/btvincent/git-local/darc-experiments-python')


from darc.delayed import models
from darc.designs import Kirby2009, Frye, DARCDesign
import numpy as np
import logging
import darc


logging.basicConfig(filename='test.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')

# CHOSE THE DESIGN METHOD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
design_thing = DARCDesign(max_trials=5)
#design_thing = Frye(DB=[7, 30, 30*6, 365], trials_per_delay=7)
#design_thing = Kirby2009()


model = models.Hyperbolic(n_particles=1_000)  # was 50_000

# set true model parameters, as a dataframe
import pandas as pd
model.θ_true = pd.DataFrame.from_dict({'logk': [np.log(1/365)], 'α': [2]})

for trial in range(666):
    design = design_thing.get_next_design(model)
    
    if design is None:
        break

    # simulated response
    design_df = darc.single_design_tuple_to_df(design)
    response = model.get_simulated_response(design_df)

    design_thing.enter_trial_design_and_response(design, response)

    # update beliefs
    model.update_beliefs(design_thing.all_data)

    logging.info(f'Trial {trial} complete')


logging.info('Parameter recovery completed: 😀 ✅')
print('Parameter recovery completed: 😀 ✅')

model.export_posterior_histograms('zzz')
design_thing.plot_all_data('zzz')

