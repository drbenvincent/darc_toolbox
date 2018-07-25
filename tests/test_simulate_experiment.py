import unittest
import pandas as pd
import numpy as np
import models_delay_discounting as models

from adaptive_experiment import make_Frye_generator

class TestSimulateExperiment(unittest.TestCase):
    
    def test_simulation(self):
        model = models.Hyperbolic(n_particles=500)

        # create empty data
        data_columns = ['RA', 'DA', 'RB', 'DB', 'R']
        all_data = pd.DataFrame(columns=data_columns)

        # deciding some experiment options
        delays = np.array([7, 30, 365])
        trials_per_delay = 5
        max_trials = delays.size * trials_per_delay

        # create the generator
        design_generator = make_Frye_generator(
            DB_vec=delays, RB=100, trials_per_delay=trials_per_delay)
        last_response_chose_delayed = None

        for _ in range(max_trials):
            # run simulated trial here ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # generate stimulus
            DA, RA, DB, RB = design_generator.send(last_response_chose_delayed)

            chosen_design = {'RA': [RA], 'DA': [DA], 'RB': [RB], 'DB': [DB]}
            chosen_design = pd.DataFrame(chosen_design)

            print(chosen_design)
            last_response_chose_delayed = model.get_simulated_response(
                chosen_design)

            # append it
            trial_data = {'RA': [RA], 'DA': [DA], 'RB': [RB], 'DB': [DB],
                'R': [int(last_response_chose_delayed)]}
            all_data = all_data.append(pd.DataFrame(trial_data))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            model.update_beliefs(all_data)
