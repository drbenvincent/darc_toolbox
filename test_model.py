import unittest
import pandas as pd
from darc.delayed import models


class TestModel(unittest.TestCase):
    
    def test_model_init(self):
        _ = models.Hyperbolic(n_particles=500)
        

    def test_model_update(self):
        model = models.ProportionalDifference(n_particles=500)

        # create data
        data_columns = ['RA', 'DA', 'RB', 'DB', 'R']
        all_data = pd.DataFrame(columns=data_columns)
        trial_data = {'RA': [100], 'DA': [0], 'PA': [1.],
                      'RB': [110], 'DB': [7], 'PB': [1.],
                      'R': [int(False)]}
        all_data = all_data.append(pd.DataFrame(trial_data))

        # update beliefs
        model.update_beliefs(all_data)


    def test_get_simulated_response(self):
        model = models.Hyperbolic(n_particles=500)

        chosen_design = {'RA': [100], 'DA': [0], 'PA': [1.],
                         'RB': [110], 'DB': [7], 'PB': [1.]}
        chosen_design = pd.DataFrame(chosen_design)
        last_response_chose_delayed = model.get_simulated_response(
            chosen_design)
        print(last_response_chose_delayed)
