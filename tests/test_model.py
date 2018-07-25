import unittest
import pandas as pd
from darc.delayed import models
from darc.designs import Kirby2009, Frye, BAD_delayed_choices
        

class TestModel(unittest.TestCase):
    
    def test_model_init(self):
        _ = models.Hyperbolic(n_particles=500)
        

    def test_model_update(self):

        design_thing = Kirby2009()
        model = models.Hyperbolic(n_particles=5000)
        design = design_thing.get_next_design(model)
        # just make a response up
        last_response_chose_delayed = True
        design_thing.enter_trial_design_and_response(
            design, last_response_chose_delayed)
        # finally update beliefs
        model.update_beliefs(design_thing.all_data)


    def test_get_simulated_response(self):
        model = models.Hyperbolic(n_particles=500)

        chosen_design = {'RA': [100], 'DA': [0], 'PA': [1.],
                         'RB': [110], 'DB': [7], 'PB': [1.]}
        chosen_design = pd.DataFrame(chosen_design)
        last_response_chose_delayed = model.get_simulated_response(
            chosen_design)
        print(last_response_chose_delayed)
