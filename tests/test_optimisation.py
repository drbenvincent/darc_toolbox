import unittest
from darc.delayed import models
from darc.designs import Kirby2009, Frye, DARC_Designs


class TestOptimisation(unittest.TestCase):
    
    def test_model_init(self):
        design_thing = DARC_Designs()
        model = models.Hyperbolic(n_particles=5_000)  # was 50_000
        design = design_thing.get_next_design(model)
        print(design)
