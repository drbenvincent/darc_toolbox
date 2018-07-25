from darc.delayed import models
from darc.designs import Kirby2009, Frye, BAD_delayed_choices


design_thing = BAD_delayed_choices()
model = models.Hyperbolic(n_particles=5_000)  # was 50_000
design = design_thing.get_next_design(model)
print(design)
print('SUCESS 😀')