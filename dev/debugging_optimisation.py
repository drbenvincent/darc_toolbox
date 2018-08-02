from darc.delayed import models
from darc.designs import Kirby2009, Frye, DARCDesign


design_thing = DARCDesign()
model = models.Hyperbolic(n_particles=5_000)  # was 50_000
design = design_thing.get_next_design(model)
print(design)
print('SUCESS 😀')