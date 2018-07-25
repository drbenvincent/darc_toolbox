from darc.delayed import models
from darc.designs import Kirby2009, Frye, BAD_delayed_choices


design_thing = BAD_delayed_choices()
model = models.Hyperbolic(n_particles=5_000)

for trial in range(3):
    design = design_thing.get_next_design(model)

    # make up a random resonse and enter it
    last_response_chose_delayed = True
    design_thing.enter_trial_design_and_response(
        design, last_response_chose_delayed)

    # update beliefs
    model.update_beliefs(design_thing.all_data)

    print(f'trial {trial} complete 🙂')


print('Simulated BAD experiment complete: 😀 ✅')
