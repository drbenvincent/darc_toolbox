from darc.delayed import models 
import pandas as pd


model = models.Hyperbolic(n_particles=5_000)

# SET UP DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data_columns = ['RA', 'DA', 'PA', 'RB', 'DB', 'PB', 'R']
# create empty dataframe
all_data = pd.DataFrame(columns=data_columns)

# fake trial data
trial_data = {'RA': [100], 'DA': [0], 'PA': [1.],
              'RB': [110], 'DB': [7], 'PB': [1.],
              'R':[int(False)]}
all_data = all_data.append(pd.DataFrame(trial_data))

# fake trial data
trial_data = {'RA': [100], 'DA': [0], 'PA': [1.],
              'RB': [150], 'DB': [7], 'PB': [1.],
              'R': [int(True)]}
all_data = all_data.append(pd.DataFrame(trial_data))

# fake trial data
trial_data = {'RA': [100], 'DA': [0], 'PA': [1.],
              'RB': [160], 'DB': [60], 'PB': [1.],
              'R': [int(False)]}
all_data = all_data.append(pd.DataFrame(trial_data))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# IF THIS WORKS THEN I WILL BE HAPPY :) 
model.update_beliefs(all_data)
print('UPDATED BELIEFS 😁')

# probe_trial = {'RA': [100], 'DA': [0], 'RB': [150], 'DB': [7]}
# probe_trial = pd.DataFrame(probe_trial)
# chose_delayed = model.get_simulated_response(probe_trial)
# print(chose_delayed)
