import pandas as pd


def single_design_tuple_to_df(design_tuple):
    ''' Convert the named tuple into a 1-row pandas dataframe'''
    trial_data = {'RA': [design_tuple.ProspectA.reward],
                  'DA': [design_tuple.ProspectA.delay],
                  'PA': [design_tuple.ProspectA.prob],
                  'RB': [design_tuple.ProspectB.reward],
                  'DB': [design_tuple.ProspectB.delay],
                  'PB': [design_tuple.ProspectB.prob]}
    design_df = pd.DataFrame.from_dict(trial_data)
    return design_df
