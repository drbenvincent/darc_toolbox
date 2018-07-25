import numpy as np


def normalise(x, return_sum=False):
    # NOTE: could alternatively use sklearn.preprocessing.normalize
    sum_x = np.sum(x)
    x = x/sum_x
    if return_sum:
        return (x, sum_x)
    else:
        return x


def sample_rows(df, size, replace, p):
    """Sample the rows of a pandas df"""
    n_rows = df.shape[0]
    iSamples = np.random.choice(n_rows, size=size, replace=replace, p=p)
    samples = df.iloc[iSamples,:]
    return (samples, iSamples)


def shuffle_rows(df):
    '''
    Shuffle the rows of a dataframe
    see https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows 
    '''
    return df.sample(frac=1).reset_index(drop=True)
