import matplotlib.pyplot as plt
import numpy as np
import logging


def data_plotter(data, filename=None):
    '''Our high level plotting function which dispatches to the appropriate
    low-level plotting functions based upon the nature of the data'''

    x = data.x.values
    R = data.R.values
    n_points, _ = data.shape

    f, ax = plt.subplots(1, 1, figsize=(9, 4))

    # DO PLOTTING HERE
    ax.scatter(x=x, y=R, c='b', alpha=0.5)

    # ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('response')

    if filename is not None:
        savename = filename + '_data_plot.pdf'
        plt.savefig(savename)
        logging.info(f'Data plot saved as: {savename}')
