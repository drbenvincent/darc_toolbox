import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import logging


def data_plotter(data, filename=None, ax=None):
    '''Our high level plotting function which dispatches to the appropriate
    low-level plotting functions based upon the nature of the data'''

    RA = data.RA.values
    DA = data.DA.values
    PA = data.PA.values
    RB = data.RB.values
    DB = data.DB.values
    PB = data.PB.values
    R = data.R.values
    n_points, _ = data.shape

    # do interrogation of the data
    if np.any(PA < 1) or np.any(PB < 1):
        risky_choices = True
    else:
        risky_choices = False

    if np.any(DA > 0) or np.any(DB > 0):
        delayed_choices = True
        if np.any(DA > 0):
            front_end_delays = True
        else:
            front_end_delays = False
    else:
        delayed_choices = False

    # make plotting decisions
    if delayed_choices and not risky_choices:
        # delay based plot only
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(9, 4))
        if front_end_delays:
            plot_delay_with_front_end_delays(ax, data)
        else:
            plot_delay_without_front_end_delays(ax, data)

    elif risky_choices and not delayed_choices:
        # risky based plot only
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(9, 4))
        plot_probability_data(ax, data)

    elif delayed_choices and risky_choices:
        # both risky and delayed plots
        f, axes = plt.subplots(1, 2, figsize=(9, 4), tight_layout=True)
        if front_end_delays:
            plot_delay_with_front_end_delays(axes[0], data)
        else:
            plot_delay_without_front_end_delays(axes[0], data)

        plot_probability_data(axes[1], data)

    if filename is not None:
        savename = filename + '_data_plot.pdf'
        plt.savefig(savename)
        logging.info(f'Data plot saved as: {savename}')


# DELAY PLOTS =========================================================

def plot_delay_with_front_end_delays(ax, data):

    print('plot_delay_with_front_end_delays')
    data = convert_delay_data_frontend(data)

    s = ax.scatter(data['x'], data['y'],
                   s=freq_to_area(data['freq']),
                   c=data['prop'],
                   cmap='Greys',
                   edgecolor='k',
                   label='response data')

    _, ymax = ax.get_ylim()
    ax.set_ylim([0, ymax])

    cbar = plt.colorbar(s)
    cbar.set_label('proportion chose delayed')

    ax.set_xlabel('time to smaller amount [days]')
    ax.set_ylabel('inter-reward delay [days]')


def plot_delay_without_front_end_delays(ax, data, cbar=True):

    data = convert_delay_data(data)

    s = ax.scatter(data['x'], data['y'],
                   s=freq_to_area(data['freq']),
                   c=data['prop'],
                   cmap='Greys',
                   edgecolor='k',
                   label='response data')

    ax.set_xlabel('delay (days)')
    ax.set_ylabel('$R^A/R^B$')
    ax.set_xscale('linear')

    if cbar:
        cbar = plt.colorbar(s)
        cbar.set_label('proportion chose delayed')
    # legend = ax.get_legend()
    # legend.legendHandles[0].set_color(plt.Greys(.5))

    #ax.set_title('Plot for data without front-end delays')


def convert_delay_data_frontend(data):
    ''' Convert raw data (dataframe) into a new data frame with:
    x = DA
    y = DB-DA = inter-reward interval
    freq = frequency of this design
    prop = average response to each unique design
    '''

    # define columns that define a unique design
    cols = ['RA', 'DA', 'PA', 'RB', 'DB', 'PB']
    # new dataframe of unique designs with additional freq count column
    new = data.groupby(cols).size().reset_index(name='freq')
    # add average response to each unique design
    new['prop'] = data.groupby(
        cols)['R'].mean().reset_index(name='prop')['prop']
    new['x'] = new['DA']
    new['y'] = new['DB']-new['DA']
    new = new[['x', 'y', 'freq', 'prop']]
    return new


def convert_delay_data(data):
    ''' Convert raw data (dataframe) into a new data frame with:
    x = DB
    y = RA/RB
    freq = frequency of this design
    prop = average response to each unique design
    '''

    # define columns that define a unique design
    cols = ['RA', 'DA', 'PA', 'RB', 'DB', 'PB']
    # new dataframe of unique designs with additional freq count column
    new = data.groupby(cols).size().reset_index(name='freq')
    # add average response to each unique design
    new['prop'] = data.groupby(
        cols)['R'].mean().reset_index(name='prop')['prop']
    new['x'] = new['DB']
    new['y'] = new['RA']/new['RB']
    new = new[['x', 'y', 'freq', 'prop']]
    return new


def freq_to_area(freq):
    '''convert frequency of occurence of designs into an area for scatter
    plotting'''
    # area = pi * r **2
    area = np.pi * freq**2
    return area*10


# RISKY PLOTS =========================================================


def plot_probability_data(ax, data):

    data = convert_risk_data(data)

    ax.scatter(data['x'], data['y'],
               s=freq_to_area(data['freq']),
               c=data['prop'],
               cmap='Greys',
               edgecolor='k',
               label='response data')

    ax.set_xlabel('$P^B$')
    ax.set_ylabel('$\pi(P^B)$')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])


def convert_risk_data(data):
    ''' Convert raw data (dataframe) into a new data frame with:
    x = RB
    y = RA/RB
    freq = frequency of this design
    prop = average response to each unique design
    '''

    # define columns that define a unique design
    cols = ['RA', 'DA', 'PA', 'RB', 'DB', 'PB']
    # new dataframe of unique designs with additional freq count column
    new = data.groupby(cols).size().reset_index(name='freq')
    # add average response to each unique design
    new['prop'] = data.groupby(
        cols)['R'].mean().reset_index(name='prop')['prop']
    new['x'] = new['PB']
    new['y'] = new['RA']/new['RB']
    new = new[['x', 'y', 'freq', 'prop']]
    return new
