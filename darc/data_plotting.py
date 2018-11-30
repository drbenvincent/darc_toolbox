import matplotlib.pyplot as plt
import numpy as np
import logging


def all_data_plotter(all_data):
    '''Our high level plotting function which dispatches to the appropriate
    low-level plotting functions based upon the nature of the data'''

    RA = all_data.RA.values
    DA = all_data.DA.values
    PA = all_data.PA.values
    RB = all_data.RB.values
    DB = all_data.DB.values
    PB = all_data.PB.values
    R = all_data.R.values
    n_points, _ = all_data.shape

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
        f, ax = plt.subplots(1, 1, figsize=(9, 4))
        if front_end_delays:
            plot_delay_with_front_end_delays(ax, all_data)
        else:
            plot_delay_without_front_end_delays(ax, all_data)

    elif risky_choices and not delayed_choices:
        # risky based plot only
        f, ax = plt.subplots(1, 1, figsize=(9, 4))
        plot_probability_data(ax, all_data)

    elif delayed_choices and risky_choices:
        # both risky and delayed plots
        f, axes = plt.subplots(1, 2, figsize=(9, 4), tight_layout=True)
        if front_end_delays:
            plot_delay_with_front_end_delays(axes[0], all_data)
        else:
            plot_delay_without_front_end_delays(axes[0], all_data)

        plot_probability_data(axes[1], all_data)

    # if filename is not None:
    #     savename = filename + '_data_plot.pdf'
    #     plt.savefig(savename)
    #     logging.info(f'Data plot saved as: {savename}')


# DELAY PLOTS =========================================================

def plot_delay_with_front_end_delays(ax, all_data):

    RA = all_data.RA.values
    DA = all_data.DA.values
    PA = all_data.PA.values
    RB = all_data.RB.values
    DB = all_data.DB.values
    PB = all_data.PB.values
    R = all_data.R.values

    # plot lines between pairs of prospects
    n_points = len(R)
    for t in range(n_points):
        ax.plot(np.array([DA[t], DB[t]]),
                np.array([RA[t], RB[t]]),
                c='k',
                linewidth=1)

    # plot option chosen as one colour
    ax.scatter(x=[DA[R == 0], DB[R == 0]], y=[RA[R == 0], RB[R == 0]],
               c='b', alpha=0.5, label='chose A')
    ax.scatter(x=[DA[R == 1], DB[R == 1]], y=[RA[R == 1], RB[R == 1]],
               c='r', alpha=0.5, label='chose B')

    ax.legend()
    ax.set_xlabel('delay (days)')
    ax.set_ylabel('reward')
    ax.set_xscale('log')
    #ax.set_title('Plot for data with front-end delays')


def plot_delay_without_front_end_delays(ax, all_data):

    RA = all_data.RA.values
    DA = all_data.DA.values
    PA = all_data.PA.values
    RB = all_data.RB.values
    DB = all_data.DB.values
    PB = all_data.PB.values
    R = all_data.R.values

    ax.scatter(x=DB[R == 0], y=RA[R == 0] / RB[R == 0], 
               c='b', alpha=0.5, label='chose A')
    ax.scatter(x=DB[R == 1], y=RA[R == 1] / RB[R == 1],
               c='r', alpha=0.5, label='chose B')
    ax.legend()
    ax.set_xlabel('delay (days)')
    ax.set_ylabel('$R^A/R^B$')
    ax.set_xscale('linear')
    #ax.set_title('Plot for data without front-end delays')


# RISKY PLOTS =========================================================


def plot_probability_data(ax, all_data):

    RA = all_data.RA.values
    DA = all_data.DA.values
    PA = all_data.PA.values
    RB = all_data.RB.values
    DB = all_data.DB.values
    PB = all_data.PB.values
    R = all_data.R.values

    ax.scatter(x=PB[R == 0], y=RA[R == 0] / RB[R == 0],
               c='b', alpha=0.5, label='chose A')
    ax.scatter(x=PB[R == 1], y=RA[R == 1] / RB[R == 1],
                c='r', alpha=0.5, label='chose B')
    ax.set_xlabel('probability (PB)')
    ax.set_ylabel('RA/RB')
    ax.legend()
    ax.set_title('Plot by reward probability (Assumes PA=1=riskless)')
