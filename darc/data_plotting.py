import matplotlib.pyplot as plt
import numpy as np
import logging


def all_data_plotter(all_data, filename):
    '''Visualise data'''

    f, axes = plt.subplots(1, 2, figsize=(9, 4), tight_layout=True)

    RA = all_data.RA.values
    DA = all_data.DA.values
    PA = all_data.PA.values
    RB = all_data.RB.values
    DB = all_data.DB.values
    PB = all_data.PB.values
    R = all_data.R.values
    n_points, _ = all_data.shape

    # delay plot ===================================================
    if np.any(DA > 0):
        plot_delay_with_front_end_delays(axes[0], RA, DA, RB, DB, R)
    else:
        plot_delay_without_front_end_delays(axes[0], RA, DA, RB, DB, R)
        
    # probability plot =============================================
    plot_probability_data(axes[1], RA, PA, RB, PB, R)
    
    # Exporting, etc
    savename = filename + '_data_plot.pdf'
    plt.savefig(savename)
    logging.info(f'Data plot saved as: {savename}')


# DELAY PLOTS =========================================================

def plot_delay_with_front_end_delays(ax, RA, DA, RB, DB, R):
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
    ax.set_title('Plot for data with front-end delays')


def plot_delay_without_front_end_delays(ax, RA, DA, RB, DB, R):
    ax.scatter(x=DB, y=RA / RB, c=R, alpha=0.5)
    ax.set_xlabel('delay (days)')
    ax.set_ylabel('RA/RB')


# RISKY PLOTS =========================================================


def plot_probability_data(ax, RA, PA, RB, PB, R):
    ax.scatter(x=PB, y=RA / RB, c=R, alpha=0.5)
    ax.set_xlabel('probability (PB)')
    ax.set_ylabel('RA/RB')
    ax.set_title('Plot by reward probability')
