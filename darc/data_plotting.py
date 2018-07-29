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

    #fig, axes = plt.subplots(1, 2, figsize=(9, 4), tight_layout=True)
    
    # delay plot ===================================================
    if np.any(DA > 0):
        # there are front end delays

        # plot lines between pairs of prospects
        for t in range(n_points):
            axes[0].plot(np.array([DA[t], DB[t]]),
                         np.array([RA[t], RB[t]]),
                         c='k',
                         linewidth=1)

        # plot option chosen as one colour

        axes[0].scatter(x=[DA[R == 0], DB[R==0]], y=[RA[R == 0], RB[R==0]],
                        c='b', alpha=0.5, label='chose A')
        axes[0].scatter(x=[DA[R == 1], DB[R==1]], y=[RA[R == 1], RB[R==1]],
                        c='r', alpha=0.5, label='chose B')

        axes[0].legend()
        axes[0].set_xlabel('delay (days)')
        axes[0].set_ylabel('reward')
        axes[0].set_xscale('log')
        axes[0].set_title('Plot for data with front-end delays')
    else:
        # there are NO front end delays
        axes[0].scatter(x=DB,
                        y=RA / RB,
                        c=all_data.R,
                        alpha=0.5)

        axes[0].set_xlabel('delay (days)')
        axes[0].set_ylabel('RA/RB')

    # probability plot =============================================
    axes[1].scatter(x=PB,
                    y=RA / RB,
                    c=R,
                    alpha=0.5)
    axes[1].set_xlabel('probability (PB)')
    axes[1].set_ylabel('RA/RB')
    axes[1].set_title('Plot by reward probability')

    savename = filename + '_data_plot.pdf'
    plt.savefig(savename)
    logging.info(f'Data plot saved as: {savename}')
