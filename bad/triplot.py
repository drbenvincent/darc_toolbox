import matplotlib.pyplot as plt
import logging


def tri_plot(θ, filename, θ_true=None, priors=None):

    n_params = len(priors)
    
    fig, axes = plt.subplots(n_params, n_params, figsize=(9, 9), tight_layout=True)

    keys = θ.keys()

    # plot marginal histograms on the diagonal
    for n, key in enumerate(keys):
        axes[n, n].hist(θ[key], 100, density=1, facecolor='green', alpha=0.75)
        axes[n, n].set_xlabel(key)
        if θ_true is not None:
            axes[n, n].axvline(x=θ_true[key][0], color='k', linestyle='-')


    savename = filename + '_parameter_plot.pdf'
    plt.savefig(savename)
    logging.info(f'Posterior histograms exported: {savename}')

