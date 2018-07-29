import matplotlib.pyplot as plt
import logging


def tri_plot(θ, filename, θ_true=None, priors=None):

    keys = θ.keys()
    n_params = len(priors)
    
    fig, axes = plt.subplots(n_params, n_params, 
        figsize=(9, 9), tight_layout=True, sharex='col')

    # plot marginal histograms on the diagonal
    for n, key in enumerate(keys):
        axes[n, n].hist(θ[key], 30, density=1, facecolor='green', alpha=0.75)
        axes[n, n].set_xlabel(key)
        if θ_true is not None:
            axes[n, n].axvline(x=θ_true[key][0], color='k', linestyle='-')

    # now plot bivariate scatter in the lower triangle
    for row in range(n_params):
        row_key = keys[row]

        for col in range(n_params):
            col_key = keys[col]

            if col < row:
                axes[row, col].scatter(θ[col_key], θ[row_key], alpha=0.1)

                axes[row, col].axvline(
                    x=θ_true[col_key][0], color='k', linestyle='-')
                axes[row, col].axhline(
                    y=θ_true[row_key][0], color='k', linestyle='-')

                axes[row, col].set_xlabel(col_key)
                axes[row, col].set_ylabel(row_key)
            
            if col > row:
                axes[row, col].remove()

    savename = filename + '_parameter_plot.pdf'
    plt.savefig(savename)
    logging.info(f'Posterior histograms exported: {savename}')

