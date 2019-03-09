import matplotlib.pyplot as plt
import logging


def tri_plot(θ, filename=None, θ_true=None, priors=None):

    logging.debug('Entered tri_plot.py')
    n_params = len(priors)

    fig, axes = plt.subplots(n_params, n_params,
        figsize=(9, 9), tight_layout=True, sharex='col')

    # plot marginal histograms on the diagonal
    for n, key in enumerate(θ.keys()):
        axes[n, n].hist(θ[key], 30, density=1, facecolor='green', alpha=0.75)
        axes[n, n].set_xlabel(key)
        if θ_true is not None:
            axes[n, n].axvline(x=θ_true[key][0], color='k', linestyle='-')

    logging.debug('Plotted marginal histograms ok')

    # now plot bivariate scatter in the lower triangle
    for row, row_key in enumerate(θ.keys()):
        for col, col_key in enumerate(θ.keys()):
            logging.debug(f'row/col = {row, col}')

            # bivariate scatter plots in lower triangle
            if col < row:
                axes[row, col].scatter(θ[col_key], θ[row_key], alpha=0.1)
                if θ_true is not None:
                    axes[row, col].axvline(
                        x=θ_true[col_key][0], color='k', linestyle='-')
                    axes[row, col].axhline(
                        y=θ_true[row_key][0], color='k', linestyle='-')
                axes[row, col].set_xlabel(col_key)
                axes[row, col].set_ylabel(row_key)

            # remove axes in the upper triangle
            if col > row:
                axes[row, col].remove()

    logging.debug('Plotted bivariate scatter plots ok')

    if filename is not None:
        savename = filename + '_parameter_plot.pdf'
        plt.savefig(savename)
        logging.info(f'Posterior histograms exported: {savename}')
