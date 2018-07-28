import numpy as np
import pandas as pd
from scipy.stats import norm, t
from bad.utils import normalise, sample_rows
import logging


# NOTE: we are expecting particles to be an a pandas dataframe, with column names
# rather than just a 2D numpy array.

def update_beliefs(p_log_pdf, θstart, data=None, n_steps=5, step_type='normal',
                   scale_walk_factor=None, ν=5, display=False):
    """
    Automatically sets some of the required parameters for pmc using a random
    walk based proposal distribution.  The calls pmc() to return a new set of
    particles representing the posterior and the estimated marginal
    likelihood.  Proposal is an independent student t in each direction or
    gaussian in each direction depending on choice of step_type option

    Args:
    p_log_pdf: Anonymous function giving the log pdf of the target distribution
        p (i.e. the posterior).  Need not be normalized.  Should take a single
        matrix as arguments where different rows are different samples and
        different columns are different dimensions
    θstart: A matrix of starting points for the sampler
    n_steps: Number of transitions to perform. Resampling is performed after
        each transition, inlcuding the last step

    Optional inputs:
    step_type ('normal' (default) or 'student_t'): Type of distribution to
        use for the proposal.  The student_t is a little more robust but is
        slower to evaluate.
    scale_walk_factor (scalar): Proposal variance scaled is by this.  For
        student t the default is 0.5, for normal the default is 1.
    ν (nu) (default 5): Student t parameter, see wikipedia (only used if
        step_type == 'student_t')
    display  (default false): Print out the mean and std dev of points after
        each step

    Returns:
    θ: Set of samples after the last transition (could potentially return
        samples from all iterations but these will be correlated)
    log_Z:  Estimate of the log marginal likelihood


    Originally written in Matlab by Tom Rainforth 05/05/2016
    Converted to Python by Benjamin Vincent May 2018
    """

    if data is None:
        data = np.array([])

    assert step_type is 'normal' or 'student_t', 'step_type must be normal or student_t'

    if scale_walk_factor is None:
        if step_type is 'normal':
            scale_walk_factor = 2
        elif step_type is 'student_t':
            scale_walk_factor = 2

    # scale walk should be a vector, length equal to number of parameter dimensions
    # NOTE: This will crash if we only have parameter, so we need to manually define a 
    # scale_walk_factor in this case
    if θstart.shape[1]:
        scale_walk = scale_walk_factor
    else:
        scale_walk = scale_walk_factor * np.std(θstart, axis=0)

    if step_type is 'normal':
        def q_log_pdf(θold, θ):
            # bit of a clusterfuck here about subtracting dataframes as the indicies are unordered
            logpdf = norm.logpdf((θold.values-θ.values)/scale_walk, loc=0, scale=1)
            return np.sum(logpdf, axis=1)

        def q_sample(θ):
            return pd.DataFrame(norm.rvs(loc=θ, scale=scale_walk, size=θ.shape),
                                         columns = θ.keys())

    elif step_type is 'student_t':
        def q_log_pdf(θold, θ):
            logpdf = t.logpdf((θold-θ)/scale_walk, loc=0, scale=1, df=ν)
            return np.sum(logpdf, axis=1, keepdims=True)

        def q_sample(θ):
            return pd.DataFrame(t.rvs(df=ν, loc=θ, scale=scale_walk, size=θ.shape),
                                         columns = θ.keys())


    return pmc(data, p_log_pdf, q_log_pdf, q_sample, θstart, n_steps, display=display)


def pmc(data, p_log_pdf, q_log_pdf, q_sample, θstart, n_steps, display=False):
    """
    pmc Population Monte Carlo inference scheme

    Carries out a number of iterations of population monte carlo (PMC) without
    proposal adaptation. See "Population Monte Carlo" (Cappe et al 2012).
    In short PMC without adaptation equates to SMC on a stationary target
    distribtuion.
    
    Inputs
    p_log_pdf  :  Anonymous function giving the log pdf of the target
                  distribution p.  Need not be normalized.  Should take a
                  single matrix as arguments where different rows are
                  different samples and different columns are different
                  dimensions
    q_log_pdf  :  Anonymous function giving the log pdf of the proposal
                  distribution q.  Must be correctly normalized.  Should
                  take two matrices as arguments - the first is the
                  start points and the second is the sampled points.
    q_sample   :  Function for sampling a transition.  Takes a single
                  argument corresponding to a matrix of starting points.
                  Must correspond to a draw from q_log_pdf.
    θstart     :  A matrix of starting points for the sampler
    n_steps    :  Number of transitions to perform.  Resampling is
                  performed after each transition, inlcuding the last step

    Optional Inputs
    display  (default false) : Print out the mean and std dev of points
                  after each step

    Outputs
    θ           :  Set of samples after the last transition (could
                  potentially return samples from all iterations but these
                  will be correlated)
    log_Z       :  Estimate of the log marginal likelihood

    Originally written in Matlab by Tom Rainforth 24/04/2016
    Converted to Python by Benjamin Vincent May 2018
    """

    θold = θstart
    log_Z_steps = np.full(n_steps, np.nan)

    for n in range(n_steps):

        θ = q_sample(θold)

        log_p = p_log_pdf(θ, data)
        assert not np.isnan(log_p).any(), 'Your p_log_pdf is generating NaNs!'
        # NOTE: You may be getting -inf values, for example if q_sample is generating
        # samples out of the domain of some of the parameters. But this is fine here,
        # and simply results in -inf log posterior prob 

        log_q = q_log_pdf(θold, θ)
        assert not np.isnan(log_q).any(), 'Your q_log_pdf is generating NaNs!'

        w, sum_w, z_max = calculate_weights(log_p, log_q)
        θ, _ = sample_rows(θ, θ.shape[0], replace=True, p=np.squeeze(w)) # TODO: do we need the squeeze?
        θold = θ

        log_Z_steps[n] = z_max + np.log(sum_w) - np.log(w.size)

        # TODO: fix this logging/printing 
        # logging.info(f'mean {np.mean(θ)}, std_dev {np.std(θ)}')
        # if display:
        #     # TODO: I've not confirmed that this works
        #     print(f'mean {np.mean(θ)}, std_dev {np.std(θ)}')
            
    log_Z_steps_max = np.max(log_Z_steps)
    Zs = np.exp(log_Z_steps - log_Z_steps_max)
    log_Z = log_Z_steps_max + np.log(np.sum(Zs)) - np.log(log_Z_steps_max.size)

    return (θ, log_Z)


def calculate_weights(log_p, log_q):
    log_w = log_p - log_q
    log_w[(np.isinf(log_w)) | (np.isnan(log_w))] = -np.inf
    z_max = np.max(log_w)
    w = np.exp(log_w-z_max)
    w, sum_w = normalise(w, return_sum=True)
    assert not np.isnan(w).any(), 'At least one of weights is NaN'
    return w, sum_w, z_max
