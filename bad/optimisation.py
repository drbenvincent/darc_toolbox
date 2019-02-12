import numpy as np
from bad.utils import normalise, sample_rows, shuffle_rows
import logging


def design_optimisation(designs, predictive_y, θ,
                        n_particles=None, n_steps=50, gamma=None,
                        output_type_force=None, pD_min_prop=0.01,
                        penalty_func=None):
    """
    Performs a smc search algorithm similar to given in Amzal et al for
    calculating the best design from a discrete set according to an entropy
    reduction criterion.  Presumes that the output of the experiment is binary.

    Note this in itself a unique algorithm that has not occured directly in
    the literature before.

    Args:
    predictive_y: Function of the form f(θ,D) that
        returns p(Y==1|θ,D), i.e. the predictive distribution of y for
        a bernoulli given a particular θ and D. MUST BE CORRECTLY NORMALIZED
    designs: (nD x dD pandas dataframe) = All the valid designs to consider.
        TODO: for higher dimensions then this will be impractical. However,
        as this will most likely require a seperate optimization algorithm
        anyway I omit it for now.
    θ: (nT x dT pandas dataframe) Set of unweighted samples representing
        the posterior over parameters. Note that can always convert a
        weighted set of particles to an unweighted set by sampling with
        replacement (see resampling code in pmc).  Note the speed is
        proportional to the number of θ therefore if things are
        going to slowly then may be advisable to use less samples.
    n_particles: (scalar <= nT) Number of θ samples to use at each
        iteration (default = nT/10). Must not be more than the number of
        samples provided. If less then the order of the samples is randomly
        permuted and the data then cycled through.
    n_steps (scalar): Number of annealing steps to run the optimizer for.
    gamma: (function with integer input) = Annealing schedule for
        the optimization. At step n+1 we sample designs in proportion to
        U.^gamma(n). Thus the larger gamma(n) is the more aggressively the
        computational reasources concentrate on the what the maximum this far
        is expected to be. Default is that gamma(n) = n. Note that can also be
        a vector of length n_steps-1 as when called with integer inputs this
        operates as in same manner as an anonymous function.
    output_type_force: ('none' (default) | 'true' | 'false') There can be a
        pathology where the model asks almost exlusively questions that have
        the same answer due to the model misspecification causing over
        confidence. We might therefore wish to ask the question that maximizes
        the entropy reduction subject to for example p(Y=1|D) > 0.5 ('true') or
        p(Y=0|D) < 0.5 ('false').  In other words we might wish try to force an
        artificial balancing of the number of questions which get answered true
        or false to make sure we have reperesentative data.  This seemed to
        help in practise with my original implementation.
    pD_min_prop: real between 0 and 1 (default = 1/100) Minimum value of the
        probability of sampling a design as a proportion of the average, i.e.
        pD_min = pD_min_prop/nD where nD is number of designs. This is done
        before the renormalizing so the true pD_min can is very slightly
        smaller. Note that the result of this is that on average pD_min_prop of
        the computational resources are randomly allocated. This is necessary
        to ensure convergence, but we will usually have pD_min_prop set to a
        small value.

    Returns:
    chosen_design: (1 x dD row of a pandas dataframe) The chosen "optimal" design
    estimated_utilities: (nD x 1 column vector) The estimated value of the
        utility for each design.  Accuracy will be more accurate in the regions
        near the maximum

    TODO: Also write the case for non binary outputs (this will need to be
    written with some noticeable differences)

    TODO: Allow for working in higher dimensions by allowing MCMC transitions
    between designs rather than global selection.

    TODO: Change to incorporate uncertainty in the estimates - we should be
    sampling from some sort of upper confidence bound or probability of the
    point being the maximum rather than an annealed version of the mean
    estimate.

    Originally written in Matlab by Tom Rainforth, 06/05/16
    Converted to Python by Benjamin T Vincent, June 2018
    """

    assert designs.ndim == 2, "designs must be 2D"

    nD = designs.shape[0]
    nT = θ.shape[0]

    # Calculate penalty factors
    if penalty_func is None:
        penalty_factors = np.ones(nD)
    else:
        penalty_factors = penalty_func(designs)

    assert penalty_factors.ndim is 1, "penalty_factors should be 1 dimensional"

    assert nD > 0, "No designs provided!"

    # This will keep track of "target" function for the optimization at each of
    # the candidate points
    U = (1/nD)*np.ones(nD)

    # Tracks number of times a design was sampled
    n_times_sampled = np.zeros(nD)

    # Tracks a running estimate of p(y | D)
    p_y_given_D = np.ones(nD)*0.5

    # Ensure all input arguments are resolved properly
    if n_particles is None:
        n_particles = np.uint32(nT)
    else:
        assert n_particles <= nT, "n_particles must be <= nT"

    if gamma is None:
        gamma = np.arange(n_steps+1)

    pD_min = pD_min_prop/nD

    # Randomly permute the samples so that if not using all of them then there
    # is not a bias originating from the ordering
    θ = shuffle_rows(θ)

    θ_pos_counter = 0

    for nSam in range(1, n_steps):

        if sum(U) == 0.0:
            logging.warning('No design helpful, off the edge of the design space!')
            chosen_design = designs[np.random.randint(0, nD), :]
            estimated_utilities = U
            return (chosen_design, estimated_utilities)

        pD = calculate_pD(U, gamma[nSam], pD_min)

        D_samples, iSamples = sample_rows(designs,
                                          size=n_particles,
                                          replace=True,
                                          p=pD)

        # Number of times a design was sampled this iteration
        #n_times_sampled_iter = np.bincount(iSamples, minlength=np.max(iSamples))
        # NOTE: BEN CHANGED TO...
        n_times_sampled_iter = np.bincount(iSamples, minlength=nD)

        # Select the θ that will be used this iteration
        θ_iter, θ_pos_counter = get_θ_subset(θ, θ_pos_counter, n_particles)

        # Call one step predictive function for each design-parameter pair,
        # note that this is already normalized so pnotY = 1-pY.
        # NOTE: both θ_iter and D_samples will have something like 5000 rows. We want
        # the output p_y_given_θ_and_D to be 5000 elements
        p_y_given_θ_and_D = predictive_y(θ_iter, D_samples)
        log_p_y_given_θ_and_D = np.log(p_y_given_θ_and_D)

        # Calculate p(Y|D) by marginalizing over θ

        p_y_given_D_iter_times_n_samples = np.bincount(iSamples,
                                                       weights=p_y_given_θ_and_D,
                                                       minlength=nD)
        p_y_given_D = ((p_y_given_D * n_times_sampled +
                        p_y_given_D_iter_times_n_samples) /
                       (n_times_sampled+n_times_sampled_iter))

        # Anything with no examples of has probability 0.5
        p_y_given_D[np.isnan(p_y_given_D)] = 0.5

        # TODO think about whether there are implications that different
        # iterations have different p_y_given_D estimates - should we have some
        # sort of burn in period?

        U = calc_utility(U, p_y_given_θ_and_D, log_p_y_given_θ_and_D,
                         p_y_given_D, iSamples, nD, n_times_sampled,
                         n_times_sampled_iter, penalty_factors)

        # Update the counts of times the design was sampled
        n_times_sampled += n_times_sampled_iter

    if output_type_force is not None:
        U = do_output_type_force_thing(U, p_y_given_D, output_type_force)

    # Choose the design for which pD is maximal
    chosen_design = designs.take([np.argmax(U)])
    return (chosen_design, U)


def calculate_pD(U, gamma, pD_min):
    U_bar = normalise(U)  # To guard against underflow
    pD = U_bar**gamma
    pD = normalise(pD)
    pD = np.maximum(pD, pD_min)
    pD = normalise(pD)
    return pD


def get_θ_subset(θ, θ_pos_counter, n_particles):
    nT = θ.shape[0]
    θ_end_position = θ_pos_counter + n_particles
    if θ_end_position < nT:
        idx = np.arange(θ_pos_counter, θ_end_position)
    else:
        idx = np.concatenate([np.arange(0, np.mod(θ_end_position, nT)),
                              np.arange(θ_pos_counter, nT)])

    θ_iter = θ.iloc[idx]
    θ_pos_counter = np.mod(θ_end_position, nT)
    return (θ_iter, θ_pos_counter)


def calc_utility(U, p_y_given_θ_and_D, log_p_y_given_θ_and_D,
                 p_y_given_D, iSamples, nD, n_times_sampled,
                 n_times_sampled_iter, penalty_factors):
    """The utility of a step is the mutual information between the parameter and
    the observation.  Note this is equal to the expected gain in Shannon
    information from prior to posterior for a single question."""
    # First marginalize over y
    U_θ = (p_y_given_θ_and_D *
           (log_p_y_given_θ_and_D - np.log(p_y_given_D[iSamples])) +
           (1-p_y_given_θ_and_D) *
           (np.log(1-p_y_given_θ_and_D) - np.log(1-p_y_given_D[iSamples])))
    # Anything with numerical instability has no utility as the instability
    # ocurs because we are certain of the result
    U_θ[np.isnan(U_θ)] = 0
    # Any negatives are just numerical instability
    U_θ = np.maximum(U_θ, 0)

    # Final utility from iterations also marginalized out over θ
    U_iter_times_n_samples = np.bincount(iSamples, U_θ, minlength=nD)
    # TODO: calculate more than just the mean to calculate probability the
    # point is the maximum

    U_iter_times_n_samples = U_iter_times_n_samples*penalty_factors

    # Update the running estimate of U for each point
    b_some_samples = (n_times_sampled+n_times_sampled_iter) > 0
    U = ((U * n_times_sampled + U_iter_times_n_samples) /
         (n_times_sampled + n_times_sampled_iter))
    U[np.logical_not(b_some_samples)] = 1/nD
    # guard against numerical error
    U[U < 0.0] = 0.0
    return U


def do_output_type_force_thing(U, p_y_given_D, output_type_force):
    if output_type_force is True:
        if np.max(p_y_given_D) >= 0.5:
            U[p_y_given_D < 0.5] = 0.0
    elif output_type_force is False:
        if np.min(p_y_given_D) <= 0.5:
            U[p_y_given_D > 0.5] = 0.0
    return U
