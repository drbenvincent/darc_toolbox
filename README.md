# darc-experiments-python

## About

**Status:  🔥 Under active development. This is pre-alpha code. 🔥** 

This code relates to the following pre-print:
> Vincent, B. T., & Rainforth, T. (2017, October 20). The DARC Toolbox: automated, flexible, and efficient delayed and risky choice experiments using Bayesian adaptive design. Retrieved from psyarxiv.com/yehjb

The pre-print is likely to appear in quite a different form when finally published.


# What does this do?
The aim is to be able to run experiments using Bayesian Adaptive Design, using the approach set out in [Vincent & Rainforth, 2017](https://psyarxiv.com/yehjb). This was originally implemented in Matlab code in the [darc-experiments-matlab](https://github.com/drbenvincent/darc-experiments-matlab) repo, but I am porting this to Python.

Further, we are embedding this Bayesian Adaptive Design code within the [PsychoPy](http://www.psychopy.org) experiment framework.


# Features

## Core features
Feature | Status | Info
--- | --- | ---
Run DARC experiments in PsychoPy | ✅ | Run experiments with ease using PsychoPy
Exporting of data | ✅ | Exports raw trial-level data, summary statistics of our posteriors over model parameters, and a full particle-based representation of our posterior distribution over parameters. Reaction times are also exported.
Pick your own experiment design + model combination| ✅ | At this point we rely on very minor changes to the PsychoPy code snippets to select model and design preferences. (We might be able to explore doing this with a simple GUI in the future)
Customisable experimental protocol | ✅ | Easy to customise the set of allowable rewards, delays and probabilities, i.e. the design space. Make simple updates to Python code snippets in the PsychoPy Builder view.
Customise the framing of choices presented to participants. | ✅ | You can customise the way in which the experiment designs (rewards, delays, and probabilities) are presented to the participant by: a) altering stimulus components in the PsychoPy Builder view, and b) making edits to the short Python code snippets in PsychoPy

## Additional or advanced features
Feature | Status | Info
--- | --- | ---
Customise your prior beliefs over parameters | ✅ | This is a key feature of running efficient adaptive experiments. At the moment you have to edit priors in our Python code.
Basic results visualization | ✅ | So far we've got basic visualisation of the marginal posteriors over parameters, and also some basic plotting of the (design, response) data collected.
Run simulated experiments | in progress | This is of use to people wanting to develop/impliment their own models, to check it works.
Interleave multiple adaptive experiments | ❌ | If you want to do interesting mixed-block experiments or react to the current estimates of model parameters (e.g. discount rates) then you can do that.
Inject custom trials | ❌ | Left to it's own devices, an experiment will choose it's own set of designs. But if you have particular experimental needs, you can inject your own (manually specified) designs amongst automatically run trials.


# Design paradigms
Paradigm | Status | Info
--- | --- | ---
Kirby paradigm | ✅ | Fixed design, delay discounting only
Frye et al paradigm | ✅ | Adaptive (heuristic) approach, delay discounting only
BAD | ✅ ❌ | **Preprint:** Vincent, B. T., & Rainforth, T. (2017, October 20). The DARC Toolbox: automated, flexible, and efficient delayed and risky choice experiments using Bayesian adaptive design. Retrieved from psyarxiv.com/yehjb


# DARC Cognitive models available

Yes, you can in run adaptive experiments to make very efficient inferences about the parameters for models of your choice 🙂. See below for a list of completed and planned model implementations.

## Delayed reward paradigm models
Model | Status | Info
--- | --- | ---
Exponential | ✅ | Samuelson, P. A. (1937). A note on measurement of utility. The Review of Economic Studies, 4(2), 155. http://doi.org/10.2307/2967612
Hyperbolic | ✅ | Mazur, J. E. (1987). An adjusting procedure for studying delayed reinforcement. In M. L. Commons, J. A. Nevin, & H. Rachlin (Eds.), Quantitative Analyses of Behavior (pp. 55–73). Hillsdale, NJ: Erlbaum.
HyperbolicMagnitudeEffect | ✅ | Vincent, B. T. (2016). Hierarchical Bayesian estimation and hypothesis testing for delay discounting tasks. Behavior Research Methods, 48(4), 1608–1620. http://doi.org/10.3758/s13428-015-0672-2
ExponentialMagnitudeEffect | ✅ | 
ConstantSensitivity | ❌ | * Negative b values causing errors *
ProportionalDifference | ✅ | González-Vallejo, C. (2002). Making trade-offs: A probabilistic and context-sensitive model of choice behavior. Psychological Review, 109(1), 137–155. http://doi.org/10.1037//0033-295X.109.1.137
HyperbolicNonLinearUtility | ❌ | Cheng, J., & González-Vallejo, C. (2014). Hyperbolic Discounting: Value and Time Processes of Substance Abusers and Non-Clinical Individuals in Intertemporal Choice. PLoS ONE, 9(11), e111378–18. http://doi.org/10.1371/journal.pone.0111378

## Risky reward paradigm models
Model | Status | Info
--- | --- | ---
Hyperbolic | ✅ | Hyperbolic discounting of odds against reward
Generalized hyperbolic | ❌ |
Prelec (1998) | ❌ |
Prospect Theory | ❌ | 
Proportional difference | ✅ | González-Vallejo, C. (2002). Making trade-offs: A probabilistic and context-sensitive model of choice behavior. Psychological Review, 109(1), 137–155. http://doi.org/10.1037//0033-295X.109.1.137

## Delayed and risky reward paradigm models
Model | Status | Info
--- | --- | ---
AdditiveHyperbolic | ❌ | Yi, R., la Piedad, de, X., & Bickel, W. K. (2006). The combined effects of delay and probability in discounting. Behavioural Processes, 73(2), 149–155. http://doi.org/10.1016/j.beproc.2006.05.001
MultiplicativeHyperbolic | ✅ | Vanderveldt, A., Green, L., & Myerson, J. (2015). Discounting of monetary rewards that are both delayed and probabilistic: delay and probability combine multiplicatively, not additively. Journal of Experimental Psychology: Learning, Memory, and Cognition, 41(1), 148–162. http://doi.org/10.1037/xlm0000029 
Proportional difference | ❌ | [think, then implement]
Probability and Time Trade-off model | ❌ | (Baucells & Heukamp)

# Requirements (likely to change 🔥)
NOTE: Advice on versions and compatibility etc will be improved over time.

This code is being developed with:
- Python 3
- To run experiments you will also need [PsychoPy](http://www.psychopy.org), currently using their Python 3 release, version 1.90.1. I'll probably update to PsychoPy 3 before we make any kind of official release.

# Installation instructions (likely to change 🔥)
1. Ensure you have a Python 3.6 installation. I recommend https://www.anaconda.com/download/
2. Install [PsychoPy](http://www.psychopy.org). Make sure you install a version that works with Python 3. 
3. Download or clone this `darc-experiments-python` repository.
4. Open up PsychoPy... Open the PsychoPy experiments `experiment.psyexp` in the builder view... Run the experiment with the green man symbol... check the auto-saved data in the `\data` folder.


# How to...

(coming soon)


# References

**NOTE:** This work is based on the pre-print below. This is not yet published and is likely to appear in a subtantially altered form.

> Vincent, B. T., & Rainforth, T. (2017, October 20). The DARC Toolbox: automated, flexible, and efficient delayed and risky choice experiments using Bayesian adaptive design. Retrieved from [psyarxiv.com/yehjb](https://psyarxiv.com/yehjb)
