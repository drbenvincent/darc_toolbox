# The DARC Toolbox: automated, flexible, and efficient delayed and risky choice experiments using Bayesian adaptive design.

[![PyPI](https://img.shields.io/pypi/v/darc_toolbox.svg?color=green)](https://pypi.org/project/darc_toolbox/)

Run efficient Bayesian adaptive experiments using Python and the [PsychoPy](http://www.psychopy.org) experiment framework.

This code relates to the following pre-print. But, the pre-print is likely to appear in quite a different form when finally published.
> Vincent, B. T., & Rainforth, T. (2017, October 20). The DARC Toolbox: automated, flexible, and efficient delayed and risky choice experiments using Bayesian adaptive design. Retrieved from psyarxiv.com/yehjb

**Status:  🔥 Under active development 🔥**

# Features

- Easily run a range of decision making experiments using PsychoPy. These include delayed decisions (inter-temporal choice), risky choice tasks, and combined delayed and risky choice tasks.
- Get more accurate measures from fewer trials.
- Rich exports: Get trial-level data, estimated parameters from your chosen decision making model(s), and visualisation of results.
- Easy customisation of experiment details within PsychoPy.

Advanced features are available with some simple edits to the Python code:

- Customise your prior beliefs over model parameters to fit your participant population.
- Simultaneously fit mulitple models to an experimental participant.
- Inject custom trials: if you have particular experimental needs, you can inject your own (manually specified) designs amongst automatically run trials.

# Installation instructions (for most people)
1. **Download PsychoPy:** The DARC Toolbox is built into PsychoPy (version 3.2.0 onwards). So go and [download PsychoPy](https://www.psychopy.org).
2. **Unpack demos:** From the menu, choose `Demos > Unpack demos` to a location of your choosing.
3. **Run demo:** Run one of the adpative demos, for example the `adaptive_decision_making_demo`.

# Installation instructions (for developers)
Coming soon...

# Adaptive Experiment = Experimental Design Space + Cognitive Model
An adaptive experiment is a combination of a set of allowable designs (questions) which we call the **design space** and a **cognitive model**. The Bayesian Adaptive Design methods select which design to present to participants on a trial-to-trial basis, in real time. The goal of this is to maximise the information we gain about our model parameters.

A range of experimental designs and cognitive models are provided and are detailed below.

## Experimental design paradigms
One of the core components of this package is to provide designs chosen through Bayesian Adaptive Design, as outlined in our prepint (Vincent & Rainforth, 2017). The core classes of design we focus on are:

- **Delayed choice tasks (aka inter-temporal choice):** You can choose between various protocols such as: front-end delays, fixed delayed reward, fixed immediate reward, fixed delay, etc.
- **Risky choice tasks:** Choose your range of reward probabilities. These can also be seen as a transformed version of odds against recieving a reward.
- **Simultaneous delayed and risky choice tasks:** Again, you can customise the range of delays and reward probability (risk) levels used in your experiment.

All of these paradigms are available, and can be fine tuned, using our Bayesian Adaptive procedure.

However we also provide the ability to run some other prominent experiment design procedures from the literature. These are:
- The Kirby (2009) delay discounting procedure.
- The 5 trial procedure outlined by Koffarnus & Bickel (2014).
- The adaptive procedure from Du, Green & Myerson (2002) for delayed and risky choice tasks.
- The 'delay slice' procecure from Frye et al (2016), closely related to the approach of Du, Green & Myerson (2002).
- The delay and risky choice procedures from Griskevicius et al (2011).


## DARC Cognitive models available

You can in run adaptive experiments to make very efficient inferences about the parameters for models of your choice. See below for a list of completed models. See the [model-related GitHub issues](https://github.com/drbenvincent/darc-experiments-python/issues?q=is%3Aissue+is%3Aopen+label%3Amodel) to see what is in progress. Feel free to impliment additional models or request one.

### Delayed reward paradigm models
Model | Info
--- | ---
Exponential | Samuelson, P. A. (1937). A note on measurement of utility. The Review of Economic Studies, 4(2), 155. http://doi.org/10.2307/2967612
Hyperbolic | Mazur, J. E. (1987). An adjusting procedure for studying delayed reinforcement. In M. L. Commons, J. A. Nevin, & H. Rachlin (Eds.), Quantitative Analyses of Behavior (pp. 55–73). Hillsdale, NJ: Erlbaum.
HyperbolicMagnitudeEffect | Vincent, B. T. (2016). Hierarchical Bayesian estimation and hypothesis testing for delay discounting tasks. Behavior Research Methods, 48(4), 1608–1620. http://doi.org/10.3758/s13428-015-0672-2
ExponentialMagnitudeEffect |
Modified Rachlin hyperboloid | Vincent, B. T., & Stewart, N. (2018, October 16). The case of muddled units in temporal discounting. https://doi.org/10.31234/osf.io/29sgd
Myerson hyperboloid | Myerson, J. and Green, L. (1995). Discounting of delayed rewards: Models of individual choice. Journal of the experimental analysis of behavior, 64(3):263–276.

### Risky reward paradigm models
Model | Info
--- | ---
Hyperbolic |Hyperbolic discounting of odds against reward
Linear in log odds | Gonzalez, R., & Wu, G. (1999). On the shape of the probability weighting function. Cognitive Psychology, 38(1), 129–166. http://doi.org/10.1006/cogp.1998.0710
Proportional difference |González-Vallejo, C. (2002). Making trade-offs: A probabilistic and context-sensitive model of choice behavior. Psychological Review, 109(1), 137–155. http://doi.org/10.1037//0033-295X.109.1.137

### Delayed and risky reward paradigm models
Model | Info
--- | ---
MultiplicativeHyperbolic | Vanderveldt, A., Green, L., & Myerson, J. (2015). Discounting of monetary rewards that are both delayed and probabilistic: delay and probability combine multiplicatively, not additively. Journal of Experimental Psychology: Learning, Memory, and Cognition, 41(1), 148–162. http://doi.org/10.1037/xlm0000029


# How to...

(coming soon)


# Other projects we rely upon

- [PsychoPy](http://www.psychopy.org) as the experiment environment.

Various Python packages including:
- [Numpy](http://www.numpy.org)
- [Pandas](https://pandas.pydata.org)
- [SciPy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html). We use the scipy distributions to represent our prior beliefs over model parameters, and to draw samples from those prior beliefs. See [the full list of distributions here](https://docs.scipy.org/doc/scipy/reference/stats.html)


# References

**NOTE:** This work is based on the pre-print below. This is not yet published and is likely to appear in a subtantially altered form.

> Vincent, B. T., & Rainforth, T. (2017, October 20). The DARC Toolbox: automated, flexible, and efficient delayed and risky choice experiments using Bayesian adaptive design. Retrieved from [psyarxiv.com/yehjb](https://psyarxiv.com/yehjb)

Du, W., Green, L., & Myerson, J. (2002). Cross-Cultural Comparisons of Discounting Delayed and Probabilistic Rewards. The Psychological Record, 52(4), 479–492.

Frye, C. C. J., Galizio, A., Friedel, J. E., DeHart, W. B., & Odum, A. L. (2016). Measuring Delay Discounting in Humans Using an Adjusting Amount Task. Journal of Visualized Experiments, (107), 1-8.

Griskevicius, V., Tybur, J. M., Delton, A. W., & Robertson, T. E. (2011). The influence of mortality and socioeconomic status on risk and delayed rewards: A life history theory approach. Journal of Personality and Social Psychology, 100(6), 1015–26.

Kirby, K. N. (2009). One-year temporal stability of delay-discount rates. Psychonomic Bulletin & Review, 16(3):457–462.

Koffarnus, M. N., & Bickel, W. K. (2014). A 5-trial adjusting delay discounting task: Accurate discount rates in less than one minute. Experimental and Clinical Psychopharmacology, 22(3), 222-228.
