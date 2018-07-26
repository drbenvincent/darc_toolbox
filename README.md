# darc-experiments-python

# Status:  🔥 Under active development. This is pre-alpha code. 🔥 

# What does this do?
The aim is to be able to run experiments using Bayesian Adaptive Design, using the approach set out in [Vincent & Rainforth, 2017](https://psyarxiv.com/yehjb). This was originally implemented in Matlab code in the [darc-experiments-matlab](https://github.com/drbenvincent/darc-experiments-matlab) repo, but I am porting this to Python.

Further, we are embedding this Bayesian Adaptive Design code within the [PsychoPy](http://www.psychopy.org) experiment framework.


# Features

## Core features
Feature | Status | Info
--- | --- | ---
Run DARC experiments in PsychoPy | ✅ | This works with a default way to display prospects. You can update this to your liking in the PsychoPy builder view.
Exporting of data | ✅ | Exports raw trial-level data. Also exports parameters (both summary statistics and full posterior distribution)
Run simulated experiments | ✅ | 
Choose design and models via GUI | ❌ | At this point we rely on very minor changes to the PsychoPy code snippets to select model and design preferences.

## Design paradigms
Paradigm | Status | Info
--- | --- | ---
Kirby paradigm | ✅ | Fixed design, delay discounting only
Frye et al paradigm | ✅ | Adaptive (heuristic) approach, delay discounting only
BAD | ❌ | PREPRINT: Vincent, B. T., & Rainforth, T. (2017, October 20). The DARC Toolbox: automated, flexible, and efficient delayed and risky choice experiments using Bayesian adaptive design. Retrieved from psyarxiv.com/yehjb

## Delayed reward paradigm models
Model | Status | Info
--- | --- | ---
Hyperbolic | ✅ | 
Exponential | ✅ | (Ben needs to double check)
HyperbolicMagnitudeEffect | ✅ | (Ben needs to double check)
ExponentialMagnitudeEffect | ✅ | (Ben needs to double check)
ConstantSensitivity | ❌ | (Ben needs to double check) negative b values
ProportionalDifference | ✅ | (Ben needs to double check)
HyperbolicNonLinearUtility | ❌ | (Ben needs to double check)

## Risky reward paradigm models
Model | Status | Info
--- | --- | ---
Hyperbolic | ❌ | 
Prospect Theory | ❌ | 
Proportional difference | ❌ |


## Delayed and risky reward paradigm models
Model | Status | Info
--- | --- | ---
MultiplicativeHyperbolic | ❌ | 
Proportional difference | ❌ |


# Requirements 🔥

NOTE: Advice on versions and compatibility etc will be improved over time.

This code is being developed with:
- Python 3
- To run experiments you will also need [PsychoPy](http://www.psychopy.org), currently using their Python 3 release, version 1.90.1. I'll probably update to PsychoPy 3 before we make any kind of official release.

# Installation instructions 🔥

1. Ensure you have a Python 3.6 installation. I recommend https://www.anaconda.com/download/
2. Install [PsychoPy](http://www.psychopy.org). Make sure you install a version that works with Python 3. 
3. Download or clone this `darc-experiments-python` repository.
4. Open up PsychoPy... Open the PsychoPy experiments `experiment.psyexp` in the builder view... Run the experiment with the green man symbol... check the auto-saved data in the `\data` folder.

# How to...

(coming soon)

# References

**NOTE:** This work is based on the pre-print below. This is not yet published and is likely to appear in a subtantially altered form.

> Vincent, B. T., & Rainforth, T. (2017, October 20). The DARC Toolbox: automated, flexible, and efficient delayed and risky choice experiments using Bayesian adaptive design. Retrieved from [psyarxiv.com/yehjb](https://psyarxiv.com/yehjb)
