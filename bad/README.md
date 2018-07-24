# The `bad` package

This code is based upon the preprint:
> Vincent, B. T., & Rainforth, T. (2017, October 20). The DARC Toolbox: automated, flexible, and efficient delayed and risky choice experiments using Bayesian adaptive design. Retrieved from [psyarxiv.com/yehjb](https://psyarxiv.com/yehjb)

The original code was implemented by Tom Rainforth and Benjamin T. Vincent in Matlab. Ben then later ported it to Python.

The `bad` package is a set of core functions and base classes for Bayesian Adaptive Design.

Everything here is very general. It is so general that it will not actually accomplish anything in isolation. 

It has been developed in conjunction with the `darc` package which builds upon the `bad` package to implement Bayesian Adaptive Design in the context of Delayed And Risky Choice tasks. 

The `bad` package itself should only be of interest to people interested in how the Bayesian Adaptive Design works, or who want to implement it in a different (non-DARC) experiment context. Even so, you should look at the `darc` package to see how it is implemented.

