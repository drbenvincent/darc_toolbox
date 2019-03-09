DARC Toolbox Demo: Decision making experiments using Bayesian Adaptive Design

This is a demo PsychoPy experiment. When you run the experiment you will get
a few GUI menu options. Here you can pick the type of experiment:
- delayed choice, also known as inter-temporal choice
- risky choice
- both delayed and risky choice

You will also get to choose which cognitive model you want to estimate the
parameters for in your experiment. The choice of options here will depend upon
the kind of experiment you are running.

The experiment:
Participants get an information screen. In each trial they will choose between
two prospects. These consist of a reward, a delay, and a probability of occurring.
Participants choose using the left or right keys.

The clever thing about the experiment is that the (reward, delay, probability)
attributes for each prospect is chosen in real time, using Bayesian Adaptive
Design. This will mean each trial will reduce our uncertainty in the model
parameters quicker than other approaches.
