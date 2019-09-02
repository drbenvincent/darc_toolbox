import pandas as pd
from collections import namedtuple


# define useful data structures
Prospect = namedtuple("Prospect", ["reward", "delay", "prob"])
Design = namedtuple("Design", ["ProspectA", "ProspectB"])
