import sys
sys.path.insert(0, '/Users/btvincent/git-local/darc-experiments-python')

from darc.designs import Kirby2009, Frye, DARC_Designs
import pytest


def test_Kirby_default_instantiation():
    design_thing = Kirby2009()
    assert isinstance(design_thing, Kirby2009)


def test_Frye_default_instantiation():
    design_thing = Frye()
    assert isinstance(design_thing, Frye)


def test_DARC_Designs_default_instantiation():
    design_thing = DARC_Designs()
    assert isinstance(design_thing, DARC_Designs)
