''' 
Allow an appropriate (design, model) combination to be chosen by the experimenter,
using a simple GUI.

This code is only intended to be used for the PsychoPy demo, to make it easy. 
When you run proper experiments, then you should define your (design, model)
combination in the code to avoid GUI input errors to ensure you are running
exactly what you want to run.
'''


import logging
import numpy as np


# define what is available
expt_type = {'Experiment type': ['delayed', 'risky', 'delayed and risky']}

delay_models_available = ['Hyperbolic', 'Exponential', 'MyersonHyperboloid',
                         'ProportionalDifference']

risky_models_available = ['Hyperbolic',
                          'ProportionalDifference', 'ProspectTheory']

delayed_and_risky_models_available = ['MultiplicativeHyperbolic']


def gui_chooser_for_demo(win, gui, core, event, DARC_Designs, expInfo):
    '''
    Get user choices about (design, model) combination and generate
    the appropriate objects
    '''
    mouse = event.Mouse(win=win)
    hide_window(win, mouse)
    desired_experiment_type = gui_get_desired_experiment_type(gui, core)
    desired_model = gui_get_desired_model(gui, core)
    design_thing, model = act_on_choices(
        desired_experiment_type, desired_model, DARC_Designs, expInfo)
    show_window(win, mouse)
    return (design_thing, model)


def hide_window(win, mouse):
    mouse.setVisible(1)  # ensure mouse is visible
    win.fullscr = False  # not sure if this is necessary
    win.winHandle.set_fullscreen(False) 
    win.winHandle.minimize()
    win.flip()

def show_window(win, mouse):
    win.winHandle.maximize()
    win.winHandle.activate()
    win.fullscr=True
    win.winHandle.set_fullscreen(True)
    win.flip()
    # hide the mouse for the rest of the experiment
    mouse.setVisible(0)


def gui_get_desired_experiment_type(gui, core):
    # expt_type = {'Experiment type': ['delayed', 'risky', 'delayed and risky']}
    dlg = gui.DlgFromDict(dictionary=expt_type,
                          title='Choose your experiment type')
    if dlg.OK == False:
        core.quit()  # user pressed cancel

    desired_experiment_type = expt_type['Experiment type']
    logging.debug(desired_experiment_type)
    return desired_experiment_type


def gui_get_desired_model(gui, core):
    if expt_type['Experiment type'] is 'delayed':
        models_available = delay_models_available

    elif expt_type['Experiment type'] is 'risky':
        models_available = risky_models_available

    elif expt_type['Experiment type'] is 'delayed and risky':     
        models_available = delayed_and_risky_models_available

    model_type = {'Model': models_available}
    dlg = gui.DlgFromDict(dictionary=model_type, title='Choose your model')
    if dlg.OK == False:
        core.quit()  # user pressed cancel 

    desired_model = model_type['Model']
    logging.debug(desired_model)
    return desired_model


def act_on_choices(desired_experiment_type, desired_model, DARC_Designs, expInfo):

    # create desired experiment object

    if desired_experiment_type is 'delayed':
        # create an appropriate design object
        design_thing = DARC_Designs(max_trials=expInfo['trials'])
        # import the appropriate set of models
        from darc.delayed import models

    elif desired_experiment_type is 'risky':
        # create an appropriate design object
        prob_list = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99]

        design_thing = DARC_Designs(max_trials=expInfo['trials'],
                                    DA=[0], DB=[0], PA=[1], PB=prob_list,
                                    RA=list(100*np.linspace(0.05, 0.95, 91)),
                                    RB=[100])
        # import the appropriate set of models
        from darc.risky import models

    elif desired_experiment_type is 'delayed and risky':
        # create an appropriate design object
        design_thing = DARC_Designs(max_trials=expInfo['trials'],
                                    PB=[0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99])
        # import the appropriate set of models
        from darc.delayed_and_risky import models


    # chose the desired model here
    if desired_model is 'Hyperbolic':
        model = models.Hyperbolic(n_particles=expInfo['particles'])

    elif desired_model is 'Exponential':
        model = models.Exponential(n_particles=expInfo['particles'])

    elif desired_model is 'MyersonHyperboloid':
        model = models.Exponential(n_particles=expInfo['particles'])

    elif desired_model is 'ProportionalDifference':
        model = models.ProportionalDifference(n_particles=expInfo['particles'])

    elif desired_model is 'ProspectTheory':
        model = models.ProspectTheory(n_particles=expInfo['particles'])

    elif desired_model is 'MultiplicativeHyperbolic':
        model = models.MultiplicativeHyperbolic(
            n_particles=expInfo['particles'])

    return (design_thing, model)
