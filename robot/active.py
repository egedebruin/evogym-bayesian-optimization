import config

if config.BRAIN_TYPE == 'nn':
    from robot.brain_nn import BrainNN as Brain
    from robot.controller_nn import ControllerNN as Controller
elif config.BRAIN_TYPE == 'sine':
    from robot.brain_sine import BrainSine as Brain
    from robot.controller_sine import ControllerSine as Controller
else:
    raise ValueError('BRAIN_TYPE does not exist')