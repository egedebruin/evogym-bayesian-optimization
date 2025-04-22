import config

if config.CONTROLLER_TYPE == 'nn':
    from robot.brain_nn import BrainNN as Brain
    from robot.controller_nn import ControllerNN as Controller
elif config.CONTROLLER_TYPE == 'sine':
    from robot.brain_sine import BrainSine as Brain
    from robot.controller_sine import ControllerSine as Controller
else:
    raise ValueError('CONTROLLER_TYPE does not exist')