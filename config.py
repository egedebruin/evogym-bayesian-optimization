FOLDER = 'results/test/'

READ_ARGS = True
LEARN_ITERATIONS = 10
INHERIT_SAMPLES = 5

POP_SIZE = 200
OFFSPRING_SIZE = 200
FUNCTION_EVALUATIONS = 500000
SIMULATION_LENGTH = 500 # 10 seconds
PARALLEL_PROCESSES = 100

SURVIVOR_SELECTION = 'generational'
PARENT_SELECTION = 'tournament'
PARENT_POOL = 4
MUTATION_STD = 0.1
CONTROLLER_DT = 0.1

GRID_LENGTH = 5
MIN_INITIAL_SIZE = 10
MAX_INITIAL_SIZE = 20
MIN_SIZE = 5
MAX_SIZE = 25
MAX_ADD_MUTATION = 3
MAX_DELETE_MUTATION = 3
MAX_CHANGE_MUTATION = 3

LEARN_LENGTH_SCALE = 0.2
LEARN_NU = 5/2
LEARN_ALPHA = 1e-10
LEARN_INHERITED_ALPHA = 2
LEARN_KAPPA = 3