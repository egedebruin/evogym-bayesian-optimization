import os

# Force single-threaded execution
# DO THIS BEFORE IMPORTS
os.environ["OMP_NUM_THREADS"] = "1"       # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"       # Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["NUMEXPR_NUM_THREADS"] = "1"   # NumExpr

import pickle
from util import restart_population
from util.logger_setup import logger_setup
from configs import config
from main import get_offspring, run_generation
from selection import Selection
from robot.brain_nn import BrainNN

config.FOLDER = ''
logger_setup()

config.OFFSPRING_SIZE = 100
config.MIN_MUTATION = 0
config.MAX_MUTATION = 0
config.PARALLEL_PROCESSES = 100
config.LEARN_ITERATIONS = 100
config.GLOBAL_CONTROLLER = False
config.MODULAR_NEIGHBOUR_VISION = 2
config.ENVIRONMENT = 'bidirectional'
config.LEARN_METHOD = 'ddpg'
config.DARWINIAN = False
config.PARENT_SELECTION = 'elitist'

BrainNN.set_modular_vision()

results = {
    'inherit': [],
    'repetition': [],
    'individual': [],
    'values': [],
}
for inherit in [(-1, 'none', 0), (1, 'parent', 1)]:
    for repetition in range(1, 21):
        config.FOLDER = f'results/learn-50_inherit-{inherit[0]}_type-{inherit[1]}_pool-{inherit[2]}_environment-bidirectional_method-ddpg_vision-2_repetition-{repetition}/'
        config.INHERIT_SAMPLES = -1
        config.INHERIT_TYPE = 'none'
        config.SOCIAL_POOL = 0

        population, generation_index = restart_population.get_population()
        rng = restart_population.get_rng()
        parent_selection = Selection(config.OFFSPRING_SIZE, config.PARENT_SELECTION)
        offspring = get_offspring(population, generation_index , parent_selection, rng)
        evaluated_offspring, _ = run_generation(offspring, None, rng)

        for i, individual in enumerate(evaluated_offspring):
            results['inherit'].append(inherit[0])
            results['repetition'].append(repetition)
            results['individual'].append(i)
            results['values'].append([value for _, value in individual.experience])

with open("results/bidirectional-delta-results.pkl", "wb") as f:
    pickle.dump(results, f)