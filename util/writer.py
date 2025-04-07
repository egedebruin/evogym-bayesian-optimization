import numpy as np

import config

def write_to_individuals_file(individual):
    with open(config.FOLDER + "individuals.txt", "a") as file:
        file.write(f"{individual.to_file_string()}\n")

def write_to_populations_file(population):
    with open(config.FOLDER + "populations.txt", "a") as file:
        for individual in population:
            file.write(f"{individual.id};")
        file.write("\n")

def write_to_rng_file(rng):
    rng_state = rng.bit_generator.state
    np.save(config.FOLDER + "rng_state.npy", rng_state)