import numpy as np
import pickle

from configs import config
from util.archive import Archive


def write_to_individuals_file(individual):
    with open(config.FOLDER + "individuals.txt", "a") as file:
        file.write(f"{individual.to_file_string()}\n")

def write_to_populations_file(population):
    with open(config.FOLDER + "populations.txt", "a") as file:
        for individual in population:
            file.write(f"{individual.id};")
        file.write("\n")
    with open(config.FOLDER + "experience.pkl", "wb") as file:
        data = [[], []]
        for individual in population:
            data[0].append(individual.id)
            data[1].append(individual.experience)
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def write_to_archive_file(archive: Archive):
    with open(config.FOLDER + "populations.txt", "a") as file:
        for row in archive.archive:
            for individual in row:
                if individual is not None:
                    file.write(f"{individual.id};")
        file.write("\n")
    with open(config.FOLDER + "experience.pkl", "wb") as file:
        data = [[], []]
        for row in archive.archive:
            for individual in row:
                if individual is not None:
                    data[0].append(individual.id)
                    data[1].append(individual.experience)
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def write_to_rng_file(rng):
    rng_state = rng.bit_generator.state
    np.save(config.FOLDER + "rng_state.npy", rng_state)

def write_to_environments_file(text):
    with open(config.FOLDER + "environments.txt", "a") as file:
        file.write(f"{text}\n")