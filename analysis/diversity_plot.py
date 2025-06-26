import ast
import os
import concurrent.futures
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from individual import Individual

# Constants
POP_SIZE = 200
EVALS_PER_GEN = 50
REPETITIONS = 20
GENERATIONS = 50
SUB_FOLDER = 'baseline'

LABELS = {
    (-1, 'none', 0): 'Individual learning',
    (8, 'best', 1): 'Social learning - Best - N=1',
    (8, 'best', 8): 'Social learning - Best - N=8',
    (8, 'parent', 1): 'Social learning - Parent',
    (8, 'random', 1): 'Social learning - Random - N=1',
    (8, 'random', 8): 'Social learning - Random - N=8',
    (8, 'similar', 1): 'Social learning - Similar - N=1',
    (8, 'similar', 8): 'Social learning - Similar - N=8',
}

COLORS = {
    (-1, 'none', 0): 'red',
    (8, 'best', 1): 'black',
    (8, 'best', 8): 'grey',
    (8, 'parent', 1): 'orange',
    (8, 'random', 1): 'blue',
    (8, 'random', 8): 'cyan',
    (8, 'similar', 1): 'purple',
    (8, 'similar', 8): 'violet',
}

LINE_STYLES = {
    (-1, 'none', 0): '--',
    (8, 'best', 1): ':',
    (8, 'best', 8): ':',
    (8, 'parent', 1): '-',
    (8, 'random', 1): '-.',
    (8, 'random', 8): '-.',
    (8, 'similar', 1): (0, (3, 1, 1, 1)),
    (8, 'similar', 8): (0, (3, 1, 1, 1)),
}

def get_diversity(folder):
    if not os.path.isdir(folder):
        return None
    if not os.path.isfile(os.path.join(folder, "individuals.txt")):
        return None
    individuals_file = open(folder + "/individuals.txt", "r")
    all_individuals = {individual.split(";")[0]: ast.literal_eval(individual.split(";")[1]) for individual in individuals_file.read().splitlines()}

    populations_file = open(folder + "/populations.txt", "r")
    generations = populations_file.read().splitlines()

    diversity_per_generation = []
    i = 0
    for generation in generations:
        print(i)
        i += 1
        generation_diversity = []
        for individual in generation.split(";")[:-1]:
            hamming_distances = []
            for nested_individual in generation.split(";")[:-1]:
                if individual == nested_individual:
                    continue
                hamming_distances.append(Individual.hamming_distance(all_individuals[individual], all_individuals[nested_individual]))
            generation_diversity.append(sum(hamming_distances) / len(hamming_distances))
        diversity_per_generation.append(generation_diversity)

    # Convert to numpy array for easy computation
    return np.array(diversity_per_generation)

def make_the_plot(inherit, inherit_type, inherit_pool, environment):
    lines = []
    key = (inherit, inherit_type, inherit_pool)
    label = LABELS.get(key)
    color = COLORS.get(key)

    if label is None or color is None:
        return None  # skip unknown combinations

    for repetition in range(1, REPETITIONS + 1):
        data_path = f'results/{SUB_FOLDER}/learn-{EVALS_PER_GEN}_inherit-{inherit}_type-{inherit_type}_pool-{inherit_pool}_environment-{environment}_repetition-{repetition}'
        data_array = get_diversity(data_path)
        line = f"{inherit};{inherit_type};{inherit_pool};{environment};{repetition};{data_array.tolist()}"
        lines.append(line)

        if data_array is None or len(data_array) < GENERATIONS:
            print(f'No data found for {environment}, {inherit}, {inherit_type}, {inherit_pool}, {repetition}')
            print(len(data_array)) if data_array is not None else print("None")
            print()
            continue

    return lines

def main_read():
    data = defaultdict(list)  # Key: (inherit, inherit_type, inherit_pool), Value: list of array_lists

    with open('ajax2.txt', 'r') as file:
        for line in file:
            parts = line.strip().split(';')
            inherit = ast.literal_eval(parts[0])
            inherit_type = parts[1]
            inherit_pool = ast.literal_eval(parts[2])
            array_list = ast.literal_eval(parts[5])
            mean_per_generation = np.mean(array_list, axis=1)

            key = (inherit, inherit_type, inherit_pool)
            data[key].append(mean_per_generation)

    plot_means(data)


def plot_means(data):
    plt.figure(figsize=(16, 10))

    for key, repetitions in data.items():
        # Convert to numpy array
        repetitions = np.array(repetitions)

        # Calculate statistics across repetitions
        mean_per_generation = np.mean(repetitions, axis=0)
        percentile_25 = np.percentile(repetitions, 25, axis=0)
        percentile_75 = np.percentile(repetitions, 75, axis=0)

        label = LABELS.get(key, str(key))
        color = COLORS.get(key, 'black')
        linestyle = LINE_STYLES.get(key, '-')

        generations = np.arange(len(mean_per_generation))

        plt.plot(generations, mean_per_generation, label=label, color=color, linestyle=linestyle, linewidth=2.5)
        plt.fill_between(generations, percentile_25, percentile_75, color=color, alpha=0.15)

    plt.xlabel('Generation', fontsize=18, fontweight='bold')
    plt.ylabel('Mean diversity', fontsize=18, fontweight='bold')
    plt.title('Mean diversity per Generation\nwith Interquartile Range', fontsize=20, fontweight='bold', pad=20)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=14, frameon=True, fancybox=True, framealpha=0.9, loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def main():
    environments = ['simple', 'catch', 'carry', 'steps']
    strategy_keys = list(LABELS.keys())  # 8 total strategies

    result = []

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=32
    ) as executor:
        futures = []
        for i, env in enumerate(environments):
            for idx, key in enumerate(strategy_keys):
                futures.append(executor.submit(make_the_plot, *key, env))

        for future in futures:
            lines = future.result()
            if lines is None:
                continue
            for line in lines:
                result.append(line)


if __name__ == '__main__':
    main()
