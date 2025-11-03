import ast
import os
import numpy as np
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor

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

REPETITIONS = 20
ENVIRONMENTS = ['simple', 'steps', 'carry', 'catch']

def process_repetition(args):
    label, environment, repetition = args
    folder = f'results/learn-50_inherit-{label[0]}_type-{label[1]}_pool-{label[2]}_environment-{environment}_repetition-{repetition}'
    individuals_path = os.path.join(folder, "individuals.txt")
    populations_path = os.path.join(folder, "populations.txt")

    if not os.path.exists(individuals_path) or not os.path.exists(populations_path):
        return None

    try:
        with open(individuals_path, "r") as f:
            individuals_file = f.read().splitlines()
            all_individuals = {
                line.split(";")[0]: (float(line.split(";")[5]) - ast.literal_eval(line.split(";")[7])[1], float(line.split(";")[5]))
                for line in individuals_file
            }

        with open(populations_path, "r") as f:
            generations = f.read().splitlines()

        learning_delta_per_generation = []
        for generation in generations:
            max_individual = None
            max_value = float('-inf')
            for individual in generation.split(";")[:-1]:
                value = all_individuals[individual][1]
                if value > max_value:
                    max_value = value
                    max_individual = individual
            if max_individual is not None:
                learning_delta_per_generation.append(all_individuals[max_individual][0])

        return np.array(learning_delta_per_generation)
    except Exception:
        return None

def main():
    plt.figure(figsize=(12, len(ENVIRONMENTS) * 4))

    for env_idx, environment in enumerate(ENVIRONMENTS):
        plt.subplot(len(ENVIRONMENTS), 1, env_idx + 1)

        for label, label_name in LABELS.items():
            all_runs = []
            args = [(label, environment, rep) for rep in range(1, REPETITIONS + 1)]

            with ProcessPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(process_repetition, args))

            for result in results:
                if result is not None:
                    all_runs.append(result)

            if len(all_runs) == 0:
                continue

            all_runs = np.array(all_runs)
            avg_over_repetitions = np.mean(all_runs, axis=0)

            x_vals = np.arange(1, len(avg_over_repetitions) + 1)
            plt.plot(x_vals, avg_over_repetitions, label=label_name)

        plt.title(f'Environment: {environment}')
        plt.xlabel('Generation')
        plt.ylabel('Mean Learning Delta')
        plt.legend()

    plt.tight_layout()
    plt.savefig('learning-delta.pdf')

if __name__ == '__main__':
    main()
