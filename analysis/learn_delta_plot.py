import ast
import os
import numpy as np
from matplotlib import pyplot as plt

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

def main():

    plt.figure(figsize=(12, len(ENVIRONMENTS) * 4))

    for env_idx, environment in enumerate(ENVIRONMENTS):
        plt.subplot(len(ENVIRONMENTS), 1, env_idx + 1)

        for label, label_name in LABELS.items():
            all_runs = []

            for repetition in range(REPETITIONS):
                folder = f'results/learn-50_inherit-{label[0]}_type-{label[1]}_pool-{label[2]}_environment-{environment}_repetition-{repetition}'
                individuals_path = os.path.join(folder, "individuals.txt")
                populations_path = os.path.join(folder, "populations.txt")

                if not os.path.exists(individuals_path) or not os.path.exists(populations_path):
                    # Skip if either file is missing
                    continue

                # Load individual learning deltas
                with open(individuals_path, "r") as f:
                    individuals_file = f.read().splitlines()
                    all_individuals = {
                        line.split(";")[0]: (float(line.split(";")[5]) - ast.literal_eval(line.split(";")[7])[1], float(line.split(";")[5]))
                        for line in individuals_file
                    }

                # Load population generations
                with open(populations_path, "r") as f:
                    generations = f.read().splitlines()

                learning_delta_per_generation = []
                for generation in generations:
                    max_individual = None
                    max_value = float('-inf')
                    for individual in generation.split(";")[:-1]:
                        value = all_individuals[individual][1]  # Index 1 is used for comparison
                        if value > max_value:
                            max_value = value
                            max_individual = individual
                    if max_individual is not None:
                        # Save the learning delta (index 0) of the best individual
                        learning_delta_per_generation.append(all_individuals[max_individual][0])

                learning_delta_per_generation = np.array(learning_delta_per_generation)

                all_runs.append(learning_delta_per_generation)

            if len(all_runs) == 0:
                # Skip plotting if no data was found for this strategy
                continue

            # Average over repetitions
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
