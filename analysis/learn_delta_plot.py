import ast
import os
import numpy as np
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def load_run(environment, label, label_name, repetition):
    folder = f'results/learn-50_inherit-{label[0]}_type-{label[1]}_pool-{label[2]}_environment-{environment}_repetition-{repetition}'
    individuals_path = os.path.join(folder, "individuals.txt")
    populations_path = os.path.join(folder, "populations.txt")

    if not os.path.isfile(individuals_path) or not os.path.isfile(populations_path):
        return None

    try:
        # Parse individuals
        with open(individuals_path, "r") as f:
            all_individuals = {}
            for line in f:
                parts = line.strip().split(";")
                if len(parts) < 8:
                    continue
                # Fast tuple parsing instead of ast.literal_eval
                baseline_str = parts[7]
                baseline = float(baseline_str.strip("() ").split(",")[1])
                eval_score = float(parts[5])
                delta = eval_score - baseline
                all_individuals[parts[0]] = (delta, eval_score)

        # Parse populations
        learning_deltas = []
        with open(populations_path, "r") as f:
            for line in f:
                individuals = line.strip().split(";")[:-1]
                best = max(
                    ((iid, all_individuals[iid][1]) for iid in individuals if iid in all_individuals),
                    key=lambda x: x[1],
                    default=(None, None)
                )
                if best[0]:
                    learning_deltas.append(all_individuals[best[0]][0])

        return np.array(learning_deltas)

    except Exception as e:
        print(f"Error processing {folder}: {e}")
        return None


def main():
    plt.figure(figsize=(12, len(ENVIRONMENTS) * 4))

    for env_idx, environment in enumerate(ENVIRONMENTS):
        plt.subplot(len(ENVIRONMENTS), 1, env_idx + 1)

        for label, label_name in LABELS.items():
            all_runs = []

            # Use thread pool to load all repetitions in parallel
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(load_run, environment, label, label_name, repetition)
                    for repetition in range(1, REPETITIONS + 1)
                ]

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None and len(result) > 0:
                        all_runs.append(result)

            if not all_runs:
                continue

            min_len = min(len(run) for run in all_runs)
            all_runs = np.array([run[:min_len] for run in all_runs])
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
