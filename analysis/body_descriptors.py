import ast
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_best_individual(folder):
    best_individual = None
    best_fitness = float("-inf")

    file_path = os.path.join(folder, "individuals.txt")

    if not os.path.exists(file_path):
        return False  # no file found

    with open(file_path, "r") as file:
        for line in file:
            if not line.strip():
                continue  # Skip empty lines
            individual = line.strip().split(";")
            try:
                fitness = float(individual[5])
            except (IndexError, ValueError) as e:
                print(f"Skipping malformed line: {line.strip()} — {e}")
                continue
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual

    return best_individual


def relative_activity(body: np.ndarray):
    return np.count_nonzero(body > 2) / np.count_nonzero(body > 0)

def compactness(body: np.ndarray) -> float:
    convex_hull = body > 0
    if True not in convex_hull:
        return 0.0
    new_found = True
    while new_found:
        new_found = False
        false_coordinates = np.argwhere(convex_hull == False)
        for coordinate in false_coordinates:
            x, y = coordinate[0], coordinate[1]
            adjacent_count = 0
            adjacent_coordinates = []
            for d in [-1, 1]:
                adjacent_coordinates.append((x, y + d))
                adjacent_coordinates.append((x + d, y))
                adjacent_coordinates.append((x + d, y + d))
                adjacent_coordinates.append((x + d, y - d))
            for adj_x, adj_y in adjacent_coordinates:
                if 0 <= adj_x < body.shape[0] and 0 <= adj_y < body.shape[1] and convex_hull[adj_x][adj_y]:
                    adjacent_count += 1
            if adjacent_count >= 5:
                convex_hull[x][y] = True
                new_found = True

    return (body > 0).sum() / convex_hull.sum()

LABELS = {
    (-1, 'none', 0): 'Individual learning',
    (8, 'parent', 1): 'Parent',
    (8, 'best', 1): 'Best - N=1',
    (8, 'best', 8): 'Best - N=8',
    (8, 'random', 1): 'Random - N=1',
    (8, 'random', 8): 'Random - N=8',
    (8, 'similar', 1): 'Similar - N=1',
    (8, 'similar', 8): 'Similar - N=8',
}

COLORS = {
    (-1, 'none', 0): 'red',
    (8, 'best', 1): 'black',
    (8, 'best', 8): 'grey',
    (8, 'parent', 1): 'orange',
    (8, 'random', 1): 'blue',
    (8, 'random', 8): 'cyan',
    (8, 'similar', 1): 'purple',
    (8, 'similar', 8): 'pink',
}

def main():
    fig, axes = plt.subplots(ncols=4, figsize=(14, 4), sharey=True)

    environments = ['simple', 'steps', 'carry', 'catch']

    for i, environment in enumerate(environments):
        csv_data = {
            'environment': [],
            'strategy': [],
            'relative_activity': [],
            'compactness': [],
        }
        for strategy in LABELS.keys():
            for repetition in range(1, 21):
                best_individual = get_best_individual(
                    f'results/learn-50_inherit-{strategy[0]}_type-{strategy[1]}_pool-{strategy[2]}_environment-{environment}_repetition-{repetition}/'
                )
                if not best_individual:
                    continue

                grid = np.array(ast.literal_eval(best_individual[1]))
                csv_data['environment'].append(environment)
                csv_data['strategy'].append(LABELS[strategy])
                csv_data['relative_activity'].append(relative_activity(grid))
                csv_data['compactness'].append(compactness(grid))

                # axes[i].scatter(
                #     relative_activity(grid),
                #     compactness(grid),
                #     c=COLORS[strategy],
                #     s=15,
                #     alpha=0.6,
                #     label=LABELS[strategy] if repetition == 1 else None  # add label only once
                # )
        pd.DataFrame(csv_data).to_csv(f'descriptors-{environment}.txt', index=False, sep='\t')
        # axes[i].set_xlim(0, 1)
        # axes[i].set_ylim(0, 1)
        # axes[i].set_title(environment.capitalize(), fontsize=12)
        # axes[i].set_xlabel("Relative activity")

    # axes[0].set_ylabel("Compactness")

    # Create one legend for the whole figure
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.15),
    #     ncol=4,
    #     fontsize=9,
    #     frameon=False
    # )
    #
    # fig.tight_layout(rect=[0, 0, 1, 1])  # leave space for legend
    #
    # # plt.savefig("descriptors.pdf")
    # plt.show()


if __name__ == "__main__":
    main()