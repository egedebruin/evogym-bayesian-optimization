import math

import numpy as np
import matplotlib.pyplot as plt

import plot

plt.figure(figsize=(8, 5))
pop_size = 100

to_label = {
    (-1, 1): 'Evolution only',
    (-1, 30): 'Learn - No inheritance',
    (0, 30): 'Learn - Inherit samples',
    (5, 30): 'Learn - Reevaluate'
}
to_color = {
    (-1, 1): 'red',
    (-1, 30): 'blue',
    (0, 30): 'green',
    (5, 30): 'orange'
}

max_y = []
for inherit in [-1, 0, 5]:
    for generations, learn in [(1000, 1), (33, 30)]:
        print("Plotting for inherit {} and learn {}".format(inherit, learn))
        if learn == 1 and inherit != -1:
            continue
        to_plot = []
        for repetition in range(1, 21):
            data_array = plot.get_data(f'results/learn-{str(learn)}_inherit-{str(inherit)}_repetition-{str(repetition)}')
            if len(data_array) < generations:
                print("There is something wrong with: ", learn, inherit, repetition)
                continue

            max_values = np.max(data_array, axis=1)
            mean_values = np.mean(data_array, axis=1)
            max_so_far = np.array([])
            for value in max_values:
                max_so_far = np.append(max_so_far, max(value, max(list(max_so_far) + [-math.inf])))

            to_plot.append(max_so_far)

        function_evaluations = np.array(range(1, generations + 1))
        function_evaluations = function_evaluations * learn * pop_size
        mean_to_plot = np.mean(to_plot, axis=0)
        q1 = np.percentile(to_plot, 1, axis=0)
        q25 = np.percentile(to_plot, 25, axis=0)
        q75 = np.percentile(to_plot, 75, axis=0)
        q99 = np.percentile(to_plot, 99, axis=0)
        plt.plot(function_evaluations, mean_to_plot, label=to_label[(inherit, learn)], color=to_color[(inherit, learn)])
        plt.fill_between(function_evaluations, q25, q75, alpha=0.3, color=to_color[(inherit, learn)])
        # plt.fill_between(function_evaluations, q1, q99, alpha=0.1, color=to_color[(inherit, learn)])
        # plt.plot(function_evaluations, q1, linestyle='dashed', color=to_color[(inherit, learn)])
        # plt.plot(function_evaluations, q99, linestyle='dashed', color=to_color[(inherit, learn)])
        max_y.append(max(q99) * 1.1)

plt.ylim(0, max(max_y))
plt.xlabel("Function evaluations")
plt.ylabel("Objective value")
plt.legend()
plt.title("Max performance averaged over repetitions")
plt.show()