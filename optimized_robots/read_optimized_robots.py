import pickle
import matplotlib.pyplot as plt

result_lists = []
for i in range(1, 6):
    with open(f"optimized_robots/results{i}.pkl", "rb") as f:
        result_list = pickle.load(f)
        result_lists.append(result_list)

changes = [0]

changes_to_body_darwinian = {
    0: [],
    1: [],
    2: [],
    3: [],
    5: [],
    10: []
}

changes_to_body_lamarckian = {
    0: [],
    1: [],
    2: [],
    3: [],
    5: [],
    10: []
}

for result_list in result_lists:
    for entry in result_list:
        if entry['changes'] in changes:
            continue

        if entry['lamarckian']:
            changes_to_body_lamarckian[entry['changes']].append(entry['qualities'])
        else:
            changes_to_body_darwinian[entry['changes']].append(entry['qualities'])
print(len(changes_to_body_lamarckian[1]))
for key, darwinian_values in changes_to_body_darwinian.items():
    lamarckian_values = changes_to_body_lamarckian[key]
    averaged_values_darwinian = [sum(value)/len(value) for value in zip(*darwinian_values)]
    averaged_values_lamarckian = [sum(value)/len(value) for value in zip(*lamarckian_values)]
    averaged_values = []
    for i, darwinian_value in enumerate(averaged_values_darwinian):
        averaged_values.append(averaged_values_lamarckian[i] - darwinian_value)
    plt.plot(range(len(averaged_values)), averaged_values, label=key)
plt.legend()
plt.show()