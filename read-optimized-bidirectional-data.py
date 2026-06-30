import pickle
import numpy as np

def main():
    # Load data
    with open("bidirectional-delta-results-with-inheritance.pkl", "rb") as f:
        results_1 = pickle.load(f)

    results_2 = None
    # with open("bidirectional2-delta-results-from-scratch-2.pkl", "rb") as f:
    #     results_2 = pickle.load(f)

    # Separate indices
    lamarckian = []
    darwinian = []
    for i, inherit in enumerate(results_1['inherit']):
        if inherit == 1:
            lamarckian.append(i)
        else:
            darwinian.append(i)

    # Collect data arrays
    lamarckian_data = np.concatenate(
        [np.array([
            results_1['values'][i][:-1]
            for i in lamarckian
        ])] +
        ([] if results_2 is None else [np.array([
            results_2['values'][i][:-1]
            for i in lamarckian
        ])]),
        axis=0
    )

    darwinian_data = np.concatenate(
        [np.array([
            results_1['values'][i][:-1]
            for i in darwinian
        ])] +
        ([] if results_2 is None else [np.array([
            results_2['values'][i][:-1]
            for i in darwinian
        ])]),
        axis=0
    )

    for label, data in [("Lamarckian", lamarckian_data), ("Darwinian", darwinian_data)]:
        output_filename = f"{label}.txt"
        with open(output_filename, "w") as f:
            # Header: cat, x, y, ymin, ymax (tab separated)
            f.write("cat\tx\ty\tymin\tymax\n")
            
            # Compute stats: y is median, ymin is 25th percentile, ymax is 75th percentile
            y_values = np.median(data, axis=0)
            ymin_values = np.percentile(data, 25, axis=0)
            ymax_values = np.percentile(data, 75, axis=0)
            x_values = np.arange(len(y_values)) + 1
            
            for x, y, ymin, ymax in zip(x_values, y_values, ymin_values, ymax_values):
                f.write(f"{label}\t{x}\t{y}\t{ymin}\t{ymax}\n")
        print(f"Successfully created {output_filename}")

if __name__ == "__main__":
    main()
