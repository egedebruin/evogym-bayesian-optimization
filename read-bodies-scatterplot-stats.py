import glob
import pickle
import numpy as np
import os
from scipy.stats import mannwhitneyu

def cliffs_delta_from_u(U, n1, n2):
    return (2 * U) / (n1 * n2) - 1

def relative_activity(body: np.ndarray):
    """
    Computes relative activity of a body.
    Assumes body > 2 means active and body > 0 means existing.
    """
    existing = np.count_nonzero(body > 0)
    if existing == 0:
        return 0.0
    return np.count_nonzero(body > 2) / existing

def elongation(body: np.ndarray, n_directions=20) -> float:
    if n_directions <= 0:
        raise ValueError("n_directions must be positive")
    diameters = []
    coordinates = np.where(body.transpose() > 0)
    x_coordinates = coordinates[0]
    y_coordinates = coordinates[1]
    if len(x_coordinates) == 0 or len(y_coordinates) == 0:
        return 0.0
    for i in range(n_directions):
        theta = i * 2 * np.pi / n_directions
        rotated_x_coordinates = x_coordinates * np.cos(theta) - y_coordinates * np.sin(theta)
        rotated_y_coordinates = x_coordinates * np.sin(theta) + y_coordinates * np.cos(theta)
        x_side = np.max(rotated_x_coordinates) - np.min(rotated_x_coordinates) + 1
        y_side = np.max(rotated_y_coordinates) - np.min(rotated_y_coordinates) + 1
        diameter = min(x_side, y_side) / max(x_side, y_side)
        diameters.append(diameter)

    return 1 - min(diameters)


def process_file(filename):
    print(f"Processing {filename}...")

    with open(filename, "rb") as f:
        data = pickle.load(f)

    inherits = np.array(data['inherit'])
    bodies = data['body']

    rel_activities = np.array([
        relative_activity(body) for body in bodies
    ])

    elongations = np.array([
        elongation(body) for body in bodies
    ])

    # Split by inheritance strategy
    darwin_mask = (inherits == -1)
    lamarck_mask = (inherits == 1)

    darwin_ra = rel_activities[darwin_mask]
    lamarck_ra = rel_activities[lamarck_mask]

    darwin_el = elongations[darwin_mask]
    lamarck_el = elongations[lamarck_mask]

    ra_stat, ra_p = mannwhitneyu(
        darwin_ra,
        lamarck_ra,
        alternative='two-sided',
        method = 'asymptotic'
    )

    el_stat, el_p = mannwhitneyu(
        darwin_el,
        lamarck_el,
        alternative='two-sided',
        method='asymptotic'
    )

    ra_delta = cliffs_delta_from_u(ra_stat, len(darwin_ra), len(lamarck_ra))
    el_delta = cliffs_delta_from_u(el_stat, len(darwin_el), len(lamarck_el))

    environment = os.path.basename(filename)\
                    .replace(".pkl", "")\
                    .split("-")[0]

    if not os.path.exists("txts"):
        os.makedirs("txts")

    out_filename = os.path.join(
        f"{environment}_stats.txt"
    )

    with open(out_filename, "w") as out:

        out.write("Relative activity:\n")
        out.write(f"  Darwin mean:   {darwin_ra.mean():.6f}\n")
        out.write(f"  Lamarck mean:  {lamarck_ra.mean():.6f}\n")
        out.write(f"  U-statistic:   {ra_stat:.6f}\n")
        out.write(f"  p-value:       {ra_p:.6e}\n")
        out.write(f"  Cliff's delta: {ra_delta:.6f}\n")
        out.write(f"  Significant:   {ra_p < 0.05}\n\n")

        out.write("Elongation:\n")
        out.write(f"  Darwin mean:   {darwin_el.mean():.6f}\n")
        out.write(f"  Lamarck mean:  {lamarck_el.mean():.6f}\n")
        out.write(f"  U-statistic:   {el_stat:.6f}\n")
        out.write(f"  p-value:       {el_p:.6e}\n")
        out.write(f"  Cliff's delta: {el_delta:.6f}\n")
        out.write(f"  Significant:   {el_p < 0.05}\n")

    print(f"Saved: {out_filename}")

if __name__ == "__main__":
    files = glob.glob("results/journal/*-bodies.pkl")
    if not files:
        print("No *-bodies.pkl files found.")
    else:
        for f in sorted(files):
            process_file(f)