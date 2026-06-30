import pickle
import random

import numpy as np
import os
import glob


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


def process_file(filename, sample_frac=0.1):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    bodies = data['body']
    inherits = np.array(data['inherit'])

    rel_activities = []
    elongations = []

    for body in bodies:
        rel_activities.append(relative_activity(body))
        elongations.append(elongation(body))

    rel_activities = np.array(rel_activities)
    elongations = np.array(elongations)

    # Indices for each inheritance class
    pos_indices = np.where(inherits == 1)[0]
    neg_indices = np.where(inherits == -1)[0]

    # Random sample from each class
    pos_idx = random.choice(pos_indices)
    neg_idx = random.choice(neg_indices)

    print(f"\nTask: {filename}")
    for idx in [pos_idx, neg_idx]:
        print(f"Inherit: {inherits[idx]}")
        print(f"Elongation: {elongations[idx]}")
        print(f"Relative activity: {rel_activities[idx]}")
        print("Body:")
        print(bodies[idx])


if __name__ == "__main__":
    files = glob.glob("results/journal/*-bodies.pkl")
    if not files:
        print("No *-bodies.pkl files found.")
    else:
        for f in sorted(files):
            process_file(f)
            break
