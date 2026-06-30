import pickle
import numpy as np
import matplotlib.pyplot as plt
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

def symmetry(body: np.ndarray, axis: str = 'horizontal') -> float:
    """
    Computes how symmetrical a body is.
    Ignores 0s by cropping to the bounding box of non-zero elements.
    axis: 'horizontal' (left-right) or 'vertical' (top-bottom).
    """
    coords = np.argwhere(body > 0)
    if len(coords) == 0:
        return 0.0
        
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    cropped = body[r_min:r_max+1, c_min:c_max+1]
    
    if axis == 'horizontal':
        flipped = np.fliplr(cropped)
    elif axis == 'vertical':
        flipped = np.flipud(cropped)
    else:
        raise ValueError("axis must be 'horizontal' or 'vertical'")
        
    return np.mean(cropped == flipped)


def process_file(filename, sample_frac=0.1):
    print(f"Processing {filename}...")

    with open(filename, "rb") as f:
        data = pickle.load(f)

    inherits = np.array(data['inherit'])
    bodies = data['body']

    rel_activities = []
    elongations = []

    for body in bodies:
        rel_activities.append(relative_activity(body))
        elongations.append(elongation(body))

    rel_activities = np.array(rel_activities)
    elongations = np.array(elongations)

    # ---- RANDOM 10% SAMPLING ----
    n = len(bodies)
    sample_size = max(1, int(n * sample_frac))
    sample_idx = np.random.choice(n, size=sample_size, replace=False)

    inherits = inherits[sample_idx]
    rel_activities = rel_activities[sample_idx]
    elongations = elongations[sample_idx]

    # Masks after sampling
    lamarckian_mask = (inherits == 1)
    darwinian_mask = (inherits == -1)

    # Environment name from file
    environment = os.path.basename(filename).replace(".pkl", "").split("-")[0]

    # Output file
    if not os.path.exists("txts"):
        os.makedirs("txts")

    out_filename = os.path.join("txts", "bodies", environment + ".txt")

    with open(out_filename, "w") as out:
        out.write("environment\tstrategy\trelative_activity\telongation\n")

        for ra, el in zip(rel_activities[darwinian_mask],
                          elongations[darwinian_mask]):
            out.write(f"{environment}\tdarwin\t{ra}\t{el}\n")

        for ra, el in zip(rel_activities[lamarckian_mask],
                          elongations[lamarckian_mask]):
            out.write(f"{environment}\tlamarck\t{ra}\t{el}\n")

    print(f"Saved: {out_filename}")

if __name__ == "__main__":
    files = glob.glob("results/journal/*-bodies.pkl")
    if not files:
        print("No *-bodies.pkl files found.")
    else:
        for f in sorted(files):
            process_file(f)
