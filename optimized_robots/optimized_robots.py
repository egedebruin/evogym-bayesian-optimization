import os

# Force single-threaded execution
# DO THIS BEFORE IMPORTS
os.environ["OMP_NUM_THREADS"] = "1"       # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"       # Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["NUMEXPR_NUM_THREADS"] = "1"   # NumExpr

import concurrent.futures
import random
import numpy as np
import pickle
import learn
from individual import Individual
from robot.body import Body
from robot.active import Brain
from robot.brain_nn import BrainNN
from util import world
from util import start
from configs import config
from util.logger_setup import logger_setup


def grid_is_ok(grid, max_size=5):
    contains_action = False
    for i in range(max_size):
        for j in range(max_size):
            if grid[i][j] == 3.0 or grid[i][j] == 4.0:
                contains_action = True
    return is_fully_connected(grid) and contains_action

def is_fully_connected(grid):
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)

    # Find a starting point with value between 1 and 4
    start = None
    for i in range(rows):
        for j in range(cols):
            if 1 <= grid[i, j] <= 4:
                start = (i, j)
                break
        if start:
            break

    if not start:
        return False  # No valid starting point (empty grid or no values between 1 and 4)

    # Flood fill (DFS)
    stack = [start]
    visited[start] = True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    while stack:
        x, y = stack.pop()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:  # Within bounds
                if not visited[nx, ny] and 1 <= grid[nx, ny] <= 4:  # Valid connection
                    visited[nx, ny] = True
                    stack.append((nx, ny))

    # Check if all 1-4 values were visited
    return np.all((grid < 1) | (grid > 4) | visited)

def create_new_grid(grid, change):
    while True:
        change_indices = random.sample(range(25), change)

        new_grid = []
        i = 0
        for horizontal in grid:
            new_horizontal = []
            for vertical in horizontal:
                new_value = vertical

                if i in change_indices:
                    while new_value == vertical:
                        new_value = random.sample(range(5), 1)[0]

                i += 1
                new_horizontal.append(new_value)
            new_grid.append(new_horizontal)

        if grid_is_ok(np.array(new_grid)):
            break

    return np.array(new_grid)

def do_it(rng, old_grid, i, new_grids):
    result = []
    current_world, _ = world.get_environment(rng)
    body = Body()
    body.replace_grid(old_grid)
    brain = Brain(5, rng)
    old_individual = Individual(f'{i}', body, brain, 0, [])
    experience, _, _ = learn.learn(old_individual, rng, current_world)
    old_individual.experience = experience

    result.append(
        {
            'robot_id': i,
            'changes': 0,
            'qualities': [t[1] for t in experience],
            'lamarckian': False
        }
    )

    for change_number, new_grid in new_grids.items():
        current_world, _ = world.get_environment(rng)
        new_body = Body()
        new_body.replace_grid(new_grid)
        new_brain = Brain(5, rng)
        new_individual = Individual(f'{i},{change_number}', new_body, new_brain, 0, [])
        new_individual.inherit_experience([], old_individual, rng)
        experience, _, _ = learn.learn(new_individual, rng, current_world)

        result.append(
            {
                'robot_id': i,
                'changes': change_number,
                'qualities': [t[1] for t in experience],
                'lamarckian': True
            }
        )

    for change_number, new_grid in new_grids.items():
        current_world, _ = world.get_environment(rng)
        new_body = Body()
        new_body.replace_grid(new_grid)
        new_brain = Brain(5, rng)
        new_individual = Individual(f'{i},{change_number}', new_body, new_brain, 0, [])
        experience, _, _ = learn.learn(new_individual, rng, current_world)

        result.append(
            {
                'robot_id': i,
                'changes': change_number,
                'qualities': [t[1] for t in experience],
                'lamarckian': False
            }
        )
    return result

def main():
    with open(f"optimized_robots/best_robots.pkl", "rb") as f:
        robots_we_gon_do = np.array(pickle.load(f))

    config.GLOBAL_CONTROLLER = False
    config.MODULAR_NEIGHBOUR_VISION = 2
    config.ENVIRONMENT = 'simple'
    config.LEARN_ITERATIONS = 50
    config.FOLDER = ''
    config.INHERIT_TYPE = 'parent'
    config.SOCIAL_POOL = 1
    config.LEARN_METHOD = 'ddpg'
    config.INHERIT_SAMPLES = 1
    change_numbers = [1, 2, 3, 5, 10]

    logger_setup()
    rng = start.make_rng_seed()

    for repetition in range(1, 6):
        result = []
        new_grids = []

        for old_grid in robots_we_gon_do:
            nested_new_grids = {}
            for change_number in change_numbers:
                nested_new_grids[change_number] = create_new_grid(old_grid, change_number)
            new_grids.append(nested_new_grids)

        BrainNN.set_modular_vision()
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=10
        ) as executor:
            futures = []
            i = 0
            for old_grid, changed_grids in list(zip(robots_we_gon_do, new_grids)):
                futures.append(executor.submit(do_it, rng, old_grid, i, changed_grids))
                i += 1

        for future in futures:
            result += future.result()
        with open(f"results{repetition}.pkl", "wb") as f:
            pickle.dump(result, f)

if __name__ == '__main__':
    main()