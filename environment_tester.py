import concurrent.futures
from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern
import pandas as pd

from configs import config
from robot.body import Body
from robot.active import Brain
from robot.active import Controller
from robot.sensors import Sensors
from util import start, world

def main():
    environments = ['simple', 'jump', 'carry', 'steps', 'climb']
    n_robots = 20
    repetitions = 20
    learn_iterations = 100
    rng = start.make_rng_seed()

    result_dict = {
        'environment': [],
        'robot_id': [],
        'learn_iteration': [],
        'objective_value': [],
        'repetition': []
    }

    for i in range(n_robots):
        body_size = rng.integers(config.MIN_INITIAL_SIZE, config.MAX_INITIAL_SIZE + 1)
        body = Body(config.GRID_LENGTH, body_size, rng)
        for environment in environments:
            config.ENVIRONMENT = environment
            result = parallelize(rng, body, learn_iterations, repetitions)
            for repetition in range(len(result)):
                for learn_iteration in range(len(result[repetition])):
                    result_dict['environment'].append(environment)
                    result_dict['robot_id'].append(i)
                    result_dict['learn_iteration'].append(learn_iteration)
                    result_dict['objective_value'].append(result[repetition][learn_iteration])
                    result_dict['repetition'].append(repetition)

    pd.DataFrame(result_dict).to_csv('environment_tester_results.csv', index=False)

def parallelize(rng, body, learn_iterations, repetitions):
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=repetitions
    ) as executor:
        futures = []
        for repetition in range(repetitions):
            futures.append(executor.submit(learn, rng, body, learn_iterations))

    result = []
    for future in futures:
        result.append(future.result())
    return result

def learn(rng, body, learn_iterations):
    objective_values = []
    brain = Brain(config.GRID_LENGTH, rng)

    sim, viewer = world.build_world(body.grid)
    actuator_indices = sim.get_actuator_indices('robot')
    optimizer = BayesianOptimization(
        f=None,
        pbounds=brain.get_p_bounds(actuator_indices),
        allow_duplicate_points=True,
        random_state=int(rng.integers(low=0, high=2 ** 32)),
        acquisition_function=acquisition.UpperConfidenceBound(kappa=config.LEARN_KAPPA,
                                                              random_state=int(
                                                                  rng.integers(low=0, high=2 ** 32)))
    )
    optimizer.set_gp_params(
        kernel=Matern(nu=config.LEARN_NU, length_scale=config.LEARN_LENGTH_SCALE, length_scale_bounds="fixed"))
    optimizer.set_gp_params(alpha=1e-10)

    for bayesian_optimization_iteration in range(learn_iterations):
        print(f"Learn generation {bayesian_optimization_iteration + 1}")
        if bayesian_optimization_iteration == 0:
            next_point = brain.to_next_point(actuator_indices)
        else:
            next_point = optimizer.suggest()

        args = brain.next_point_to_controller_values(next_point, actuator_indices)
        controller = Controller(args)
        sensors = Sensors(body.grid)

        result = world.run_simulator(sim, controller, sensors, viewer, config.SIMULATION_LENGTH, True)

        objective_values.append(result)

        optimizer.register(params=next_point, target=result)
    sim.reset()
    viewer.close()
    return objective_values

if __name__ == '__main__':
    main()