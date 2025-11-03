from copy import deepcopy

import numpy as np
from scipy.ndimage import shift

from configs import config
from robot.body import Body
from robot.active import Brain


class Individual:
    id: str
    body: Body
    brain: Brain
    experience: list[tuple]
    inherited_experience: list[tuple]
    best_brain: dict
    best_inherited_objective_value: float
    objective_value: float
    original_generation: int
    parent_id: str = "-1"

    def __init__(self, individual_id, body, brain, original_generation, inherited_experience=None):
        self.id = individual_id
        self.body = body
        self.brain = brain
        self.original_generation = original_generation
        self.inherited_experience = inherited_experience

    def add_restart_values(self, objective_value, best_brain, experience, parent_id):
        self.add_evaluation(objective_value, best_brain, experience, -1)
        self.parent_id = parent_id

    def add_evaluation(self, objective_value, best_brain, experience, best_inherited_objective_value):
        self.objective_value = objective_value
        self.best_brain = best_brain
        self.experience = experience
        self.best_inherited_objective_value = best_inherited_objective_value

    def to_file_string(self):
        return f"{self.id};{self.body.grid.tolist()};{self.brain.to_string()};{self.best_brain};{self.parent_id};{self.objective_value};{self.original_generation};{self.best_inherited_objective_value}"

    def to_experience_string(self):
        return f"{self.id};{self.experience}"

    def mutate(self, rng):
        self.body.mutate(rng)
        self.brain.mutate(rng)

    def generate_new_individual(self, generation_index, offspring_id, rng):
        new_individual = deepcopy(self)
        new_individual.mutate(rng)
        new_individual.parent_id = self.id
        new_individual.id = f"{generation_index}-{offspring_id}"
        new_individual.original_generation = generation_index
        return new_individual

    def inherit_experience(self, population, parent, rng):
        self.inherited_experience = []
        if config.INHERIT_TYPE == 'parent':
            selected_individuals = [parent]
        elif config.INHERIT_TYPE == 'best':
            selected_individuals = sorted(population, key=lambda ind: -ind.objective_value)[:config.SOCIAL_POOL]
        elif config.INHERIT_TYPE == 'random':
            selected_individuals = rng.choice(population, size=config.SOCIAL_POOL, replace=False).tolist()
        elif config.INHERIT_TYPE == 'similar':
            selected_individuals = sorted(population, key=lambda ind: Individual.hamming_distance(self.body.grid, ind.body.grid))[:config.SOCIAL_POOL]
        elif config.INHERIT_TYPE == 'none':
            return
        else:
            raise ValueError(f"Unknown INHERIT TYPE: {config.INHERIT_TYPE}")

        pre_sorted_experiences = [
            sorted(ind.experience, key=lambda x: x[1], reverse=True)
            for ind in selected_individuals
        ]
        for i in range(config.LEARN_ITERATIONS):
            for j in range(config.SOCIAL_POOL):
                if pre_sorted_experiences[j][i][0] not in [exp[0] for exp in self.inherited_experience]:
                    self.inherited_experience.append(pre_sorted_experiences[j][i])
                # TODO: Deal with same-samples-problem
                # TODO: For now we skip the similar samples and continue, so we do end up with INHERIT_SAMPLES samples to reevaluate

    @staticmethod
    def hamming_distance(A, B):
        A = np.array(A)
        B = np.array(B)
        gl = config.GRID_LENGTH

        A_non_zero = np.count_nonzero(A)
        B_non_zero = np.count_nonzero(B)

        min_dist = np.inf
        shifts = range(-gl + 1, gl)

        for dx_a in shifts:
            A_shifted = shift(A, shift=(dx_a, 0), order=0, cval=0)
            for dy_a in shifts:
                A_final = shift(A_shifted, shift=(0, dy_a), order=0, cval=0)

                A_nz = np.count_nonzero(A_final)
                if A_nz != A_non_zero:
                    continue

                for dx_b in shifts:
                    B_shifted = shift(B, shift=(dx_b, 0), order=0, cval=0)
                    for dy_b in shifts:
                        B_final = shift(B_shifted, shift=(0, dy_b), order=0, cval=0)

                        B_nz = np.count_nonzero(B_final)
                        if B_nz != B_non_zero:
                            continue

                        dist = np.count_nonzero(A_final != B_final)
                        min_dist = min(min_dist, dist)

        return min_dist
