from copy import deepcopy

import numpy as np

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
        self.add_evaluation(objective_value, best_brain, experience)
        self.parent_id = parent_id

    def add_evaluation(self, objective_value, best_brain, experience):
        self.objective_value = objective_value
        self.best_brain = best_brain
        self.experience = experience

    def to_file_string(self):
        return f"{self.id};{self.body.grid.tolist()};{self.brain.to_string()};{self.best_brain};{self.parent_id};{self.objective_value};{self.original_generation}"

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

        min_dist = np.inf
        A_non_zero = np.count_nonzero(A)
        B_non_zero = np.count_nonzero(B)
        for dx_a in range(-config.GRID_LENGTH + 1, config.GRID_LENGTH):
            for dy_a in range(-config.GRID_LENGTH + 1, config.GRID_LENGTH):
                for dx_b in range(-config.GRID_LENGTH + 1, config.GRID_LENGTH):
                    for dy_b in range(-config.GRID_LENGTH + 1, config.GRID_LENGTH):
                        A_shift_non_zero = 0
                        B_shift_non_zero = 0
                        dist = 0
                        for i in range(A.shape[0]):
                            for j in range(A.shape[1]):
                                x_a, y_a = i + dx_a, j + dy_a
                                x_b, y_b = i + dx_b, j + dy_b
                                a_val = A[
                                    x_a, y_a] if 0 <= x_a < config.GRID_LENGTH and 0 <= y_a < config.GRID_LENGTH else 0
                                b_val = B[
                                    x_b, y_b] if 0 <= x_b < config.GRID_LENGTH and 0 <= y_b < config.GRID_LENGTH else 0

                                if a_val != b_val:
                                    dist += 1
                                A_shift_non_zero += a_val != 0.0
                                B_shift_non_zero += b_val != 0.0
                        if A_non_zero == A_shift_non_zero and B_non_zero == B_shift_non_zero:
                            min_dist = min(min_dist, dist)

        return min_dist
