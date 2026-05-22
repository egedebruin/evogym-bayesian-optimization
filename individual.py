import math
from copy import deepcopy

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
        self.objective_value = objective_value
        self.best_brain = best_brain
        self.experience = experience
        self.best_inherited_objective_value = -1
        self.parent_id = parent_id

    def add_evaluation(self, experience, inherited_objective_values):
        best_experience = max(experience, key=lambda x: x[1])

        self.experience = experience
        self.best_brain = best_experience[0]
        self.objective_value = best_experience[1]
        self.best_inherited_objective_value = max(inherited_objective_values) if len(inherited_objective_values) > 0 else -math.inf

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

    def do_the_inherit(self, selected_individuals):
        pre_sorted_experiences = [
            sorted(ind.experience, key=lambda x: x[1], reverse=True)
            for ind in selected_individuals
        ]
        for i in range(config.INHERIT_SAMPLES):
            for j in range(config.SOCIAL_POOL):
                if pre_sorted_experiences[j][i][0] not in [exp[0] for exp in self.inherited_experience]:
                    self.inherited_experience.append(pre_sorted_experiences[j][i])

    def inherit_experience(self, population, parent, rng, archive=None):
        self.inherited_experience = []
        if config.INHERIT_TYPE == 'parent':
            selected_individuals = [parent]
        elif config.INHERIT_TYPE == 'best':
            selected_individuals = sorted(population, key=lambda ind: -ind.objective_value)[:config.SOCIAL_POOL]
        elif config.INHERIT_TYPE == 'random':
            selected_individuals = rng.choice(population, size=config.SOCIAL_POOL, replace=False).tolist()
        elif config.INHERIT_TYPE == 'similar':
            selected_individuals = sorted(population, key=lambda ind: Body.hamming_distance(self.body.grid, ind.body.grid))[:config.SOCIAL_POOL]
        elif config.INHERIT_TYPE == 'cell':
            if archive is None:
                raise ValueError("Invalid INHERIT TYPE: cell. Archive is none.")
            if config.SOCIAL_POOL != 1:
                raise ValueError("SOCIAL_POOL must be equal to 1")

            archive_similar = archive.get_from_cell(self)
            if archive_similar is None:
                archive_similar = parent
            selected_individuals = [archive_similar]
        elif config.INHERIT_TYPE == 'none':
            return
        else:
            raise ValueError(f"Unknown INHERIT TYPE: {config.INHERIT_TYPE}")

        self.do_the_inherit(selected_individuals)
