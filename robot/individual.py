from copy import deepcopy

from robot.body import Body
from robot.brain import Brain


class Individual:
    id: str
    body: Body
    brain: Brain
    experience: list[tuple]
    inherited_experience: list[tuple]
    objective_value: float
    original_generation: int
    parent_id: str = "-1"

    def __init__(self, individual_id, body, brain, original_generation, inherited_experience=None):
        self.id = individual_id
        self.body = body
        self.brain = brain
        self.original_generation = original_generation
        self.inherited_experience = inherited_experience

    def add_restart_values(self, objective_value, experience, parent_id):
        self.add_evaluation(objective_value, experience)
        self.parent_id = parent_id

    def add_evaluation(self, objective_value, experience):
        self.objective_value = objective_value
        self.experience = experience

    def to_file_string(self):
        return f"{self.id};{self.body.grid.tolist()};{self.brain.grid.tolist()};{self.experience};{self.parent_id};{self.objective_value};{self.original_generation}"

    def mutate(self, rng):
        self.body.mutate(rng)
        self.brain.mutate(rng)

    def generate_new_individual(self, generation_index, offspring_id, rng):
        new_individual = deepcopy(self)
        new_individual.mutate(rng)
        new_individual.inherited_experience = self.experience
        new_individual.parent_id = self.id
        new_individual.id = f"{generation_index}-{offspring_id}"
        new_individual.original_generation = generation_index
        return new_individual