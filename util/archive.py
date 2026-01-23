from configs import config
from individual import Individual
import numpy as np


class Archive:

    def __init__(self, size: int):
        self.size = size
        self.archive = []

        for i in range(size):
            nested_archive = []
            for j in range(size):
                nested_archive.append(None)
            self.archive.append(nested_archive)

    def append_population(self, population: list[Individual]):
        for individual in population:
            self.append(individual)

    def append(self, individual: Individual):
        first_descriptor = Archive.descriptor(1, individual.body.grid)
        second_descriptor = Archive.descriptor(2, individual.body.grid)

        first_location = self.size - 1 if first_descriptor == 1 else int(first_descriptor * self.size)
        second_location = self.size - 1 if second_descriptor == 1 else int(second_descriptor * self.size)

        if self.archive[first_location][second_location] is None or individual.objective_value >= self.archive[first_location][second_location].objective_value:
            self.archive[first_location][second_location] = individual

    def receive(self, rng: np.random.Generator):
        result = None
        while result is None:
            result = self.archive[rng.integers(self.size)][rng.integers(self.size)]
        return result

    def get_all_individuals(self):
        all_individuals = []
        for row in self.archive:
            for individual in row:
                if individual is not None:
                    all_individuals.append(individual)
        return all_individuals

    def get_from_cell(self, individual: Individual):
        first_descriptor = Archive.descriptor(1, individual.body.grid)
        second_descriptor = Archive.descriptor(2, individual.body.grid)

        first_location = int(first_descriptor * self.size) - 1
        second_location = int(second_descriptor * self.size) - 1

        return self.archive[first_location][second_location]

    def get_most_similar(self, individual: Individual, amount: int):
        all_individuals = self.get_all_individuals()
        return sorted(all_individuals, key=lambda ind: Individual.hamming_distance(individual.body.grid, ind.body.grid))[:amount]

    def get_best(self, amount: int):
        all_individuals = self.get_all_individuals()
        return sorted(all_individuals, key=lambda ind: -ind.objective_value)[:amount]

    def get_random(self, amount: int, rng):
        all_individuals = self.get_all_individuals()
        return rng.choice(all_individuals, size=amount, replace=False).tolist()

    @staticmethod
    def descriptor(number: int, body: np.ndarray):
        descriptor = config.DESCRIPTORS[number - 1]
        if descriptor == 'relative_activity':
            return Archive.relative_activity(body)
        elif descriptor == 'elongation':
            return Archive.elongation(body)
        elif descriptor == 'compactness':
            return Archive.compactness(body)
        else:
            raise ValueError(f"Unknown descriptor: {descriptor}")

    @staticmethod
    def relative_activity(body: np.ndarray):
        return np.count_nonzero(body > 2) / np.count_nonzero(body > 0)

    @staticmethod
    def compactness(body: np.ndarray) -> float:
        convex_hull = body > 0
        if True not in convex_hull:
            return 0.0
        new_found = True
        while new_found:
            new_found = False
            false_coordinates = np.argwhere(convex_hull == False)
            for coordinate in false_coordinates:
                x, y = coordinate[0], coordinate[1]
                adjacent_count = 0
                adjacent_coordinates = []
                for d in [-1, 1]:
                    adjacent_coordinates.append((x, y + d))
                    adjacent_coordinates.append((x + d, y))
                    adjacent_coordinates.append((x + d, y + d))
                    adjacent_coordinates.append((x + d, y - d))
                for adj_x, adj_y in adjacent_coordinates:
                    if 0 <= adj_x < body.shape[0] and 0 <= adj_y < body.shape[1] and convex_hull[adj_x][adj_y]:
                        adjacent_count += 1
                if adjacent_count >= 5:
                    convex_hull[x][y] = True
                    new_found = True

        return (body > 0).sum() / convex_hull.sum()

    @staticmethod
    def elongation(body: np.ndarray) -> float:
        """
        Computes elongation of a 2D voxel robot using
        the ellipse of equal second moments.

        Returns a value in [0, 1).
        """
        body = body.astype(bool)
        coords = np.argwhere(body)

        # Fewer than 2 voxels → no elongation
        if len(coords) < 2:
            return 0.0

        # Center the coordinates
        coords = coords.astype(float)
        centroid = coords.mean(axis=0)
        centered = coords - centroid

        # Covariance matrix (2x2)
        cov = np.cov(centered, rowvar=False)

        # Eigenvalues (sorted)
        eigvals = np.linalg.eigvalsh(cov)
        lambda_min, lambda_max = eigvals

        if lambda_max <= 0:
            return 0.0

        # Semi-axis ratio
        b_over_a = np.sqrt(lambda_min / lambda_max)

        # Elongation definition
        return np.sqrt(1.0 - b_over_a ** 2)
