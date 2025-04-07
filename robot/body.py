import numpy as np

import config


class Body:
    grid: np.array

    def __init__(self, max_size=-1, initial_size=-1, rng=None):
        if max_size < 0 and initial_size < 0:
            self.grid = np.array([])
            return

        self.random_grid(max_size, initial_size, rng)

    def replace_grid(self, new_grid):
        self.grid = new_grid

    def random_grid(self, max_size, initial_size, rng):
        self.grid = np.full((max_size, max_size), 0.0)

        self.grid[rng.integers(0, max_size)][rng.integers(0, max_size)] = float(rng.integers(3, 5))
        for i in range(initial_size - 1):
            self.add_mutation(rng)

    def grid_size(self):
        grid_size = 0
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] > 0.0:
                    grid_size += 1
        return grid_size

    def mutate(self, rng: np.random.Generator):
        success = False
        while not success:
            choice = rng.random()
            if choice < 1/3:
                number_of_additions = rng.integers(1, config.MAX_ADD_MUTATION + 1)
                if self.grid_size() + number_of_additions <= config.MAX_SIZE:
                    for _ in range(number_of_additions):
                        self.add_mutation(rng)
                    success = True
            elif choice < 2/3:
                for _ in range(rng.integers(1, config.MAX_CHANGE_MUTATION + 1)):
                    self.change_mutation(rng)
                success = True
            else:
                number_of_deletions = rng.integers(1, config.MAX_DELETE_MUTATION + 1)
                if self.grid_size() - number_of_deletions >= config.MIN_SIZE:
                    for _ in range(number_of_deletions):
                        self.delete_mutation(rng)
                    success = True

    def add_mutation(self, rng):
        max_size = self.grid.shape[0]
        success = False
        while not success:
            x = rng.integers(0, max_size)
            y = rng.integers(0, max_size)
            if self.grid[x][y] == 0.0:
                continue

            possible_connections = self.get_connections(x, y, max_size)
            if len(possible_connections) == 0:
                continue

            (connection_x, connection_y) = rng.choice(possible_connections)
            self.grid[connection_x, connection_y] = float(rng.integers(1, 5))
            success = True

    def delete_mutation(self, rng):
        max_size = self.grid.shape[0]
        success = False
        while not success:
            new_grid = np.copy(self.grid)
            x = rng.integers(0, max_size)
            y = rng.integers(0, max_size)
            if new_grid[x][y] == 0.0:
                continue

            new_grid[x][y] = 0.0
            if not Body.grid_is_ok(new_grid, max_size):
                continue

            self.grid = new_grid
            success = True

    def change_mutation(self, rng):
        max_size = self.grid.shape[0]
        success = False
        while not success:
            new_grid = np.copy(self.grid)
            x = rng.integers(0, max_size)
            y = rng.integers(0, max_size)
            if new_grid[x][y] == 0.0:
                continue

            old = new_grid[x][y]
            new = new_grid[x][y]
            while old == new:
                new = float(rng.integers(1, 5))
            new_grid[x][y] = new
            if not Body.grid_is_ok(new_grid, max_size):
                continue

            self.grid = new_grid
            success = True

    def get_connections(self, x, y, max_size):
        connections = []
        for i in range(-1, 2):
            if (x + i < 0) or (x + i >= max_size):
                continue
            for j in range(-1, 2):
                if (y + j < 0) or (y + j >= max_size):
                    continue
                if i != 0 and j != 0:
                    continue
                if self.grid[x + i][y + j] != 0.0:
                    continue
                connections.append((x + i, y + j))
        return connections

    @staticmethod
    def grid_is_ok(grid, max_size):
        contains_action = True
        for i in range(max_size):
            for j in range(max_size):
                if grid[i][j] == 3.0 or grid[i][j] == 4.0:
                    contains_action = True
        return Body.is_fully_connected(grid) and contains_action

    @staticmethod
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