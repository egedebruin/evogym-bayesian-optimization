class Selection:
    _selection_size: int
    _mode: str
    _extra_arguments: dict

    def __init__(self, selection_size, mode, extra_arguments=None):
        if extra_arguments is None:
            extra_arguments = dict()
        self._selection_size = selection_size
        self._mode = mode
        self._extra_arguments = extra_arguments

    def _tournament(self, population, rng):
        if 'pool_size' not in self._extra_arguments:
            raise ValueError('Extra argument pool_size is required for tournament selection')
        selection = []
        for i in range(self._selection_size):
            pool = rng.choice(population, self._extra_arguments['pool_size'])
            selection.append(sorted(pool, key=lambda p: p.objective_value, reverse=True)[0])
        return selection

    def _simple(self, population):
        if self._mode == 'generational':
            key_func = lambda ind: (-ind.original_generation, -ind.objective_value)
        elif self._mode == 'elitist':
            key_func = lambda ind: -ind.objective_value
        else:
            raise ValueError(f"Unknown simple selection mode: {self._mode}")

        return sorted(population, key=key_func)[:self._selection_size]

    def select(self, population, rng):
        if self._mode in ['generational', 'elitist']:
            return self._simple(population)

        elif self._mode == 'random':
            return [population[rng.integers(len(population))]
                    for _ in range(self._selection_size)]

        elif self._mode == 'tournament':
            return self._tournament(population, rng)
        else:
            raise ValueError(f"Unknown selection mode: {self._mode}")