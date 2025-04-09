def tournament(population, selection_size, pool_size, rng):
    selection = []
    for i in range(selection_size):
        pool = rng.choice(population, pool_size)
        selection.append(sorted(pool, key=lambda p: p.objective_value, reverse=True)[0])
    return selection

def simple(population, selection_size, mode):
    if mode == 'generational':
        key_func = lambda ind: (-ind.original_generation, -ind.objective_value)
    elif mode == 'elitist':
        key_func = lambda ind: -ind.objective_value
    else:
        raise ValueError(f"Unknown simple selection mode: {mode}")

    return sorted(population, key=key_func)[:selection_size]

def select(population, selection_size, mode, rng=None, pool_size=-1):
    if mode == 'generational' or mode == 'elitist':
        return simple(population, selection_size, mode)
    elif mode == 'tournament':
        return tournament(population, selection_size, pool_size, rng)
    else:
        raise ValueError(f"Unknown selection mode: {mode}")