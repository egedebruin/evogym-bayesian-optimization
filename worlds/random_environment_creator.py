def make(rng):
    indices = []
    types = []
    neighbours = {}
    height = 0
    for i in range(100):
        if i < 6:
            height = 1
        else:
            choices = [0]
            if height > 1:
                choices.append(-1)
            if height < 10:
                choices.append(1)
            height += rng.choice(choices)

        for j in range(height):
            current = i + j * 100
            down = current - 100
            left = current - 1

            indices.append(current)
            types.append(5)

            current_neighbours = []
            if down in indices:
                current_neighbours.append(down)
                neighbours[str(down)].append(current)
            if left in indices:
                current_neighbours.append(left)
                neighbours[str(left)].append(current)

            neighbours[str(current)] = current_neighbours

    contents = {
        "grid_width": 100,
        "grid_height": 15,
        "objects": {
            "ground": {
                "indices": indices,
                "types": types,
                "neighbors": neighbours,
            }
        }
    }
    return contents