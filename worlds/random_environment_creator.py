from configs import config

def make(rng):
    heights = []
    height = 0
    previous_choice = 0
    for i in range(100):
        if i < 11:
            height = 1
        else:
            choices = get_choices(height, previous_choice)

            previous_choice = rng.choice(choices)
            height += previous_choice

        heights.append(height)

    contents = heights_to_contents(heights)
    return contents, heights

def change(rng, last_heights):
    heights = []
    height = 0
    previous_choice = 0
    for i, last_height in enumerate(last_heights):
        if i < 11:
            height = 1
        else:
            choices = get_choices(height, previous_choice)

            last_choice = last_height - last_heights[i-1]
            if last_choice in choices and rng.random() < (1 - config.CHANGE_PROB):
                previous_choice = last_choice
            else:
                previous_choice = rng.choice(choices)

            height += previous_choice

        heights.append(height)

    contents = heights_to_contents(heights)
    return contents, heights

def get_choices(height, previous_choice):
    choices = [0]
    if height > 1 and previous_choice == 0:
        choices.append(-1)
    if height < 10 and previous_choice == 0:
        choices.append(1)
    return choices

def heights_to_contents(heights):
    indices = []
    types = []
    neighbours = {}

    for i, height in enumerate(heights):
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

    return {
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