from worlds.random_environment_creator import heights_to_contents, object_heights_to_contents


def make(obstacle_height, ceiling_height):
    length = 100
    heights = []
    for i in range(length):
        if i % 10 == 9:
            heights.append(obstacle_height + 1)
        else:
            heights.append(1)
    ground = heights_to_contents(heights)

    ceiling_heights = []
    ceiling_types = []
    for i in range(length):
        ceiling_heights.append(ceiling_height)
        if i % 10 == 0 or i == length - 1:
            ceiling_types.append(5)
        else:
            ceiling_types.append(2)
    ceiling = object_heights_to_contents(ceiling_heights, ceiling_types)

    contents = {
        "grid_width": 100,
        "grid_height": 15,
        "objects": {
            'ground': ground,
            'ceiling': ceiling,
        }
    }

    return contents