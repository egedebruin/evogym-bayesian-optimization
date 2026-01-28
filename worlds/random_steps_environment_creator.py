from worlds.random_environment_creator import heights_to_contents

def make(platform_length):
    length = 100
    heights = [1, 1, 1, 1, 1]
    height = 1
    i = 0
    while i < length - 5:
        for j in range(platform_length):
            heights.append(height)
            i += 1
        height += 1
    heights = heights[:length]
    contents = heights_to_contents(heights)
    return contents, heights