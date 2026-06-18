import plot

# Constants
EVALS_PER_GEN = 50
REPETITIONS = 20

LABELS = {
    (-1, 'none', 0): 'Darwinian',
    (8, 'best', 1): 'Best - N=1',
    (8, 'best', 8): 'Best - N=8',
    (1, 'parent', 1): 'Lamarckian',
    (8, 'parent', 1): 'Lamarckian',
    (8, 'random', 1): 'Random - N=1',
    (8, 'random', 8): 'Random - N=8',
    (8, 'similar', 1): 'Similar - N=1',
    (8, 'similar', 8): 'Similar - N=8',
}


def collect_data(inherit, inherit_type, inherit_pool, environment, learn_method):
    extra = ""
    if environment == 'changing':
        extra += f"_changing-1.0"
    extra += f"_vision-2"

    GENERATIONS = 60
    key = (inherit, inherit_type, inherit_pool)
    label = LABELS.get(key)

    if label is None:
        return None

    values = []
    for repetition in range(1, REPETITIONS + 1):
        data_path = f'../results/learn-{EVALS_PER_GEN}_inherit-{inherit}_type-{inherit_type}_pool-{inherit_pool}_environment-{environment}_method-{learn_method}{extra}_repetition-{repetition}'
        data_array = plot.get_data(data_path, GENERATIONS)

        values.extend(data_array[-1])

    return values


def main():
    environments = ['simple', 'bidirectional2', 'changing', 'bidirectional']

    for env in environments:
        for learn_method in ['ddpg']:
            d_values = collect_data(-1, 'none', 0, env, learn_method)
            l_values = collect_data(1, 'parent', 1, env, learn_method)

            # Create a separate file per cat
            output_filename = f"{env}.txt"
            with open(output_filename, 'w') as f:
                # Header: cat, x, y(tab separated)
                f.write("Darwinian\tLamarckian\n")
                for darwinian, lamarckian in zip(d_values, l_values):
                    f.write(f"{darwinian}\t{lamarckian}\n")
            print(f"Successfully created {output_filename}")


if __name__ == '__main__':
    main()
