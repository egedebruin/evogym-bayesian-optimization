import numpy as np
import plot

# Constants
POP_SIZE = 100
EVALS_PER_GEN = 50
REPETITIONS = 20
SUB_FOLDER = ''

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
    def downsample_mean(x, w):
        x = np.asarray(x)
        return np.array([
            np.mean(x[i:i + w]) for i in range(0, len(x), w)
        ])

    extra = ""
    if environment == 'changing':
        extra += f"_changing-1.0"
    extra += f"_vision-2"

    GENERATIONS = 60
    key = (inherit, inherit_type, inherit_pool)
    label = LABELS.get(key)

    if label is None:
        return None

    curves = []
    for repetition in range(1, REPETITIONS + 1):
        data_path = f'results/learn-{EVALS_PER_GEN}_inherit-{inherit}_type-{inherit_type}_pool-{inherit_pool}_environment-{environment}_method-{learn_method}{extra}_repetition-{repetition}'
        data_array = plot.get_data(data_path, GENERATIONS)

        if data_array is None:
            continue
        if len(data_array) < GENERATIONS:
            continue

        mean_vals = np.nanmean(data_array, axis=1)
        curves.append(mean_vals)

    if not curves:
        return None

    curves = np.array(curves)

    median_vals = np.median(curves, axis=0)
    q25 = np.percentile(curves, 25, axis=0)
    q75 = np.percentile(curves, 75, axis=0)
    x_vals = np.arange(1, GENERATIONS + 1)
    
    window = 1
    median_vals = downsample_mean(median_vals, window)
    q25 = downsample_mean(q25, window)
    q75 = downsample_mean(q75, window)
    x_vals = downsample_mean(x_vals, window)
    
    return label, x_vals, median_vals, q25, q75

def main():
    environments = ['simple', 'changing']
    strategy_keys = list(LABELS.keys())
    
    output_path = 'analysis/baseline_data.txt'
    
    with open(output_path, 'w') as f:
        # Header: cat, x, y, ymin, ymax (tab separated)
        f.write("cat\tx\ty\tymin\tymax\n")
        
        for env in environments:
            # Environment display name logic from baseline_plot.py
            env_name = 'Changing-0' if env == 'changing-1e-07' else (
                'Bidirectional with direction sensor' if env == 'bidirectional2' else (
                'Bidirectional without direction sensor' if env == 'bidirectional' else env
                )
            )
            
            for key in strategy_keys:
                for learn_method in ['ddpg']:
                    result = collect_data(*key, env, learn_method)
                    if result:
                        label, x_vals, median_vals, q25, q75 = result
                        cat = f"{env_name} - {label}"
                        for x, y, ymin, ymax in zip(x_vals, median_vals, q25, q75):
                            f.write(f"{cat}\t{x}\t{y}\t{ymin}\t{ymax}\n")
                            
    print(f"Successfully created {output_path}")

if __name__ == '__main__':
    main()
