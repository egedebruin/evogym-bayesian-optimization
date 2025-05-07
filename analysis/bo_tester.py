import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern
from matplotlib.cm import get_cmap

# Define three target functions with different maxima
def peak_middle(x):
    return np.exp(-40 * (x - 0.5)**2) + 0.1 * np.cos(15 * x)

def peak_left(x):
    return np.exp(-5 * x) + 0.1 * np.sin(10 * x)

def peak_right(x):
    return np.exp(-5 * (1 - x)) + 0.1 * np.sin(10 * x)

# Random Search function (with repetition)
def random_search(f, bounds, n_iter, reps):
    all_runs = []
    for _ in range(reps):
        xs = np.random.uniform(bounds[0], bounds[1], n_iter)
        ys = [f(x) for x in xs]
        all_runs.append(np.maximum.accumulate(ys))
    return np.mean(all_runs, axis=0), np.std(all_runs, axis=0)

# Bayesian Optimization function (with repetition)
def run_bo(f, kappa, n_iter, reps):
    all_runs = []
    for _ in range(reps):
        optimizer = BayesianOptimization(
            f=f,
            pbounds={'x': (0, 1)},
            allow_duplicate_points=True,
            acquisition_function=acquisition.UpperConfidenceBound(kappa=kappa),
            verbose=0
        )
        optimizer.set_gp_params(
            kernel=Matern(nu=2.5, length_scale=0.2, length_scale_bounds="fixed"),
            alpha=1e-10
        )
        best_so_far = []
        for _ in range(n_iter):
            next_point = optimizer.suggest()
            result = f(next_point['x'])
            optimizer.register(params=next_point, target=result)
            if not best_so_far:
                best_so_far.append(result)
            else:
                best_so_far.append(max(best_so_far[-1], result))
        all_runs.append(best_so_far)
    return np.mean(all_runs, axis=0), np.std(all_runs, axis=0)

# Parameters
n_iter = 30
n_reps = 20
kappas = [1, 5, 20]
functions = {
    "Peak at 0": peak_left,
    "Peak at 0.5": peak_middle,
    "Peak at 1": peak_right
}

# Collect traces for all functions
results = {}
for name, func in functions.items():
    random_mean, random_std = random_search(func, (0, 1), n_iter, n_reps)
    bo_means = {}
    bo_stds = {}
    for kappa in kappas:
        mean, std = run_bo(func, kappa, n_iter, n_reps)
        bo_means[kappa] = mean
        bo_stds[kappa] = std
    results[name] = (random_mean, random_std, bo_means, bo_stds)

# Plotting
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharey=True)
cmap = get_cmap("viridis")

for ax, (name, (random_mean, random_std, bo_means, bo_stds)) in zip(axs, results.items()):
    # Plot random search with variance
    ax.plot(random_mean, '--', color='black', label='Random Search')
    ax.fill_between(range(n_iter), random_mean - random_std, random_mean + random_std, color='black', alpha=0.2)

    # Plot BO results with variance
    for i, kappa in enumerate(kappas):
        mean = bo_means[kappa]
        std = bo_stds[kappa]
        color = cmap(i / len(kappas))
        ax.plot(mean, label=f"BO (kappa={kappa})", color=color)
        ax.fill_between(range(n_iter), mean - std, mean + std, color=color, alpha=0.2)

    ax.set_title(name)
    ax.set_xlabel("Iteration")
    ax.grid(True)

axs[0].set_ylabel("Best Value So Far")
axs[1].legend()
plt.suptitle(f"Random Search vs BO (Mean ± Std over {n_reps} Runs)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
