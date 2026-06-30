# EvoGym with Bayesian optimization, Reinforcement learning, and Lamarckian inheritance

A co-evolutionary robotics framework that jointly evolves robot morphologies and learns controllers using Bayesian Optimization and Reinforcement Learning, built on top of [EvoGym](https://github.com/EvolutionGym/evogym).

## Overview

This framework evolves both the **body** (morphology) and **brain** (controller) of modular soft robots:

- **Morphology Evolution**: Robots are represented as grids of voxels (empty, rigid, soft, horizontal/vertical actuators). Bodies evolve across generations via mutation operators.
- **Controller Learning**: Each robot's controller is trained via Bayesian Optimization (BO) with Gaussian Processes, PPO, or DDPG.
- **Experience Inheritance**: Offspring can inherit learned controller parameters from their parents, warm-starting the search and reducing learning time.
- **MAP-Elites**: Optional quality-diversity optimization to maintain a diverse archive of high-performing solutions.

## Installation

```bash
pip install -r requirements.txt
```

> EvoGym may require additional system dependencies. See the [EvoGym installation guide](https://github.com/EvolutionGym/evogym#installation) for details.

## Running an Experiment

The main entry point is `main.py`:

```bash
python main.py
```

By default, the configuration is read from `configs/config.py`. To pass arguments via command line (requires `READ_ARGS = True` in config):

```bash
python main.py \
  --learn 5 \
  --inherit-samples -1 \
  --repetition 1 \
  --environment rugged \
  --inherit-type best \
  --social-pool 3 \
  --learn-method bo
```

### Key Arguments

| Argument | Description | Example |
|---|---|---|
| `--environment` | Simulation environment | `simple`, `rugged`, `bidirectional` |
| `--learn` | Number of BO/RL iterations per individual | `5` |
| `--learn-method` | Learning method | `bo`, `ppo`, `ddpg` |
| `--inherit-samples` | Controller samples inherited from parent (`-1` = all) | `-1` |
| `--inherit-type` | Inheritance strategy | `parent`, `best`, `random`, `similar`, `cell` |
| `--social-pool` | Number of individuals to draw experience from | `3` |
| `--repetition` | Run index (for repeated experiments) | `1` |

## Configuration

All parameters are defined in `configs/config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `POP_SIZE` | `10` | Population size |
| `OFFSPRING_SIZE` | `10` | Offspring per generation |
| `FUNCTION_EVALUATIONS` | `500000` | Total simulation steps budget |
| `LEARN_ITERATIONS` | `5` | BO/RL steps per individual |
| `PARALLEL_PROCESSES` | `10` | Parallel worker processes |
| `CONTROLLER_TYPE` | `nn` | Controller type (`nn` or `sine`) |
| `MAP_ELITES` | `False` | Enable MAP-Elites archive |
| `GRID_LENGTH` | `5` | Robot body grid size |
| `SIMULATION_LENGTH` | `500` | Steps per simulation rollout |

## Project Structure

```
.
├── main.py                        # Evolution loop entry point
├── learn.py                       # Controller learning (BO + RL)
├── individual.py                  # Robot individual representation
├── selection.py                   # Selection operators
├── custom_bayesian_optimization.py
├── monkey_patch.py                # Scipy L-BFGS-B boundary fix
├── configs/
│   └── config.py                  # All experiment parameters
├── robot/
│   ├── body.py                    # Morphology representation and mutation
│   ├── brain_nn.py                # Neural network controller
│   ├── brain_sine.py              # Sinusoidal controller
│   ├── controller_nn.py           # NN controller execution
│   └── sensors.py                 # Sensory inputs
├── reinforcement_learning/
│   ├── ppo.py                     # PPO agent
│   └── ddpg.py                    # DDPG agent
├── util/
│   ├── archive.py                 # MAP-Elites archive
│   ├── world.py                   # EvoGym environment wrapper
│   ├── writer.py                  # Results file I/O
│   └── start.py                   # CLI argument parsing
├── analysis/                      # Post-hoc analysis and plotting scripts
├── worlds/                        # Environment JSON definitions
├── optimized_robots/              # Pre-optimized robot morphologies
└── results/                       # Experiment outputs (generated at runtime)
```

## Output

Results are saved to `results/<folder>/`:

| File | Contents |
|---|---|
| `populations.txt` | Individual IDs per generation |
| `individuals.txt` | Full individual data (body, controller, fitness) |
| `experience.pkl` | Learned controller parameters |
| `rng_state.npy` | RNG state for reproducibility |

## Resuming a Run

If a run is interrupted, it will automatically resume from the last saved generation when restarted with the same output folder configuration.

## Analysis

The `analysis/` directory contains scripts for visualizing results, plotting fitness curves, comparing morphologies, and more. Run individual scripts after an experiment completes:

```bash
python analysis/baseline_data.py
python analysis/run_best.py
```
