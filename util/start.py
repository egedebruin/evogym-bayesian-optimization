import argparse
from datetime import datetime

import numpy as np

import config
from util.logger_setup import logger


def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--learn', help='Number learn generations.', required=True, type=int)
	parser.add_argument('--inherit-samples', help='Number of samples to inherit.', required=True, type=int)
	parser.add_argument('--repetition', help='Experiment number.', required=True, type=int)

	args = parser.parse_args()
	config.LEARN_ITERATIONS = args.learn
	config.INHERIT_SAMPLES = args.inherit_samples
	config.FOLDER = f"results/learn-{args.learn}_inherit-{args.inherit_samples}_repetition-{args.repetition}/"

def make_rng_seed():
	seed = int(datetime.now().timestamp() * 1e6) % 2**32
	logger.info(f"Random Seed: {seed}")
	return np.random.Generator(np.random.PCG64(seed))