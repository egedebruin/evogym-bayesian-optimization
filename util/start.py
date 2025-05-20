import argparse
from datetime import datetime

import numpy as np

from configs import config
from util.logger_setup import logger


def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--learn', help='Number learn generations.', required=True, type=int)
	parser.add_argument('--inherit-samples', help='Number of samples to inherit.', required=True, type=int)
	parser.add_argument('--repetition', help='Experiment number.', required=True, type=int)
	parser.add_argument('--environment', help='Environment', required=True, type=str)
	parser.add_argument('--inherit-alpha', help='Alpha for inherited parameters', required=False, type=float)
	parser.add_argument('--kappa', help='Kappa for UCB', required=False, type=float)

	args = parser.parse_args()
	config.LEARN_ITERATIONS = args.learn
	config.INHERIT_SAMPLES = args.inherit_samples
	config.ENVIRONMENT = args.environment

	extra = ''
	if args.inherit_alpha:
		config.LEARN_INHERITED_ALPHA = args.inherit_alpha
		extra += "_alpha-" + str(args.inherit_alpha)
	if args.kappa:
		config.LEARN_KAPPA = args.kappa
		extra += "_kappa-" + str(args.kappa)
	config.FOLDER = f"results/learn-{args.learn}_inherit-{args.inherit_samples}_environment-{args.environment}{extra}_repetition-{args.repetition}/"

def make_rng_seed():
	seed = int(datetime.now().timestamp() * 1e6) % 2**32
	logger.info(f"Random Seed: {seed}")
	return np.random.Generator(np.random.PCG64(seed))