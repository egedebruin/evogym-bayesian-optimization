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
	parser.add_argument('--inherit-type', help='Inheritance type', required=True, type=str)
	parser.add_argument('--social-pool', help='Pool of robots to inherit', required=True, type=int)
	parser.add_argument('--learn-method', help='Controller learn method', required=True, type=str)
	parser.add_argument('--inherit-alpha', help='Alpha for inherited parameters', required=False, type=float)
	parser.add_argument('--kappa', help='Kappa for UCB', required=False, type=float)
	parser.add_argument('--random-learn', help='Learn only with random controllers', required=False, type=int)
	parser.add_argument('--bo-restarts', help='Number of restarts for the BO sample finder', required=False, type=int)

	args = parser.parse_args()
	config.LEARN_ITERATIONS = args.learn
	config.INHERIT_SAMPLES = args.inherit_samples
	config.ENVIRONMENT = args.environment
	config.INHERIT_TYPE = args.inherit_type
	config.SOCIAL_POOL = args.social_pool
	config.LEARN_METHOD = args.learn_method

	extra = ''
	if args.inherit_alpha:
		config.LEARN_INHERITED_ALPHA = args.inherit_alpha
		extra += "_alpha-" + str(args.inherit_alpha)
	if args.kappa:
		config.LEARN_KAPPA = args.kappa
		extra += "_kappa-" + str(args.kappa)
	if args.random_learn:
		config.RANDOM_LEARNING = args.random_learn == 1
		extra += "_random-" + str(args.random_learn)
	if args.bo_restarts:
		config.BO_RESTARTS = args.bo_restarts
	else:
		config.BO_RESTARTS = 1
	config.FOLDER = f"results/learn-{args.learn}_inherit-{args.inherit_samples}_type-{args.inherit_type}_pool-{args.social_pool}_environment-{args.environment}_method-{args.learn_method}{extra}_repetition-{args.repetition}/"

def make_rng_seed():
	seed = int(datetime.now().timestamp() * 1e6) % 2**32
	logger.info(f"Random Seed: {seed}")
	return np.random.Generator(np.random.PCG64(seed))