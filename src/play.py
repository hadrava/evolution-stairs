import numpy as np
import tensorflow as tf
import os.path, gym
import roboschool

import mutations
import crossovers
import policies

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re
    import random
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir',  type=str)
    parser.add_argument('-e',  '--epoch', type=int, default=10)

    args = parser.parse_args()

    REPEAT = 5
    POPULATION_SIZE = 1
    env = gym.make("RoboschoolWalker2d-v1")
    population = [policies.WalkerPolicy() for _ in range(POPULATION_SIZE)]
    out_weights = "walker_best_weights"

    for policy in population:
        out_epoch_weights = out_weights + "e-{}".format(args.epoch)
        if os.path.isdir(args.dir): out_epoch_weights = os.path.join(args.dir, out_epoch_weights)
        policy.load_weights(out_epoch_weights)

    run_seed = random.randrange(sys.maxsize)
    for i, policy in enumerate(population):
      policy.run_policy_in_env(env, run_seed=run_seed, render=True)
