import numpy as np
import tensorflow as tf
import os.path, gym
import roboschool

import mutations
import crossovers
import policies


def tournament_sel(population, rewards, index):
    new_population = [population[index]]
    for i in range(POPULATION_SIZE - 1):
        (x, y) = np.random.choice(POPULATION_SIZE, 2)
        if rewards[x] > rewards[y]:
            winner, loser = population[x], population[y]
        else:
            loser, winner = population[x], population[y]
        if np.random.rand() < TOUR_SEL_WINNER:
            new_population.append(winner)
        else:
            new_population.append(loser)
    return new_population


def mutation(population):
    for i, policy in enumerate(population[1:]):
        population[i+1] = one_mutation(policy)
    return population


def crossover(population):
    for i in range(1, (POPULATION_SIZE+1) // 2):
        k = 2*i
        first, second = one_crossover(population[k-1], population[k])
        population[k-1], population[k] = first, second
    return population


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re
    import random
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mutation',  type=str, choices=[i for i in mutations. MUTATIONS],  default="normal", help='Select mutation kind')
    parser.add_argument('-c', '--crossover', type=str, choices=[i for i in crossovers.CROSSOVERS], default="onepoint", help='Select crossover kind')

    parser.add_argument('-cp',  '--crossover_prob',         type=float, default=0.30)
    parser.add_argument('-clp', '--crossover_layer_prob',   type=float, default=0.30)
    parser.add_argument('-cwp', '--crossover_weights_prob', type=float, default=0.01)

    parser.add_argument('-mp',  '--mutation_prob',         type=float, default=0.30)
    parser.add_argument('-mlp', '--mutation_layer_prob',   type=float, default=0.30)
    parser.add_argument('-mwp', '--mutation_weights_prob', type=float, default=0.01)
    parser.add_argument('-ms',  '--mutation_sigma',        type=float, default=0.01)

    args = parser.parse_args()

    args.logdir = os.path.join("logs", "{}-{}".format(
            os.path.basename(__file__),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))
    import os
    try:
        os.makedirs(args.logdir)
        print("Directory ", args.logdir, " Created ")
    except FileExistsError:
        print("Directory ", args.logdir,  " already exists")

    one_mutation = mutations.MUTATIONS[args.mutation]
    one_crossover = crossovers.CROSSOVERS[args.crossover]

    crossovers.CROSSOVER_PROB = args.crossover_prob
    crossovers.CROSSOVER_LAYER_PROB = args.crossover_layer_prob
    crossovers.CROSSOVER_WEIGHT_PROB = args.crossover_weights_prob

    mutations.MUTATION_PROB = args.mutation_prob
    mutations.MUTATION_LAYER_PROB = args.mutation_layer_prob
    mutations.MUTATION_WEIGHT_PROB = args.mutation_weights_prob
    mutations.MUTATION_SIGMA = args.mutation_sigma

    np.random.seed(42)
    tf.random.set_seed(42)

    POPULATION_SIZE = 11
    EPOCH_COUNT = 1000
    TOUR_SEL_WINNER = 0.9
    REPEAT = 5

    env = gym.make("RoboschoolWalker2d-v1")
    population = [policies.WalkerPolicy() for _ in range(POPULATION_SIZE)]
    out_weights = "walker_best_weights"
    out_path = "walker_rewards.csv"

    from default_weights_2d import default_weights


    for policy in population:
        new_weights = []
        for geterated, pretrain in zip(policy.get_weights(), default_weights):
            if len(pretrain.shape) == 2:
                pretrain = np.pad(pretrain, ((0, geterated.shape[0] - pretrain.shape[0]), (0, geterated.shape[1] - pretrain.shape[1])), 'constant', constant_values=0)
            else:
                pretrain = np.pad(pretrain,
                                  (0, geterated.shape[0] - pretrain.shape[0]),
                                  'constant', constant_values=0)
            new_weights.append(geterated * 0.3 + pretrain)
        policy.set_weights(new_weights)

    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for epoch in range(EPOCH_COUNT):
            if epoch > 0:
                population = tournament_sel(population, rewards, index)
                population = crossover(population)
                population = mutation(population)
            rewards = [0 for _ in population]
            max_reward = -1e10

            index = 0
            for _ in range(REPEAT):
                run_seed = random.randrange(sys.maxsize)
                for i, policy in enumerate(population):
                    rewards[i] += policy.run_policy_in_env(env, run_seed=run_seed)
                    if rewards[i] > max_reward:
                        max_reward, index = rewards[i], i

            if epoch % 10 == 0:
                out_epoch_weights = out_weights + "e-{}".format(epoch)
                if os.path.isdir(args.logdir): out_weights = os.path.join(args.logdir, out_epoch_weights)
                population[index].save_weights(out_epoch_weights)

            print("Epoch = %4i:" % epoch, max_reward)
            print(";".join(str(r) for r in rewards), file=out_file)

    for i, policy in enumerate(population):
        out_policy_weights = out_weights + "p-{}".format(i)
        if os.path.isdir(args.logdir): out_weights = os.path.join(args.logdir, out_policy_weights)
        policy.save_weights(out_policy_weights)
