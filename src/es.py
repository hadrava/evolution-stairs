#!/usr/bin/env

import numpy as np
import tensorflow as tf
import os.path, gym
import roboschool


class HumanoidPolicy(object):
    def __init__(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input((44,)),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(17, activation=None),
        ])
        self.model = model

    def create_new(self):
        return HumanoidPolicy()

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def act(self, obs_data):
        input_data = np.expand_dims(np.array(obs_data), axis = 0)
        action = self.model(input_data, training = False)
        return action[0].numpy()

    def run_policy_in_env(self, env, run_seed):
        env.seed(run_seed)
        obs = env.reset()

        score = 0
        frame = 0
        done = False
        while not done:
            action = self.act(obs)

            obs, reward, done, _ = env.step(action)

            score += reward
            frame += 1
        return score


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

def one_mutation(policy):
    if np.random.rand() < MUTATION_PROB:
        new_weights = []
        for weights in policy.get_weights():
            if np.random.rand() < MUTATION_LAYER_PROB:
                mutate_weight = np.random.rand(*weights.shape) < MUTATION_WEIGHT_PROB
                new_weights.append(weights + np.random.normal(0, MUTATION_SIGMA, weights.shape) * mutate_weight )
            else:
                new_weights.append(weights)
        policy = HumanoidPolicy()
        policy.set_weights(new_weights)
    return policy

def mutation(population):
    for i, policy in enumerate(population[1:]):
        population[i+1] = one_mutation(policy)
    return population

import crossovers
one_crossover = crossovers.move_per_layer

def crossover(population):
    for i in range(1, (POPULATION_SIZE+1) // 2):
        k = 2*i
        first , second = one_crossover(population[k-1], population[k])
        population[k-1], population[k] = first, second
    return population

np.random.seed(42)
tf.random.set_seed(42)
env = gym.make("RoboschoolHumanoid-v1")
env_render = gym.make("RoboschoolHumanoid-v1")

POPULATION_SIZE = 11
EPOCH_COUNT = 100
MUTATION_WEIGHT_PROB = 0.01
MUTATION_PROB = 0.3
MUTATION_LAYER_PROB = 0.3
MUTATION_SIGMA = 0.01
TOUR_SEL_WINNER = 0.9
population = [HumanoidPolicy() for _ in range(POPULATION_SIZE)]

#import pdb
#pdb.set_trace()
import default_weights

for policy in population:
    new_weights = []
    for random, pretrain in zip(policy.get_weights(), default_weights.default_weights):
        new_weights.append(random *0.3*0 + pretrain)
    policy.set_weights(new_weights)

import random
import sys
for epoch in range(EPOCH_COUNT):
    rewards = []
    max_reward = -1e10
    run_seed = random.randrange(sys.maxsize)

    index = 0
    for i, policy in enumerate(population):
        rewards.append(policy.run_policy_in_env(env_render if i == 0 and epoch % 1 == 0 else env, run_seed=run_seed))
        if rewards[-1] > max_reward:
            max_reward, index = rewards[-1], i
    #env_render.render("human")

    population = tournament_sel(population, rewards, index)
    population = mutation(population)
    population = crossover(population)

    print("Epoch = %4i:" % epoch, max_reward)



