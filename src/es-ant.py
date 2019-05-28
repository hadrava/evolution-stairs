#!/usr/bin/env

import numpy as np
import tensorflow as tf
import os.path, gym
import roboschool


class HumanoidPolicy(object):
    def __init__(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input((28,)),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(8, activation=None),
        ])
        self.model = model

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def act(self, obs_data):
        input_data = np.expand_dims(np.array(obs_data), axis = 0)
        action = self.model(input_data, training = False)
        return action[0].numpy()

    def run_policy_in_env(self, env):
        obs = env.reset()

        score = 0
        frame = 0
        done = False
        while not done:
            action = self.act(obs)

            obs, reward, done, _ = env.step(action)
            env.render("human")

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

def one_croosover(first, second):
    if np.random.rand() < CROSSOVER_PROB:
        gamma = np.random.rand()
        new_first_weights = []
        new_second_weights = []
        for first_weights, second_weights in zip(first.get_weights(), second.get_weights()):
            if np.random.rand() < CROSSOVER_LAYER_PROB:
                mutate_weights = np.random.rand(*first_weights.shape) < CROSSOVER_WEIGHT_PROB
                new_first_weights.append(np.where(mutate_weights, (1-gamma)*first_weights + (gamma * second_weights), first_weights))
                new_second_weights.append(np.where(mutate_weights, gamma*first_weights + ((1-gamma) * second_weights), second_weights))
            else:
                new_first_weights.append(first_weights)
                new_second_weights.append(second_weights)
        first, second = HumanoidPolicy(), HumanoidPolicy()
        first.set_weights(new_first_weights)
        second.set_weights(new_second_weights)
    return first, second

def crossover(population):
    for i in range(1, (POPULATION_SIZE+1) // 2):
        k = 2*i
        first , second = one_croosover(population[k-1], population[k])
        population[k-1], population[k] = first, second
    return population

np.random.seed(42)
tf.random.set_seed(42)
env = gym.make("RoboschoolAnt-v1")
env_render = env
#env_render = gym.make("RoboschoolAnt-v1")

POPULATION_SIZE = 11
EPOCH_COUNT = 100
CROSSOVER_WEIGHT_PROB = 0.01
CROSSOVER_PROB = 0.3
CROSSOVER_LAYER_PROB = 0.3
MUTATION_WEIGHT_PROB = 0.01
MUTATION_PROB = 0.3
MUTATION_LAYER_PROB = 0.3
MUTATION_SIGMA = 0.01
TOUR_SEL_WINNER = 0.9
population = [HumanoidPolicy() for _ in range(POPULATION_SIZE)]

#import pdb
#pdb.set_trace()
import default_weights_ant

for policy in population:
    new_weights = []
    for random, pretrain in zip(policy.get_weights(), default_weights_ant.default_weights):
        new_weights.append(random *0.3 + pretrain)
    policy.set_weights(new_weights)

for epoch in range(EPOCH_COUNT):
    rewards = []
    max_reward = -1e10
    index = 0
    for i, policy in enumerate(population):
        rewards.append(policy.run_policy_in_env(env_render if i == 0 and epoch % 1 == 0 else env))
        if rewards[-1] > max_reward:
            max_reward, index = rewards[-1], i
    #env_render.render("human")

    population = tournament_sel(population, rewards, index)
    population = mutation(population)
    population = crossover(population)

    print("Epoch = %4i:" % epoch, max_reward)



