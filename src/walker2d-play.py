#!/usr/bin/env

import numpy as np
import tensorflow as tf
import os.path, gym
import roboschool


class HumanoidPolicy(object):
    def __init__(self):
        self.memory_dim = 10
        model = tf.keras.Sequential([
            tf.keras.layers.Input((self.memory_dim + 22,)),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(self.memory_dim + 6, activation=None),
        ])
        self.model = model
        self.restart_simulation()

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def restart_simulation(self):
        self.memory = np.zeros((self.memory_dim))

    def set_pretrain(self, pretrain_weights_without_memory, current_coef):
        pretrain_weights = [ x for x in pretrain_weights_without_memory]

        input_weights = pretrain_weights[0]
        input_weights = np.concatenate((input_weights, np.zeros((self.memory_dim, input_weights.shape[1]))))
        pretrain_weights[0] = input_weights

        output_weights = pretrain_weights[-2]
        output_weights = np.concatenate((output_weights, np.zeros((output_weights.shape[0], self.memory_dim))), axis=1)
        pretrain_weights[-2] = output_weights

        output_weights = pretrain_weights[-1]
        output_weights = np.concatenate((output_weights, np.zeros((self.memory_dim,))), axis=0)
        pretrain_weights[-1] = output_weights

        new_weights = []
        for random, pretrain in zip(self.get_weights(), pretrain_weights):
            new_weights.append(random * current_coef + pretrain)
        self.set_weights(new_weights)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def act(self, obs_data):
        observation = np.array(obs_data)
        input_with_memory = np.concatenate((observation, self.memory))

        input_data = np.expand_dims(input_with_memory, axis = 0)
        action = self.model(input_data, training = False)
        output = action[0].numpy()
        self.memory = output[-10:]
        return output[:-10]

    def run_policy_in_env(self, env):
        self.restart_simulation()
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
env = gym.make("RoboschoolWalker2d-v1")
env_render = env
#env_render = gym.make("RoboschoolAnt-v1")

POPULATION_SIZE = 11
EPOCH_COUNT = 10000
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
import default_weights_2d

for policy in population:
    #policy.set_pretrain(default_weights_2d.default_weights, 0.3)
    policy.load_weights("logdir/" + "walker2d.py" + "-epoch-%i" % 30)
    pass

for epoch in range(EPOCH_COUNT):
    rewards = []
    max_reward = -1e10
    index = 0
    for i, policy in enumerate(population):
        rewards.append(policy.run_policy_in_env(env_render if i == 0 and epoch % 1 == 0 else env))
        if rewards[-1] > max_reward:
            max_reward, index = rewards[-1], i
    #env_render.render("human")

    #if epoch % 10 == 0:
        #population[index].save_weights("logdir/" + __file__ + "-epoch-%i" % epoch)

    population = tournament_sel(population, rewards, index)
    population = mutation(population)
    #population = crossover(population)

    print("Epoch = %4i:" % epoch, max_reward)



