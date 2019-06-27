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


class AntPolicy(object):
    def __init__(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input((28,)),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(8, activation=None),
        ])
        self.model = model

    def create_new(self):
        return AntPolicy()

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
            env.render("human")

            score += reward
            frame += 1
        return score


class WalkerPolicy(object):
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

    def create_new(self):
        return WalkerPolicy()

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

    def run_policy_in_env(self, env, run_seed, render=False):
        env.seed(run_seed)
        self.restart_simulation()
        obs = env.reset()

        score = 0
        frame = 0
        done = False
        while not done:
            action = self.act(obs)

            obs, reward, done, _ = env.step(action)
            if render:
              env.render("human")

            score += reward
            frame += 1
        return score
