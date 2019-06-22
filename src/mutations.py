MUTATION_WEIGHT_PROB = 0.01
MUTATION_PROB = 0.3
MUTATION_LAYER_PROB = 0.3
MUTATION_SIGMA = 0.01

import numpy as np

def normal(policy):
    if np.random.rand() < MUTATION_PROB:
        new_weights = []
        for weights in policy.get_weights():
            if np.random.rand() < MUTATION_LAYER_PROB:
                mutate_weight = np.random.rand(*weights.shape) < MUTATION_WEIGHT_PROB
                new_weights.append(weights + np.random.normal(0, MUTATION_SIGMA, weights.shape) * mutate_weight )
            else:
                new_weights.append(weights)
        policy = policy.create_new()
        policy.set_weights(new_weights)
    return policy

def uniform(policy):
    if np.random.rand() < MUTATION_PROB:
        new_weights = []
        for weights in policy.get_weights():
            if np.random.rand() < MUTATION_LAYER_PROB:
                mutate_weight = np.random.rand(*weights.shape) < MUTATION_WEIGHT_PROB
                new_weights.append(weights + np.random.uniform(-MUTATION_SIGMA, MUTATION_SIGMA, weights.shape) * mutate_weight)
            else:
                new_weights.append(weights)
        policy = policy.create_new()
        policy.set_weights(new_weights)
    return policy

MUTATIONS = {
    "normal": normal,
    "uniform": uniform,
}