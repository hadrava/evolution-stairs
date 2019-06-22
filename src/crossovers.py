import numpy as np

CROSSOVER_WEIGHT_PROB = 0.01
CROSSOVER_PROB = 0.3
CROSSOVER_LAYER_PROB = 0.3

def linearmove_per_layer(first, second):
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
        first, second = first.create_new(), second.create_new()
        first.set_weights(new_first_weights)
        second.set_weights(new_second_weights)
    return first, second


def onepoint_per_layer(first, second):
    if np.random.rand() < CROSSOVER_PROB:
        new_first_weights = []
        new_second_weights = []
        for first_weights, second_weights in zip(first.get_weights(), second.get_weights()):
            if np.random.rand() < CROSSOVER_LAYER_PROB:
                point = np.random.randint(first_weights.shape[0])
                new_first_weights.append(np.concatenate((first_weights[:point], second_weights[point:]),axis=0))
                new_second_weights.append(np.concatenate((second_weights[:point], first_weights[point:]),axis=0))
            else:
                new_first_weights.append(first_weights)
                new_second_weights.append(second_weights)
        first, second = first.create_new(), second.create_new()
        first.set_weights(new_first_weights)
        second.set_weights(new_second_weights)
    return first, second


def twopoint_per_layer(first, second):
    if np.random.rand() < CROSSOVER_PROB:
        new_first_weights = []
        new_second_weights = []
        for first_weights, second_weights in zip(first.get_weights(), second.get_weights()):
            if np.random.rand() < CROSSOVER_LAYER_PROB:
                point1 = np.random.randint(first_weights.shape[0])
                point2 = np.random.randint(first_weights.shape[0])
                (point1, point2) = (point2, point1) if point2 < point1 else (point1, point2)
                new_first_weights.append(np.concatenate((first_weights[:point1], second_weights[point1:point2], first_weights[point2:]),axis=0))
                new_second_weights.append(np.concatenate((second_weights[:point1], first_weights[point1:point2], second_weights[point2:]),axis=0))
            else:
                new_first_weights.append(first_weights)
                new_second_weights.append(second_weights)
        first, second = first.create_new(), second.create_new()
        first.set_weights(new_first_weights)
        second.set_weights(new_second_weights)
    return first, second


def onepoint(first, second):
    if np.random.rand() < CROSSOVER_PROB:
        first_weights = first.get_weights()
        second_weights = second.get_weights()
        point = np.random.randint(len(first_weights) // 2) * 2 # by 2 is because 2 weights are per layer (second is bias)
        new_first_weights = first_weights[:point] + second_weights[point:]
        new_second_weights = second_weights[:point] + first_weights[point:]
        first, second = first.create_new(), second.create_new()
        first.set_weights(new_first_weights)
        second.set_weights(new_second_weights)
    return first, second


def twopoint(first, second):
    if np.random.rand() < CROSSOVER_PROB:
        first_weights = first.get_weights()
        second_weights = second.get_weights()
        point1 = np.random.randint(len(first_weights) // 2) * 2 # by 2 is because 2 weights are per layer (second is bias)
        point2 = np.random.randint(len(first_weights) // 2) * 2 # by 2 is because 2 weights are per layer (second is bias)
        (point1, point2) = (point2, point1) if point2 < point1 else (point1, point2)
        new_first_weights = first_weights[:point1] + second_weights[point1:point2] + first_weights[point2:]
        new_second_weights = second_weights[:point1] + first_weights[point1:point2] + second_weights[point2:]
        first, second = first.create_new(), second.create_new()
        first.set_weights(new_first_weights)
        second.set_weights(new_second_weights)
    return first, second

CROSSOVERS = {
    "linearmove_per_layer": linearmove_per_layer,
    "onepoint_per_layer": onepoint_per_layer,
    "twopoint_per_layer": twopoint_per_layer,
    "onepoint": onepoint,
    "twopoint": twopoint,
              }