# This code uses a genetic algorithm to train a feed forward neural network to learn to approximate the cos(x) function. It is explicitly written to be easily understandable, at the cost of it's size.
# It has no dependencies except Numpy (and could easily be modified to not even require that).
# Note that this is written and published only with education purposes in mind, real-world training of neural networks is much more efficient with backpropagation.
# Genetic algorithms are worth understanding because, while being resource intensive, they make no requirements for end-to-end differentiability, can easily handle discrete variables and are easy to code.

import random
import numpy as np
import math

# Our dataset
dataset = []

# Where to start the X's
x_runner = -1.0

# Generate the dataset, we're going to learn the cos(x) function
for i in range(200):
    dataset.append([x_runner, math.cos(x_runner)])
    x_runner += 0.01

# Shuffle the dataset to make the training and testing sets more evenly distributed
random.shuffle(dataset)

# Split into training and testing sets
training_dataset = dataset[:150]
testing_dataset = dataset[:-50]
# Training inputs
x_training = list(item[0] for item in training_dataset)
# Training outputs
y_training = list(item[1] for item in training_dataset)
# Testing inputs
x_testing = list(item[0] for item in training_dataset)
# Testing outputs
y_testing = list(item[1] for item in training_dataset)

# Since we're going to be using a genetic algorithm, we need something that will give us a randomly initialized neural network with the structure we want. We will assume one hidden layer.
def produce_neural_network(inputs_size, hidden_layer_size, outputs_size, randomize = True):
    net = {}
    net['inputs_size'] = inputs_size
    net['hidden_size'] = hidden_layer_size
    net['outputs_size'] = outputs_size
    if randomize:
        net['weights_in'] = np.subtract(np.random.rand(inputs_size*hidden_layer_size), 0.5)
        net['weights_out'] = np.subtract(np.random.rand(hidden_layer_size * outputs_size), 0.5)
    net['biases'] = np.subtract(np.random.rand(hidden_layer_size), 0.5)
    return net

# And we need a way to run that neural network on inputs to generate outputs
def run_neural_network(x, net):
    if len(x) <> net['inputs_size']:
        raise Exception('Input dimensions do not match network structure')
    hidden_layer = np.add(np.multiply(x, net['weights_in']), net['biases'])
    # Now for RELU
    hidden_layer = hidden_layer * (hidden_layer > 0)
    output_layer = np.matmul( np.matrix([hidden_layer]), np.matrix([net['weights_out']]).T)
    return output_layer

# Genetic algorithm function: As we'll see in a little bit, we need a way to "breed" arrays
def breed_arrays(mother, father, crossover, mutation):
    if len(mother) <> len(father):
        raise Exception('Incompatible parents')
    child = []
    for i in range(len(mother)):
        crossover_decider = random.random()
        childvalue = 0
        if crossover_decider < crossover:
            childvalue = mother[i]
        else:
            childvalue = father[i]
        mutation_decider = random.random()
        if mutation_decider < mutation:
            childvalue += random.random() - 0.5
        child.append(childvalue)
    return child

# Genetic algorithm function: We need to breed neural networks, creating one new network from two parents
def breed_neural_networks(mother_net, father_net, crossover = 0.6, mutation = 0.02 ):
    if mother_net['inputs_size'] <> father_net['inputs_size'] or mother_net['hidden_size'] <> father_net['hidden_size'] or  mother_net['outputs_size'] <> father_net['outputs_size']:
        raise Exception('Incompatible parents')
    child_net = produce_neural_network(mother_net['inputs_size'], mother_net['hidden_size'], mother_net['outputs_size'], randomize=False)
    child_net['biases'] = breed_arrays(mother_net['biases'], father_net['biases'], crossover, mutation)
    child_net['weights_in'] = breed_arrays(mother_net['weights_in'], father_net['weights_in'], crossover, mutation)
    child_net['weights_out'] = breed_arrays(mother_net['weights_out'], father_net['weights_out'], crossover, mutation)
    return child_net

# The final part of the genetic stuff is a fitness function
def evaluate_neural_network(net, input_data, output_data):
    if len(input_data) <> len(output_data):
        raise Exception('Input and output data arrays have different lengths')
    sumSquaredError = 0
    for dataIndex in range(len(input_data)):
        output = run_neural_network([ input_data[dataIndex] ], net)
        sampleError = abs(output - output_data[dataIndex])
        sumSquaredError += sampleError * sampleError
    # We return the sum squared error
    return sumSquaredError

# Need to set some parameters now

# Number of networks in each generation
generation_size = 40
# Size of the hidden layer
hidden_layer_size = 30
# Maximum number of generations to cycle through
generations_max = 100
# Acceptable error threshold
acceptable_error = 0.01

# Let's go ahead and create the first generation
current_generation = []
sorted_by_fitness = []
for i in range(generation_size):
    current_generation.append(produce_neural_network(1, hidden_layer_size, 1))

# Our main loop that does the evolution
for j in range(generations_max):
    # Evaluate the current generation
    for k in range(generation_size):
        current_generation[k]['sse'] = evaluate_neural_network(current_generation[k], x_training, y_training)
    # Sort by fitness
    sorted_by_fitness = sorted(current_generation, key=lambda k: k['sse'])

    # Print the best one
    print 'Best one in generation', j, 'has error', sorted_by_fitness[0]['sse']

    if sorted_by_fitness[0]['sse'] < acceptable_error:
        print 'Done!'
        break

    # Create a new generation and make sure we hang onto the best network so far
    new_generation = []
    new_generation.append(sorted_by_fitness[0])
    # Do breeding to create the next generation
    for l in range (1, generation_size):
        # Randomly pick mothers and fathers
        mother_index = 0
        father_index = 0
        while mother_index == father_index:
            mother_index = int(round((generation_size-1) * (random.random() * random.random())))
            father_index = int(round((generation_size-1) * random.random()))
        child = breed_neural_networks(sorted_by_fitness[mother_index], sorted_by_fitness[father_index])
        new_generation.append(child)
    current_generation = new_generation

# Let's test what we learned on the testing data
for i in range(len(x_testing)):
    input_value = x_testing[i]
    desired_output_value = y_testing[i]
    actual_output = run_neural_network([input_value], sorted_by_fitness[0])
    print 'From X=',input_value,'we wanted ',desired_output_value,' and got ',actual_output[0]