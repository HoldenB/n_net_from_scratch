import numpy as np


def ex_five():
    # Passed in gradient from the next layer (remember we're going backwards...)
    d_values = np.array([[1., 1., 1.],
                         [2., 2., 2.],
                         [3., 3., 3.]])

    # 3 sets of inputs (i. e. sample size of 3)
    inputs = np.array([[1, 2, 3, 2.5],
                       [2., 5., -1., 2],
                       [-1.5, 2.7, 3.3, -0.8]])


    # 3 sets of weights, one for each neuron
    # 4 inputs (i.e. 4 features), thus 4 weights
    # Remember: We keep the weights transposed
    weights = np.array([[0.2, 0.8, -0.5, 1],
                        [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87]]).T

    # One bias for each neuron
    # Biases are a row vector with shape (1, neurons)
    biases = np.array([[2, 3, 0.5]])

    # Forward pass
    layer_outputs = np.dot(inputs, weights) + biases    # Dense layer
    reLu_outputs = np.maximum(0, layer_outputs)         # ReLU activation

    # Optimize and test back-propagation here
    # ReLU activation - simulates derivative with respect to input values
    # from next layer passed to current layer during back-prop
    d_reLu = reLu_outputs.copy()
    d_reLu[layer_outputs <= 0] = 0
    
    # Dense layer
    d_inputs = np.dot(d_reLu, weights.T)
    d_weights = np.dot(inputs.T, d_reLu)
    
    # Sum bias values over samples (first axis) - keep_dims
    # Will give us a plain list
    d_biases = np.sum(d_reLu, axis=0, keepdims=True)
    
    # Update parameters
    weights += -0.001 * d_weights
    biases += -0.001 * d_biases
    
    print(weights)
    print(biases)


ex_five()
