import numpy as np


class LayerDense:
    def __init__(self, n_inputs, n_neurons) -> None:
        # Initial weights are random gaussian bounded around 0 with variance 1
        # By doing inputs X neurons we avoid having to transpose
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(shape=(1, n_neurons))

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases


def ex_one() -> None:
    X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

    layer_one = LayerDense(4, 5)
    layer_two = LayerDense(5, 2)

    output = layer_one.forward(X)
    output_two = layer_two.forward(output)
    print(output_two)

    # Example without the layer obj
    # ------------------------------------

    # Example inputs and weights
    X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

    weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]

    # Need to transpose weights in order to dot the matrices
    weights_transposed = np.array(weights).T

    biases = [2, 3, 0.5]

    weights_second_layer = [
        [0.1, -0.14, 0.5],
        [-0.5, 0.12, -0.33],
        [-0.44, 0.73, -0.13],
    ]

    weights_second_layer_transposed = np.array(weights_second_layer).T

    biases_second_layer = [-1, 2, -0.5]

    # Become input to second layer
    layer_one_outputs = np.dot(X, weights_transposed) + biases

    layer_two_outputs = (
        np.dot(layer_one_outputs, weights_second_layer_transposed) + biases_second_layer
    )
    print(layer_two_outputs)


ex_one()
