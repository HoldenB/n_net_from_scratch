from functools import reduce
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()


class LayerDense:
    def __init__(self, n_inputs, n_neurons) -> None:
        # Initial weights are random gaussian bounded around 0 with variance 1
        # By doing inputs * neurons we avoid having to transpose
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(shape=(1, n_neurons))

    def forward(self, inputs):
        # Cache the inputs
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values):
        # Gradients on parameters
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
        # Gradient on values
        self.d_inputs = np.dot(d_values, self.weights.T)


class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, d_values):
        # We need to modify original so we need to copy
        self.d_inputs = d_values.copy()
        # Zero gradient where input values are negative
        self.d_inputs[self.inputs <= 0] = 0


class ActivationSoftMax:
    def forward(self, inputs):
        self.inputs = inputs
        # Un-normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Norm for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities


class Loss:
    # Calculates the data and regularization losses given model output
    # and ground truth values
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class LossCategoricalCrossEntropy(Loss):
    def forward(self, y_predict, y_true):
        sample_length = len(y_predict)
        # Clip data on both sides to prevent division by 0 and -inf
        y_predict_clipped = np.clip(y_predict, 1e-7, 1 - 1e-7)

        # If Scalar values (not one-hot encoded), i.e. scalar values for
        # categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_predict_clipped[range(sample_length), y_true]
        else:
            # If One-Hot encoded (shape == 2 /  vector of vectors)
            correct_confidences = np.sum(y_predict_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, d_values, y_true):
        num_samples = len(d_values)
        num_labels = len(d_values[0])

        # If labels are sparse, turn them into one hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(num_labels)[y_true]

        # Calculate gradient
        self.d_inputs = -y_true / d_values
        # Normalize gradient
        self.d_inputs = self.d_inputs / num_samples


def main():
    # 100 feature sets of 3 classes
    (X, y) = spiral_data(samples=100, classes=3)

    # Inputs are 2 because we have 2 unique features that describe
    # the data (x and y axis)
    layer_one = LayerDense(2, 3)
    layer_two = LayerDense(3, 3)
    activation_ReLU = ActivationReLU()
    activation_softMax = ActivationSoftMax()
    loss_function = LossCategoricalCrossEntropy()

    # Layer one forward
    layer_one_output = layer_one.forward(X)
    activation_one_output = activation_ReLU.forward(layer_one_output)

    # Layer two forward
    layer_two_output = layer_two.forward(activation_one_output)
    activation_two_output = activation_softMax.forward(layer_two_output)

    print(activation_two_output[:5])

    loss = loss_function.calculate(activation_two_output, y)
    print(loss)


if __name__ == "__main__":
    main()
