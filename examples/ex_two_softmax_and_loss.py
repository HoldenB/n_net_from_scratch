import math
import nnfs
import numpy as np
from functools import reduce


nnfs.init()


def ex_two() -> None:
    # Needs to work on a batch of data
    layer_outputs = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]

    # 2.7182...
    # E = math.e

    # exp_values = [E**output for output in layer_outputs]
    # Numpy version
    exp_values = np.exp(layer_outputs)
    print(exp_values)
    print()

    # We need to normalize
    # norm_base = sum(exp_values)
    # norm_values = [val / norm_base for val in exp_values]
    # Norm with numpy
    # Axis 1 and keepdims will ensure the sum is a matrix of the same size
    # as inputs
    norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    print(norm_values)
    # Should be very close to 1
    print(sum(norm_values))
    print()

    # Some practice with logs
    """
    Solving for x
    
    e ** x = b
    """
    b = 5.2
    print(np.log(b))  # ~1.6485
    print(math.e**1.6485)  # ~5.2
    print()

    # Calculating Categorical Cross-Entropy
    softmax_output = [0.7, 0.1, 0.2]
    target_output_one_hot_vector = [1, 0, 0]
    loss = -reduce(
        lambda a, b: a + b,
        *[
            map(
                lambda tup: math.log(tup[0]) * tup[1],
                zip(softmax_output, target_output_one_hot_vector),
            )
        ]
    )
    print(loss)
    print()

    # We can do this easily with numpy
    # # According to our class_targets:
    softmax_outputs = [
        [0.7, 0.2, 0.1],  # index 0 for dog = 0.7
        [0.5, 0.1, 0.4],  # index 1 for cat = 0.5
        [0.02, 0.9, 0.08],  # index 1 for cat = 0.9
    ]
    # For example if Dog = 0, Cat = 1
    class_targets = [0, 1, 1]

    softmax_outputs = np.array(softmax_outputs)
    # This will correctly chose the values for the class targets
    print(softmax_outputs[range(len(softmax_outputs)), class_targets])
    # Now for loss:
    loss = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
    print(loss)
    # However now we cannot take the mean/avg because what if we have 0.0 in our outputs?
    # This will result in inf when you take the negative log value, and will cause our mean
    # to become inf
    # To accommodate for this, we will need to clip the y-predicted-value
    # y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

    # Using argmax to calculate accuracy:
    # Argmax will choose the indices with the max values, we want these to
    # match what the class targets say because those are the expected outputs
    # and means that we correctly classified the class(es)
    predictions = np.argmax(softmax_outputs, axis=1)

    # Accuracy will be a vector with outputs of [0, 1, 1, 0, etc] where 1
    # appears when something was properly classified
    accuracy = np.mean(predictions == class_targets)
    print(accuracy)


ex_two()
