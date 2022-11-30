

def ex_four():
    # Results from a forward pass
    x = [1.0, -2.0, 3.0]    # input values
    w = [-3.0, -1.0, 2.0]   # weights
    b = 1.0                 # bias

    # ReLu = max(x, 0) so dReLu = 1(x > 0) so 1 if z > 0 else 0
    # z will be the output of ReLu(w * i + b) or C(w * i + b)

    # Multiply inputs by weights
    x_w_result = [x * w for (x, w) in zip(x, w)]

    # Add weights + bias for z i.e. sum(all_weights) + b
    z = sum(x_w_result) + b

    # Activation function
    y = max(z, 0)

    print(f'First y: {y}')

    # Backwards pass
    # Deriv from the next layer
    d_value = 1.0

    # Deriv of ReLu and the chain rule (needs result from next layer
    # hence the "backwards pass")
    d_ReLu_dz = d_value * (1. if z > 0 else 0.)

    # Moving backwards, what comes before we perform the activation function?
    # The sum of the weighted inputs and bias.
    # We need to calc the partial deriv of the sum function, then with the chain
    # rule, multiply this by the partial deriv of the subsequent outer function,
    # which is ReLu.

    # Notation def: d_ReLu_dx_w0 = the partial deriv of ReLu w.r.t the first
    # weighted input, w0x0.

    # Note: Partial deriv of sum(x, y) op is always 1, regardless of outputs

    # Partial deriv of sum w.r.t x, weighted for the 0th pair of inputs and weights

    # Now the bias
    d_sum_db = 1

    # Gradient on bias (just one bias)
    d_ReLu_db = d_ReLu_dz * d_sum_db

    # Gradients on weights
    d_ReLu_dw = [d_ReLu_dz * 1, d_ReLu_dz * 1, d_ReLu_dz * 1]

    # Now the multiplication comes before the addition of weights & biases
    # (Multiplication of each input with each weight)

    # Note: Partial deriv of multi(x, y) is always the opposite of w.r.t
    # d/dx f(x, y) = y
    # d/dy f(x, y) = x

    # Since the above is true, the deriv w.r.t x will be the weight i.e. y
    # Gradient on inputs
    d_ReLu_dx = [d_w * d_ReLu_dx_w for (d_w, d_ReLu_dx_w) in zip(w, d_ReLu_dw)]

    # Typically this is the job of the optimizer. We can apply a simple version
    # of this by directly applying a negative fraction of the gradient to current
    # weights and biases. This will do a "gradient descent".
    updated_weights = [w + -0.001 * d_w for (w, d_w) in zip(w, d_ReLu_dw)]

    updated_bias = b + -0.001 * d_ReLu_db

    print('Updated weights and bias')
    print(updated_weights)
    print(updated_bias)

    x_w_result = [x * w for (x, w) in zip(x, updated_weights)]
    z = sum(x_w_result) + updated_bias
    y = max(z, 0)

    print(f'Second y: {y}')


ex_four()