import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 2 * x**2


def tangent_line(deriv, x, b):
    return deriv * x + b


def ex_three() -> None:

    plot_colors = ["k", "g", "r", "b", "c"]
    stop = len(plot_colors)

    # More granular approach than simply range(x)
    x = np.arange(0, stop, 0.001)

    y = f(x)
    plt.plot(x, y)

    for i in range(stop):
        # Numerical Differentiation
        delta = 0.001
        x1 = i
        x2 = x1 + delta

        y1 = f(x1)
        y2 = f(x2)
        print((x1, y1), (x2, y2))

        approx_deriv = (y2 - y1) / (x2 - x1)
        # Solving for b
        b = y2 - approx_deriv * x2

        # Plot the tangent line
        # +-0.9 to draw visible tan line on graph
        # Calc y for given x using tan line
        to_plot = [x1 - 0.9, x1, x1 + 0.9]

        plt.scatter(x1, y1, c=plot_colors[i])

        plt.plot(
            to_plot,
            [tangent_line(approx_deriv, x, b) for x in to_plot],
            c=plot_colors[i],
        )

        print(f"Approx deriv for f(x) where x={x1} is {approx_deriv}")

    plt.show()


ex_three()
