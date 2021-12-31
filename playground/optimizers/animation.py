from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def create_animation(
    paths: list,
    colors: list,
    names: list,
    func: Callable,
    figsize: tuple = (12, 12),
    x_lim: tuple = (-2, 2),
    y_lim: tuple = (-1, 3),
    n_seconds: int = 5,
) -> FuncAnimation:
    """

    Args:
        paths: List of arrays representing the paths (history of x,y coordinates) the optimizer went through
        colors: List of strings representing colours for each path.
        names: List of string representing names for each path.
        figsize: Size of the figure.
        x_lim: x coordinate limit of figure.
        y_lim: y coordinate limit fo figure.
        n_seconds: Number of seconds the animation should last for.
        func: Function to optimize for.

    Returns:
        anim: Animation of the paths of all the optimizers.
    """
    if not (len(paths) == len(colors) == len(names)):
        raise ValueError

    path_length = max(len(path) for path in paths)

    n_points = 300
    x = np.linspace(*x_lim, n_points)
    y = np.linspace(*y_lim, n_points)
    X, Y = np.meshgrid(x, y)
    Z = func([X, Y])

    minimum = (1.0, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contour(X, Y, Z, 90, cmap="jet")

    scatters = [
        ax.scatter(None, None, label=label, c=c)
        for c, label in zip(colors, names)
    ]

    ax.legend(prop={"size": 25})
    ax.plot(*minimum, "rD")

    def animate(i):
        for path, scatter in zip(paths, scatters):
            scatter.set_offsets(path[:i, :])

        ax.set_title(str(i))

    ms_per_frame = 1000 * n_seconds / path_length

    anim = FuncAnimation(
        fig, animate, frames=path_length, interval=ms_per_frame
    )

    return anim
