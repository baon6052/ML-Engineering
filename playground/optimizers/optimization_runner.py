from typing import Callable, Tuple

import numpy as np
import torch
from animation import create_animation
from torch.optim import SGD, Adam
from tqdm import tqdm


def rosenbrock(xy: Tuple) -> float:
    """
    Args:
        xy: Two element tuple of floats representing the x and y coordinates.

    Returns:
        The Rosenbrock function value evaluated at the point xy.
    """
    x, y = xy
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def run_optimization(
    xy_init: Tuple,
    optimizer_class: object,
    n_iter: int,
    func: Callable,
    **optimizer_kwargs: dict
) -> np.ndarray:
    """Run optimization finding the minimum of a given function.

    Args:
        xy_init: Starting point - two floats representing the x and y coordinates.  # noqa: E501
        optimizer_class: Optimizer class which we want to run.
        n_iter: Number of iterations to run the optimization for.
        func: Function to find the minimum against.
        optimizer_kwargs: Additional parameters to be passed into the optimizer.

    Returns:
        path: 2D array of shape (n_iter + 1, 2). Where the rows represent
        the iteration and the columns represent the x,y coordinates.
    """
    xy_t = torch.tensor(xy_init, requires_grad=True)
    optimizer = optimizer_class([xy_t], **optimizer_kwargs)

    path = np.empty((n_iter + 1, 2))
    path[0, :] = xy_init

    for i in tqdm(range(1, n_iter + 1)):
        optimizer.zero_grad()
        loss = func(xy_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm(xy_t, 1.0)
        optimizer.step()

        path[i, :] = xy_t.detach().numpy()

    return path


if __name__ == "__main__":
    xy_init = (0.3, 0.8)
    n_iter = 1000

    path_adam = run_optimization(xy_init, Adam, n_iter, rosenbrock)
    path_sgd = run_optimization(xy_init, SGD, n_iter, rosenbrock, lr=1e-3)

    freq = 10

    paths = [path_adam[::freq], path_sgd[::freq]]
    colors = ["green", "blue"]
    names = ["Adam", "SGD"]

    anim = create_animation(
        paths,
        colors,
        names,
        rosenbrock,
        figsize=(12, 7),
        x_lim=(-0.1, 1.1),
        y_lim=(-0.1, 1.1),
        n_seconds=7,
    )

    anim.save("result.gif", writer="imagemagick")
