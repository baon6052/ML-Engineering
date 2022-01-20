from typing import List, Tuple, Type

import click
import numpy as np
import torch
import torch.optim as optim
from livelossplot import PlotLosses
from torch import Tensor

from models.generative_adversarial_networks.wasserstein_gan.data.dataset import (
    train_loader,
)
from models.generative_adversarial_networks.wasserstein_gan.model.discriminator import (
    Discriminator,
)
from models.generative_adversarial_networks.wasserstein_gan.model.generator import (
    Generator,
)
from models.generative_adversarial_networks.wasserstein_gan.model.initializer import (
    initialize_weights,
)
from utilities.visualize import (
    visualize_snapshots_grid_progress_2d,
)
from utilities.utils import EasyDict


def train(
    gen: Type[Generator],
    disc: Type[Discriminator],
    optimizer_gen: Type[optim.RMSprop],
    optimizer_disc: Type[optim.RMSprop],
    c: Type[EasyDict],
) -> List[Tuple[int, Tensor, Tensor]]:

    # For logging and visualizing results
    fixed_noise = torch.randn(32, c.g_kwargs.z_dim, 1, 1).to(c.device)
    snapshots = []
    gen_loss_array = []
    disc_loss_array = []

    liveplot = PlotLosses()

    for epoch in range(c.num_epochs):

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(c.device)
            cur_batch_size = data.shape[0]
            print(cur_batch_size)
            # Train Discriminator
            for _ in range(c.d_kwargs.d_iterations):
                noise = torch.randn(cur_batch_size, c.g_kwargs.z_dim, 1, 1).to(
                    c.device
                )
                fake = gen(noise)
                disc_real = disc(data).reshape(-1)
                disc_fake = disc(fake).reshape(-1)
                loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake))
                disc.zero_grad()
                loss_disc.backward(retain_graph=True)
                optimizer_disc.step()

                # Clip Discriminator weights
                for p in disc.parameters():
                    p.data.clamp_(
                        c.d_kwargs.weight_clip_lower,
                        c.d_kwargs.weight_clip_upper,
                    )

            # Train Generator
            gen_fake = disc(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()

            gen_loss_array.append(loss_gen.item())
            disc_loss_array.append(loss_disc.item())

        fake = gen(noise)
        snapshots.append((epoch, fake))

        liveplot.update(
            {
                "Generator Loss": np.mean(gen_loss_array),
                "Discriminator Loss": np.mean(disc_loss_array),
            }
        )
        liveplot.draw()

    return snapshots


@click.command()

# Training and data config
@click.option(
    "--num_epochs", help="The number of epochs to train model", default=5
)
@click.option(
    "--channels_img",
    help="The number of colour channels in image training set",
    default=1,
)

# Generator config
@click.option("--glr", help="Learning Rate of the Generator", default=5e-5)
@click.option("--g_batch_size", help="Batch size of Generator", default=64)
@click.option(
    "--z_dim", help="The random vector input size to the Generator", default=128
)
@click.option(
    "--g_features",
    help="The Generator's number of initial features to transpose to",
    default=64,
)

# Discriminator config
@click.option("--dlr", help="Learning Rate of the Discriminator", default=5e-5)
@click.option("--d_batch_size", help="Batch size of Discriminator", default=64)
@click.option(
    "--d_iterations",
    help="The number of iterations to train discriminator per epoch",
    default=5,
)
@click.option(
    "--d_features",
    help="The Generator's number of initial features to convolve to",
    default=64,
)
@click.option(
    "--weight_clip_upper",
    help="The upper value to clip weights for the discriminator",
    default=0.01,
)
@click.option(
    "--weight_clip_lower",
    help="The lower value to clip weights for the discriminator",
    default=-0.01,
)
def main(**kwargs):
    print(kwargs)
    # Command line arguments.
    opts = EasyDict(kwargs)

    # Main config
    c = EasyDict()
    c.g_kwargs = EasyDict(class_name="Generator", features_gen=64)
    c.d_kwargs = EasyDict(class_name="Discriminator", features_gen=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c.device = device

    c.num_epochs = opts.num_epochs
    c.channels_img = opts.channels_img

    # Generator config
    c.g_kwargs.lr = opts.glr
    c.g_kwargs.g_batch_size = opts.g_batch_size
    c.g_kwargs.z_dim = opts.z_dim
    c.g_kwargs.g_features = opts.g_features

    # Discriminator config
    c.d_kwargs.lr = opts.dlr
    c.d_kwargs.d_batch_size = opts.d_batch_size
    c.d_kwargs.d_iterations = opts.d_iterations
    c.d_kwargs.d_features = opts.d_features
    c.d_kwargs.weight_clip_upper = opts.weight_clip_upper
    c.d_kwargs.weight_clip_lower = opts.weight_clip_lower

    # Initialize models
    gen = Generator(c.g_kwargs.z_dim, c.channels_img, c.g_kwargs.g_features).to(
        device
    )
    disc = Discriminator(c.channels_img, c.d_kwargs.d_features).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    # Initialize optimizers
    optimizer_gen = optim.RMSprop(gen.parameters(), lr=c.g_kwargs.lr)
    optimizer_disc = optim.RMSprop(disc.parameters(), lr=c.d_kwargs.lr)

    # Train model and log training snapshots
    snapshots = train(gen, disc, optimizer_gen, optimizer_disc, c)

    visualize_snapshots_grid_progress_2d(snapshots)


if __name__ == "__main__":
    main()
