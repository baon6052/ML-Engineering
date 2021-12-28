import pathlib
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch.utils.tensorboard import SummaryWriter


class Network(Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = Linear(10, 20)
        self.fc_2 = Linear(20, 30)
        self.fc_3 = Linear(30, 2)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)

        x = F.relu(x)

        return x


if __name__ == "__main__":
    log_dir = pathlib.Path.cwd() / "tensorboard_logs"
    writer = SummaryWriter(log_dir)

    x = torch.randn(1, 10)
    net = Network()

    def activation_hook(
        inst: torch.nn.Module, inp: Tuple[torch.Tensor], out: torch.Tensor
    ):
        """
        Args:
            inst: The layer we want to attach the hook to.
            inp: The input to the `forward` method.
            out: The output of the `forward` method.
        """
        writer.add_histogram(repr(inst), out)

    handle_1 = net.fc_1.register_forward_hook(activation_hook)
    handle_2 = net.fc_2.register_forward_hook(activation_hook)
    handle_3 = net.fc_3.register_forward_hook(activation_hook)

    y = net(x)

    """
    Remove hooks via the following
    handle_1.remove()
    handle_2.remove()
    handle_3.remove()
    """
