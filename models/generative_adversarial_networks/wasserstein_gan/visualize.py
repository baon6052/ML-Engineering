from typing import Tuple

import matplotlib.pyplot as plt
from torch import Tensor


def visualize_snapshots(snapshots: list[Tuple[int, Tensor, Tensor]]):
    for snapshot_epoch, snapshot_A, snapshot_B in snapshots:
        print(f"\n\n Results At Epoch {snapshot_epoch} \n\n")
        plt.figure(figsize=(6, 6))
        for i in range(2):
            plt.subplot(1, 2, 1)
            plt.xticks([])
            plt.yticks([])
            plt.axis("off")
            plt.grid(False)
            plt.imshow(snapshot_A[0].swapaxes(0, 1).swapaxes(1, 2).cpu())
            plt.subplot(1, 2, 2)
            plt.xticks([])
            plt.yticks([])
            plt.axis("off")
            plt.grid(False)
            plt.imshow(
                snapshot_B[0].swapaxes(0, 1).swapaxes(1, 2).detach().cpu()
            )
        plt.show()
