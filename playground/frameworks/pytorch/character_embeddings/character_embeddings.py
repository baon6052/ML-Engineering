from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from network import Network
from torch.nn import Module
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from dataset import CharacterDataset


def compute_loss(loss: Callable, net, dataloader):
    """Compute average loss over a dataset"""

    net.eval()
    all_losses = []
    for X_batch, y_batch in dataloader:
        probs, _, _ = net(X_batch)

        all_losses.append(loss(probs, y_batch).item())

    return np.mean(all_losses)


def generate_text(
    n_chars: int,
    net: Module,
    dataset: CharacterDataset,
    initial_text: str = "Hello",
    random_state: int = None,
):
    """
    Args:
        n_chars: Number of characters to generate.
        net: Charater-level model.
        dataset: Instance of the `CharacterDataset`.
        initial_text: The starting test to be used as the initial condition for the model.  # noqa: E501
        random_state: If not None, then the result is reproducible.

    Returns:
        res: Generated text.
    """
    if not initial_text:
        raise ValueError("You need to specify the initial text")

    res = initial_text
    net.eval()
    h, c = None, None

    if random_state is not None:
        np.random.seed(random_state)

    for _ in range(n_chars):
        previous_chars = initial_text if res == initial_text else res[-1]
        features = torch.LongTensor(
            [[dataset.ch2ix[c] for c in previous_chars]]
        )
        logits, h, c = net(features, h, c)
        probs = F.softmax(logits[0], dim=0).detach().numpy()
        new_ch = np.random.choice(dataset.vocabulary, p=probs)
        res += new_ch

    return res


if __name__ == "__main__":
    with open("text.txt", "r") as f:
        text = "\n".join(f.readlines())

    # Hyper-parameters model
    vocab_size = 70
    window_size = 20
    embedding_dim = 2
    hidden_dim = 16
    dense_dim = 32
    n_layers = 1
    max_norm = 2

    # Training config
    n_epochs = 15
    train_val_split = 0.8
    batch_size = 128
    random_state = 13

    torch.manual_seed(random_state)

    loss_f = torch.nn.CrossEntropyLoss()
    dataset = CharacterDataset(
        text, window_size=window_size, vocab_size=vocab_size
    )

    n_samples = len(dataset)
    split_ix = int(n_samples * train_val_split)

    train_indices, val_indices = np.arange(split_ix), np.arange(
        split_ix, n_samples
    )

    train_dataloader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(train_indices),
        batch_size=batch_size,
    )
    val_dataloader = DataLoader(
        dataset, sampler=SubsetRandomSampler(val_indices), batch_size=batch_size
    )

    DEVICE = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    net = Network(
        vocab_size,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dense_dim=dense_dim,
        embedding_dim=embedding_dim,
        max_norm=max_norm,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    emb_history = []

    for e in range(n_epochs + 1):
        net.train()
        for X_batch, y_batch in tqdm(train_dataloader):
            if e == 0:
                break

            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            prob, _, _ = net(X_batch)
            loss = loss_f(prob, y_batch)
            loss.backward()

            optimizer.step()

        train_loss = compute_loss(loss_f, net, train_dataloader)
        val_loss = compute_loss(loss_f, net, val_dataloader)
        print(f"Epoch: {e}, {train_loss:.3f}, {val_loss=:.3f}")

        # Generate ine sentence
        initial_text = "Once upon a time, "
        generated_text = generate_text(
            100,
            net,
            dataset,
            initial_text=initial_text,
            random_state=random_state,
        )
        print(generated_text)

        # Prepare DataFrame
        weights = net.embedding.weight.detach().clone().numpy()

        df = pd.DataFrame(
            weights, columns=[f"dim_{i}" for i in range(embedding_dim)]
        )
        df["epoch"] = e
        df["character"] = dataset.vocabulary

        emb_history.append(df)

    final_df = pd.concat(emb_history)
    final_df.to_csv("res.csv", index=False)
