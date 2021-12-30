import torch
from torch.nn import LSTM, Embedding, Linear, Module


class Network(Module):
    """Custom network predicting the next character of a string.

    Parameters:
        vocab_size: The number of characters in the vocabulary.
        embedding_dim: Dimension of the character embedding vectors.
        dense_dim: Number of neurons in the linear layer that follows the LSTM.
        hidden_dim: Size of the LSTM hidden state.
        max_norm: If any of the embedding vectors has a higher L2 norm that max_norm, it is rescaled.
        n_layers: Number of layers of the LSTM.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=2,
        dense_dim=32,
        hidden_dim=8,
        max_norm=2,
        n_layers=1,
    ):
        super().__init__()
        self.embedding = Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=vocab_size - 1,
            norm_type=2,
            max_norm=max_norm,
        )
        self.lstm = LSTM(
            embedding_dim, hidden_dim, batch_first=True, num_layers=n_layers
        )
        self.linear_1 = Linear(hidden_dim, dense_dim)
        self.linear_2 = Linear(dense_dim, vocab_size)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor = None, c: torch.Tensor = None
    ):
        """
        Args:
            x: Input tensor of shape `(n_samples, window_size)` of dtype `torch.int64`
            h: Hidden states of the LSTM
            c: Hidden states of the LSTM


        Returns:
           logits: Tensor of shape `(n_samples, vocab_size)`
           h: Hidden states of the LSTM.
           c: Hidden states of the LSTM.
        """
        emb = self.embedding(x)
        if h is not None and c is not None:
            _, (h, c) = self.lstm(emb, (h, c))
        else:
            _, (h, c) = self.lstm(emb)

        # (n_samples, hidden_dim)
        h_mean = h.mean(dim=0)

        # (n_samples, dense_dim)
        x = self.linear_1(h_mean)

        # (n_samples, vocab_size)
        logits = self.linear_2(x)

        return logits, h, c
