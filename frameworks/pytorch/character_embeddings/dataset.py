from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset


class CharacterDataset(Dataset):
    """Custom dataset.

    Parameters:
        text: Input test that will be used to create the entire database.
        window_size: Number of characters to use as input features.
        vocab_size: Number of characters in the vocabulary. The last character is always reserved for a special "-" out-of-vocabulary character.

    Attributes:
        ch2ix: Mapping from the character to the position of that character in the vocabulary. Note that all characters that are not in the vocabulary will get mapped into the index `vocab_size - 1`.
        ix2ch: Mapping from the character position in the vocabulary to the actual character.
        vocabulary: List of all characters. `len(vocabulary) == vocab_size`
    """

    def __init__(self, text: str, window_size: int = 1, vocab_size: int = 50):
        self.text = text.replace("\n", " ")
        self.window_size = window_size
        self.ch2ix = defaultdict(lambda: vocab_size - 1)

        most_common_ch2ix = {
            x[0]: i
            for i, x in enumerate(
                Counter(self.text).most_common()[: (vocab_size - 1)]
            )
        }

        self.ch2ix.update(most_common_ch2ix)
        self.ch2ix["~"] = vocab_size - 1

        self.ix2ch = {v: k for k, v in self.ch2ix.items()}
        self.vocabulary = [self.ix2ch[i] for i in range(vocab_size)]

    def __len__(self):
        return len(self.text) - self.window_size

    def __getitem__(self, ix: int):
        X = torch.LongTensor(
            [self.ch2ix[c] for c in self.text[ix : ix + self.window_size]]
        )
        y = self.ch2ix[self.text[ix + self.window_size]]

        return X, y
