import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Split image into patches and embed them.
    Args:
        img_size: Size of image.
        patch_size: Size of patches.
        in_channels: Number of input channels.
        embed_dim: The embedding dimension.

    Attributes:
        n_patches: The number of patches made from out image.
        proj: Convolutional layer to split image into patches and generate their embedding.  # noqa: E501

    """

    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Shape (n_samples, in_channels, img_size, img_size)

        Returns:
            A tensor with shape, (n_samples, n_patches, embed_dim)
        """

        # (n_sample, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = self.proj(x)

        # (n_samples, embed_dim, n_patches)
        x = x.flatten(2)

        # (n_samples, n_patches, embed_dim)
        x = x.transpose(1, 2)

        return x


class Attention(nn.Module):
    """Attention mechanism.

    Parameters:
        dim: The input and output dimension of per token features.
        n_heads: Number of attention heads.
        qkv_bias: If True then we include bias to the query, key and value projections.  # noqa: E501
        attn_p: Dropout probability applied to the query, key and value tensors.
        proj_p: Dropout probability applied to the output tensor.

    Attributes:
        scale: Normalizing constant for the dot product.
        qkv: Linear projection for the query, key and value.
        proj: Linear mapping that takes in the concatenated output of all attention heads and maps it into a new space.  # noqa: E501
        attn_drop, proj_drop: Dropout layers
    """

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0.0, proj_p=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.prof_drop = nn.Dropout(proj_p)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor with shape (n_samples, n_patches + 1, dim)

        Returns:
            A torch.Tensor of shape (n_Samples, n_patches + 1, dim)
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        # (n_sample, n_patches + 1, 3 * dim)
        qkv = self.qkv(x)

        # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)

        # (3, n_samples, n_heads, n_patches + 1, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        k_t = k.transpose(-2, -1)

        # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        dot_product = (q @ k_t) * self.scale

        # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dot_product.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = attn @ v

        # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)

        # (n_samples, n_patches + 1, dim)
        weighted_avg = weighted_avg.flatten(2)

        # (n_samples, n_patches + 1, dim)
        x = self.proj(weighted_avg)

        # (n_samples, n_patches + 1, dim)
        x = self.prof_drop(x)

        return x


class MLP(nn.Module):
    """Multilayer Perceptron

    Parameters:
        in_features: Number of input features.
        hidden_feature: Number of nodes in the hidden layer.
        out_features: Number of output features.
        p: Dropout probability.

    Attributes:
        fc: The first linear layer.
        act: GELU activation function
        fc2: The second linear layer.
    """

    def __init__(self, in_features, hidden_features, out_features, p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor):
        """Run forward pass.

        Args:
            x: Input tensor of shape (n_samples, n_patches + 1, in_features)

        Returns:
            An output tensor of shape (n_Samples, n_patches + 1, out_features)
        """

        # (n_samples, n_patches + 1, hidden_features)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        # (n_samples, n_patches + 1, out_features)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Block(nn.Module):
    """Transformer block

    Parameters:
        dim: Embedding dimension.
        n_heads: Number of attention heads.
        mlp_ratio: Determines the hidden dimension size of the `MLP` module with respect to `dim`.  # noqa: E501
        qkv_bias: If True then we include bias to the query, key and value projections.  # noqa: E501
        p, attn_p: Dropout probability.

    Attributes:
        norm1, norm2: Layer normalization.
        attn: Attention module.
        mlp: MLP module.
    """

    def __init__(
        self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0.0, attn_p=0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=hidden_features, out_features=dim
        )

    def forward(self, x):
        """Run forward pass.

        Args:
            x: A Tensor of shape (n_samples, n_patches + 1, dim)

        Returns:
            An output tensor of shape (n_samples, n_patches + 1, dim)
        """

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """
    Parameters:
        img_size: Height and the width of the image (square).
        patch_size: Height and width of the patch (square).
        in_channels: Number of input channels.
        n_classes: Number of classes.
        embed_dim: Dimensionality of the token/patch embeddings.
        depth: Number of blocks.
        n_heads: Number of attention heads.
        mlp_ratio: Determines the dimension of the MLP module.
        qkv_bias: If True then we include a bias to the query, key and value projections.  # noqa: E501
        p, attn_p: Dropout probability.

    Attributes:
        patch_embed: Instance of `PatchEmbed` layer.
        cls_token: Learnable parameter that will represent the first token in te sequence. It has `embed_dim` elements.  # noqa: E501
        pos_emb: Positional embedding of the cls token + all the patches. It has `(n_patches + 1) * embed_dim` elements.  # noqa: E501
        pos_drop: Dropout layer.
        blocks: List of `Block` modules.
        norm: Layer normalization.
    """

    def __init__(
        self,
        img_size=384,
        patch_size=16,
        in_channels=3,
        n_classes=1000,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        p=0.0,
        attn_p=0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """
        Parameters:
            x: Input tensor of shape `(n_samples, in_channels, img_size, img_size)`  # noqa: E501

        Returns:
            logits: torch.Tensor - Logits over all the classes - `(n_samples, n_classes)`  # noqa: E501
        """

        n_samples = x.shape[0]
        x = self.patch_embed(x)

        # (n_samples, 1, embed_dim)
        cls_token = self.cls_token.expand(n_samples, -1, -1)

        # (n_samples, 1 + n_patches, embed_dim)
        x = torch.cat((cls_token, x), dim=1)

        # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # just the CLS token
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)

        return x
