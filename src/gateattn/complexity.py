"""Analytical dense-equivalent operation counts, not hardware measurements."""


def estimate_macs(token_counts, embed_dim, hidden_dim, image_size=224, patch_size=16, classes=1000):
    """Count patch projection, qkv/projection, attention products, MLP, and head.

    One multiply-accumulate is one MAC. Bias, norm, softmax, gating, and sorting
    costs are excluded. Tokenization runs on the full image, before pruning.
    """
    patches = (image_size // patch_size) ** 2
    projection = patches * (3 * patch_size ** 2) * embed_dim
    blocks = sum(4 * n * embed_dim ** 2 + 2 * n * n * embed_dim + 2 * n * embed_dim * hidden_dim for n in token_counts)
    return int(projection + blocks + embed_dim * classes)
