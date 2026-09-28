"""Notebook-style fixed-point simulation; tensors and arithmetic remain floating point."""
import torch
from torch import nn


def fixed_point(tensor, word_bits=16, fractional_bits=8):
    if not 2 <= word_bits <= 32 or not 0 <= fractional_bits < word_bits:
        raise ValueError('Require 2 <= word_bits <= 32 and 0 <= fractional_bits < word_bits.')
    scale = 2 ** fractional_bits
    lower, upper = -(2 ** (word_bits - 1)), 2 ** (word_bits - 1) - 1
    return torch.clamp(torch.round(tensor * scale), lower, upper) / scale


def quantize_weights(model, word_bits=16, fractional_bits=8):
    """Quantize Linear/Conv2d weights and biases in-place, as in source notebooks."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.weight.copy_(fixed_point(module.weight, word_bits, fractional_bits))
                if module.bias is not None:
                    module.bias.copy_(fixed_point(module.bias, word_bits, fractional_bits))
    return model
