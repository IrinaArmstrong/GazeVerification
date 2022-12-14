import torch
import numpy as np


def init_linear(module: torch.nn.Module):
    """
    Initialize linear transformation.
    Fills the input Tensor with values according to the method described in:
    Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010),
    using a uniform distribution.
    """
    bias = np.sqrt(6.0 / (module.weight.size(0) + module.weight.size(1)))
    torch.nn.init.uniform(module.weight, -bias, bias)
    if module.bias is not None:
        module.bias.data.zero_()


def init_embeddings(module: torch.nn.Module):
    """
    Initialize embeddings layer.
    Fills the input Tensor with values drawn from the normal distribution (mu = 0, std = 0).
    """
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


def init_layernorm(module: torch.nn.Module):
    """
    Initialize Layer Normalization layer.
    Fills the input Tensor with values drawn from the normal distribution (mu = 0, std = 0).
    """
    if isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


