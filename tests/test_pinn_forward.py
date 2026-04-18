"""Minimal neural network sanity check on CPU."""

import torch
import torch.nn as nn


def test_pinn_forward_small_batch_cpu():
    from PINN import PINN

    in_params = ["a", "b"]
    out_params = ["x", "y"]
    model = PINN(
        input_params=in_params,
        output_params=out_params,
        hidden_layers=2,
        neurons_per_layer=[8, 8],
        activation=nn.ReLU,
        use_batch_norm=False,
        dropout_rate=None,
    ).cpu()
    x = torch.randn(4, len(in_params), dtype=torch.float32)
    y = model(x)
    assert y.shape == (4, len(out_params))
