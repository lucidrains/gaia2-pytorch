import pytest

import torch
from gaia2_pytorch.gaia2 import Gaia2

def test_gaia2():
    model = Gaia2(
        dim_input = 77,
        dim = 32,
        depth = 1,
        heads = 4
    )

    tokens = torch.randn(2, 8, 16, 16, 77)

    out = model(tokens)
    assert out.shape == tokens.shape

    loss = model(tokens, return_flow_loss = True)
    loss.backward()
