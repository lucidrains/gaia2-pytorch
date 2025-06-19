import pytest

import torch
from gaia2_pytorch.gaia2 import Gaia2

def test_gaia2():
    model = Gaia2(
        dim = 512,
        depth = 24,
        heads = 16
    )

    tokens = torch.randn(2, 8, 16, 16, 512)

    out = model(tokens)
    assert out.shape == tokens.shape
