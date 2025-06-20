import pytest

import torch
from gaia2_pytorch.gaia2 import Gaia2

@pytest.mark.parametrize('use_logit_norm_distr', (False, True))
def test_gaia2(
    use_logit_norm_distr
):
    model = Gaia2(
        dim_latent = 77,
        dim = 32,
        depth = 1,
        heads = 4,
        use_logit_norm_distr = use_logit_norm_distr
    )

    tokens = torch.randn(2, 8, 16, 16, 77)

    out = model(tokens, return_flow_loss = False)
    assert out.shape == tokens.shape

    loss = model(tokens)
    loss.backward()

    sampled = model.generate((8, 16, 16), batch_size = 2)
    assert sampled.shape == tokens.shape

def test_tokenizer():
    from gaia2_pytorch.gaia2 import VideoTokenizer

    video = torch.randn(1, 3, 10, 16, 16)

    tokenizer = VideoTokenizer()

    loss = tokenizer(video)
    loss.backward()