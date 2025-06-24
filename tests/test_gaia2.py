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
        dim_context = 55,
        use_logit_norm_distr = use_logit_norm_distr
    )

    tokens = torch.randn(2, 8, 16, 16, 77)

    context = torch.randn(2, 32, 55)
    context_mask = torch.randint(1, 2, (2, 32)).bool()

    context_kwargs = dict(context = context, context_mask = context_mask)

    out = model(tokens, **context_kwargs, return_flow_loss = False)
    assert out.shape == tokens.shape

    loss = model(tokens, **context_kwargs)
    loss.backward()

    sampled = model.generate((8, 16, 16), batch_size = 2)
    assert sampled.shape == tokens.shape

@pytest.mark.parametrize('apply_grad_penalty', (False, True))
def test_tokenizer(
    apply_grad_penalty
):
    from gaia2_pytorch.gaia2 import VideoTokenizer

    video = torch.randn(1, 3, 10, 16, 16)

    tokenizer = VideoTokenizer()

    loss = tokenizer(video, apply_grad_penalty = apply_grad_penalty)
    loss.backward()

def test_optional_adversarial_loss():
    from gaia2_pytorch.gaia2 import VideoTokenizer, ReconDiscriminator

    video = torch.randn(1, 3, 10, 16, 16)

    tokenizer = VideoTokenizer(enc_depth = 1, dec_depth = 1)

    discr = ReconDiscriminator(dim = 32, depth = 2)

    discr_loss = tokenizer(video, recon_discr = discr, return_discr_loss = True)
    discr_loss.backward()

    loss = tokenizer(video, recon_discr = discr)
    loss.backward()

    loss = tokenizer(video)
    loss.backward()
