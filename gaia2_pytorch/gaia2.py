from __future__ import annotations
from gaia2_pytorch.tensor_typing import Float, Int, Bool

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, tensor, is_tensor
from torch.nn import Module, ModuleList, Linear, Sequential
from torch.distributions import Normal, Categorical

import einx
from einops import rearrange, repeat, pack, unpack, einsum
from einops.layers.torch import Rearrange

from ema_pytorch import EMA

# einstein notation

# b - batch
# n - sequence
# d - feature dimension
# t - time
# h, w - height width of feature map or video
# i, j - sequence (source, target)

# constants

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def first(arr):
    return arr[0]

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def normalize(t, eps = 1e-6):
    shape = t.shape[-1:]
    return F.layer_norm(t, shape, eps = eps)

def pack_with_inverse(t, pattern):
    pack_one = is_tensor(t)

    if pack_one:
        t = [t]

    packed, shapes = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        out = unpack(out, shapes, inv_pattern)

        if pack_one:
            out = first(out)

        return out

    return packed, inverse

# action transforms

def symlog(value, value_max, scale):
    # symmetric logarithmic transformation (5)
    return value.sign() * log(1 + scale * value.abs()) / log(1 + scale * value_max.abs())

def curvature_symlog(value, value_max, scale = 1000): # m^-1 (.0001 - .1)
    return symlog(value, value_max, scale)

def speed_symlog(value, value_max, scale = 3.6): # m/s (0-75)
    return symlog(value, value_max, scale)

# attention, still the essential ingredient

class Attention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()

        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.norm = nn.RMSNorm(dim)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_q = LinearNoBias(dim, dim_inner)
        self.to_kv = LinearNoBias(dim, dim_inner * 2)
        self.to_out = LinearNoBias(dim_inner, dim)

    def forward(
        self,
        tokens: Float['b i d'],
        context: Float['b j d'] | None = None,
        context_mask: Bool['b j'] | None = None
    ):
        """
        q - queries
        k - keys
        v - values
        """

        kv_tokens = default(context, tokens)

        tokens = self.norm(tokens)

        q = self.to_q(tokens)

        k, v = self.to_kv(kv_tokens).chunk(2, dim = -1)

        q, k, v = tuple(self.split_heads(t) for t in (q, k, v))

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)

        return self.to_out(out)

# feedforward

def FeedForward(dim, expansion_factor = 4.):
    dim_inner = int(dim * expansion_factor)

    return Sequential(
        nn.RMSNorm(dim),
        Linear(dim, dim_inner),
        nn.GELU(),
        Linear(dim_inner, dim)
    )

# the main model is just a flow matching transformer, with the same type of conditioning from DiT (diffusion transformer)
# the attention is factorized space / time

class Gaia2(Module):
    def __init__(
        self,
        dim_input,
        dim = 512,
        *,
        depth = 24,
        heads = 16,
        dim_head = 64,
        ff_expansion_factor = 4.,
        use_logit_norm_distr = True,
        logit_norm_distr = [
            (.8, (.5, 1.4)),
            (.2, (-3., 1.))
        ]
    ):
        super().__init__()

        self.to_tokens = Linear(dim_input, dim)

        layers = []

        attn_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head
        )

        ff_kwargs = dict(
            dim = dim,
            expansion_factor = ff_expansion_factor
        )

        for _ in range(depth):

            space_attn = Attention(**attn_kwargs)
            time_attn = Attention(**attn_kwargs)

            space_ff = FeedForward(**ff_kwargs)
            time_ff = FeedForward(**ff_kwargs)

            layers.append(ModuleList([
                space_attn,
                space_ff,
                time_attn,
                time_ff
            ]))

        self.layers = ModuleList(layers)

        self.final_norm = nn.RMSNorm(dim)

        # flow related

        self.use_logit_norm_distr = use_logit_norm_distr

        # construct their bimodal normal distribution - they have a second mode to encourage learning ego-motions and object trajectories

        mode_probs = []
        normal_distrs = []

        for prob, (mean, std) in logit_norm_distr:
            mode_probs.append(prob)
            normal_distrs.append(tensor([mean, std]))

        self.register_buffer('mode_distr',tensor(mode_probs), persistent = False)
        self.register_buffer('normal_mean_std', stack(normal_distrs), persistent = False)

        # transformer to predicted flow

        self.to_pred_flow = LinearNoBias(dim, dim_input)

    def forward(
        self,
        data: Float['b t h w d'],
        return_flow_loss = False
    ):

        batch, device = data.shape[0], data.device

        # normalize data to zero mean, unit variance

        data = normalize(data)

        # flow matching is easy
        # you just noise some random amount and store the flow as data - noise, then force it to predict that velocity

        if return_flow_loss:
            time_shape = (batch,)

            if self.use_logit_norm_distr:
                # sample from bimodal normal distribution - section 2.2.4

                expanded_normal_mean_std = repeat(self.normal_mean_std, '... -> b ...', b = batch)
                mean, std = expanded_normal_mean_std.unbind(dim = -1)
                all_sampled_times = torch.normal(mean, std)

                batch_arange = torch.arange(batch, device = device)[:, None]
                sel_normal_indices = Categorical(self.mode_distr).sample(time_shape)[:, None]

                times = all_sampled_times[batch_arange, sel_normal_indices]
                times = rearrange(times, 'b 1 -> b')

            else:
                # else uniform
                times = torch.rand(time_shape, device = device)

            noise = torch.randn_like(data)

            flow = data - noise

            times = rearrange(times, 'b -> b 1 1 1 1')
            tokens = noise.lerp(data, times) # read as (noise * (1. - time) + data * time)

        # transformer

        tokens = self.to_tokens(data)

        tokens, inv_pack_space = pack_with_inverse(tokens, 'b t * d')

        for (
            space_attn,
            space_ff,
            time_attn,
            time_ff
        ) in self.layers:

            # space attention

            tokens, inv_pack_batch = pack_with_inverse(tokens, '* n d')

            tokens = space_attn(tokens) + tokens
            tokens = space_ff(tokens) + tokens

            tokens = inv_pack_batch(tokens)

            # time attention

            tokens = rearrange(tokens, 'b t n d -> b n t d')
            tokens, inv_pack_batch = pack_with_inverse(tokens, '* t d')

            tokens = time_ff(tokens) + tokens
            tokens = time_ff(tokens) + tokens

            tokens = inv_pack_batch(tokens)
            tokens = rearrange(tokens, 'b n t d -> b t n d')

        tokens = inv_pack_space(tokens)

        tokens = self.final_norm(tokens)

        # flow matching

        pred_flow = self.to_pred_flow(tokens)

        if not return_flow_loss:
            return pred_flow

        return F.mse_loss(pred_flow, flow)

