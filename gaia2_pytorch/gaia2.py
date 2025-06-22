from __future__ import annotations
from gaia2_pytorch.tensor_typing import Float, Int, Bool

from functools import partial
from itertools import zip_longest

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, tensor, is_tensor
from torch.nn import Module, ModuleList, Linear, Conv3d, Sequential
from torch.distributions import Normal, Categorical, kl_divergence

from torchdiffeq import odeint

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
# tl, hl, wl - time, height, width of latents feature map
# nc - sequence length of context tokens cross attended to
# dc - feature dimension of context tokens

# constants

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def first(arr):
    return arr[0]

def xnor(x, y):
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def repeat_batch_to(t, batch):
    if t.shape[0] >= batch:
        return t

    return repeat(t, 'b ... -> (b r) ...', r = batch // t.shape[0])

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

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
        dim_context = None,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()

        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_q = LinearNoBias(dim, dim_inner)

        dim_kv = default(dim_context, dim)
        self.to_kv = LinearNoBias(dim_kv, dim_inner * 2)

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

        q = self.to_q(tokens)

        q = q * self.scale

        k, v = self.to_kv(kv_tokens).chunk(2, dim = -1)

        q, k, v = tuple(self.split_heads(t) for t in (q, k, v))

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        if exists(context_mask):
            sim = einx.where('b j, b h i j,', context_mask, sim, max_neg_value(sim))

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)

        return self.to_out(out)

# feedforward

def FeedForward(dim, expansion_factor = 4.):
    dim_inner = int(dim * expansion_factor)

    return Sequential(
        Linear(dim, dim_inner),
        nn.GELU(),
        Linear(dim_inner, dim)
    )

# adaptive norms for time conditioning (and ada-ln-zero fo DiT)

class AdaRMSNorm(Module):
    def __init__(
        self,
        dim,
        dim_cond = None
    ):
        super().__init__()
        self.scale = dim ** 0.5
        dim_cond = default(dim_cond, dim)

        self.to_gamma = LinearNoBias(dim_cond, dim)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(
        self,
        x,
        *,
        cond
    ):
        normed = l2norm(x) * self.scale
        gamma = self.to_gamma(cond)
        return normed * (gamma + 1.)

class AdaLNZero(Module):
    def __init__(
        self,
        dim,
        dim_cond = None,
        init_bias_value = -2.
    ):
        super().__init__()
        dim_cond = default(dim_cond, dim)
        self.to_gamma = Linear(dim_cond, dim)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.constant_(self.to_gamma.bias, init_bias_value)

    def forward(
        self,
        x,
        *,
        cond
    ):
        gamma = self.to_gamma(cond).sigmoid()
        return x * gamma

# conditioning related

class PreNormConfig(Module):
    def __init__(
        self,
        fn: Module,
        *,
        dim
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.RMSNorm(dim)

    def forward(
        self,
        t,
        *args,
        **kwargs
    ):
        return self.fn(self.norm(t), *args, **kwargs)

class AdaNormConfig(Module):
    def __init__(
        self,
        fn: Module,
        *,
        dim,
        dim_cond = None
    ):
        super().__init__()
        dim_cond = default(dim_cond, dim)

        self.ada_norm = AdaRMSNorm(dim = dim, dim_cond = dim_cond)

        self.fn = fn

        self.ada_ln_zero = AdaLNZero(dim = dim, dim_cond = dim_cond)

    def forward(
        self,
        t,
        *args,
        cond,
        **kwargs
    ):
        cond = repeat_batch_to(cond, t.shape[0])
        cond = rearrange(cond, 'b d -> b 1 d')

        t = self.ada_norm(t, cond = cond)

        out = self.fn(t, *args, **kwargs)

        return self.ada_ln_zero(out, cond = cond)

# random projection fourier embedding

class RandomFourierEmbed(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        assert divisible_by(dim, 2)
        self.register_buffer('weights', torch.randn(dim // 2))

    def forward(self, x):
        freqs = einx.multiply('i, j -> i j', x, self.weights) * 2 * torch.pi
        fourier_embed, _ = pack((x, freqs.sin(), freqs.cos()), 'b *')
        return fourier_embed

# transformer

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 16,
        ff_expansion_factor = 4.,
        has_time_attn = True,
        cross_attend = False,
        dim_cross_attended_tokens = None,
        accept_cond = False
    ):
        super().__init__()

        space_layers = []
        time_layers = []
        cross_attn_layers = []

        attn_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head
        )

        ff_kwargs = dict(
            dim = dim,
            expansion_factor = ff_expansion_factor
        )

        # if using time conditioning, use the ada-rmsnorm + ada-rms-zero

        self.accept_cond = accept_cond

        if accept_cond:
            norm_config = partial(AdaNormConfig, dim = dim)
        else:
            norm_config = partial(PreNormConfig, dim = dim)

        # layers through depth

        for _ in range(depth):

            space_attn = Attention(**attn_kwargs)
            space_ff = FeedForward(**ff_kwargs)

            space_layers.append(ModuleList([
                norm_config(space_attn),
                norm_config(space_ff),
            ]))

            if has_time_attn:

                time_attn = Attention(**attn_kwargs)
                time_ff = FeedForward(**ff_kwargs)

                time_layers.append(ModuleList([
                    norm_config(time_attn),
                    norm_config(time_ff),
                ]))

            if cross_attend:
                dim_context = default(dim_cross_attended_tokens, dim)

                cross_attn = Attention(**attn_kwargs, dim_context = dim_context)
                cross_ff = FeedForward(**ff_kwargs)

                cross_attn_layers.append(ModuleList([
                    norm_config(cross_attn),
                    norm_config(cross_ff)
                ]))

        self.space_layers = ModuleList(space_layers)
        self.time_layers = ModuleList(time_layers)
        self.cross_attn_layers = ModuleList(cross_attn_layers)

        self.final_norm = nn.RMSNorm(dim)

    def forward(
        self,
        tokens: Float['b tl hl wl d'],
        context: Float['b nc dc'] | None = None,
        context_mask: Bool['b nc'] | None = None,
        cond: Float['b dim_cond'] | None = None
    ):
        batch = tokens.shape[0]
        assert xnor(exists(cond), self.accept_cond)

        block_kwargs = dict()

        if exists(cond):
            block_kwargs.update(cond = cond)

        tokens, inv_pack_space = pack_with_inverse(tokens, 'b t * d')

        for (
            space_attn,
            space_ff
        ), maybe_time_layer, maybe_cross_attn_layer in zip_longest(self.space_layers, self.time_layers, self.cross_attn_layers):

            # space attention

            tokens, inv_pack_batch = pack_with_inverse(tokens, '* n d')

            tokens = space_attn(tokens, **block_kwargs) + tokens
            tokens = space_ff(tokens, **block_kwargs) + tokens

            tokens = inv_pack_batch(tokens)

            if exists(maybe_time_layer):

                time_attn, time_ff = maybe_time_layer

                # time attention

                tokens = rearrange(tokens, 'b t n d -> b n t d')
                tokens, inv_pack_batch = pack_with_inverse(tokens, '* t d')

                tokens = time_attn(tokens, **block_kwargs) + tokens
                tokens = time_ff(tokens, **block_kwargs) + tokens

                tokens = inv_pack_batch(tokens)
                tokens = rearrange(tokens, 'b n t d -> b t n d')

            if exists(context):
                assert exists(maybe_cross_attn_layer), f'`cross_attend` must be set to True on Transformer to receive context'

                cross_attn, cross_ff = maybe_cross_attn_layer

                # maybe cross attention

                tokens, inv_time_space_pack = pack_with_inverse(tokens, 'b * d')

                tokens = cross_attn(tokens, context = context, context_mask = context_mask, **block_kwargs) + tokens
                tokens = cross_ff(tokens, **block_kwargs) + tokens

                tokens = inv_time_space_pack(tokens)


        tokens = inv_pack_space(tokens)

        tokens = self.final_norm(tokens)
        return tokens

# video tokenizer

class VideoTokenizer(Module):
    def __init__(
        self,
        *,
        channels = 3,
        dim = 512,
        dim_latent = 64, # they do a really small latent dimension, claims this brings about improvements
        eps = 1e-6,
        latent_loss_weight = 1.,
        dim_head = 64,
        heads = 16,
        enc_depth = 2,
        dec_depth = 2
    ):
        super().__init__()

        self.eps = eps

        # encoder

        self.to_encode_tokens = Conv3d(channels, dim, 3, padding = 1)

        self.encode_transformer = Transformer(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            depth = enc_depth,
            has_time_attn = False
        )

        # latents

        self.to_latents = LinearNoBias(dim, dim_latent * 2)

        self.latent_loss_weight = latent_loss_weight

        self.gaussian = Normal(0., 1.)

        # decoder

        self.to_decode_tokens = LinearNoBias(dim_latent, dim)

        self.decode_transformer = Transformer(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            depth = dec_depth
        )

        self.to_recon = Conv3d(dim, channels, 1, bias = False)

    def encode(
        self,
        video: Float['b c t h w'],
        return_sampled = False
    ):
        tokens = self.to_encode_tokens(video)

        tokens = rearrange(tokens, 'b d ... -> b ... d')

        tokens = self.encode_transformer(tokens)

        latents = self.to_latents(tokens)

        mean, log_var = rearrange(latents, 'b ... (mean_var d) -> mean_var b ... d', mean_var = 2)

        var = log_var.exp()

        if not return_sampled:
            return mean, var

        return torch.normal(mean, var.sqrt())

    def decode(
        self,
        latents: Float['b tl hl wl d']
    ):
        tokens = self.to_decode_tokens(latents)

        tokens = self.decode_transformer(tokens)

        tokens = rearrange(tokens, 'b ... d -> b d ...')

        recon = self.to_recon(tokens)

        return recon

    def forward(
        self,
        video: Float['b c t h w'],
        return_breakdown = False,
        return_recon_only = False
    ):

        orig_video = video

        latent_mean, latent_var = self.encode(video)

        latent_normal_distr = Normal(latent_mean, latent_var.sqrt())

        sampled_latents = latent_normal_distr.sample()

        recon = self.decode(sampled_latents)

        if return_recon_only:
            return recon

        recon_loss = F.mse_loss(orig_video, recon)

        latent_loss = kl_divergence(latent_normal_distr, self.gaussian).mean()

        breakdown = (recon_loss, latent_loss)

        total_loss = (
            recon_loss + 
            latent_loss * self.latent_loss_weight
        )

        if not return_breakdown:
            return total_loss

        return total_loss, breakdown

# the main model is just a flow matching transformer, with the same type of conditioning from DiT (diffusion transformer)
# the attention is factorized space / time

class Gaia2(Module):
    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        dim_latent = 64,
        dim = 512,
        *,
        depth = 24,
        heads = 16,
        dim_head = 64,
        dim_context = None,
        ff_expansion_factor = 4.,
        use_logit_norm_distr = True,
        logit_norm_distr = [
            (.8, (.5, 1.4)),
            (.2, (-3., 1.))
        ],
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
    ):
        super().__init__()

        self.dim_latent = dim_latent

        self.tokenizer = tokenizer

        self.to_tokens = Linear(dim_latent, dim)

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            ff_expansion_factor = ff_expansion_factor,
            dim_cross_attended_tokens = default(dim_context, dim),
            cross_attend = True,
            accept_cond = True
        )

        # time conditioning

        self.to_time_cond = nn.Sequential(
            RandomFourierEmbed(dim),
            Linear(dim + 1, dim),
            nn.SiLU(),
        )

        # flow related

        self.use_logit_norm_distr = use_logit_norm_distr

        # construct their bimodal normal distribution - they have a second mode to encourage learning ego-motions and object trajectories

        mode_probs = []
        normal_distrs = []

        for prob, (mean, std) in logit_norm_distr:
            mode_probs.append(prob)
            normal_distrs.append(tensor([mean, std]))

        mode_probs = tensor(mode_probs)
        assert mode_probs.sum().item() == 1.

        self.register_buffer('mode_distr',mode_probs, persistent = False)
        self.register_buffer('normal_mean_std', stack(normal_distrs), persistent = False)

        # transformer to predicted flow

        self.to_pred_flow = LinearNoBias(dim, dim_latent)

        # sampling

        self.odeint_fn = partial(odeint, **odeint_kwargs)

        self.register_buffer('dummy', tensor(0), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    @torch.no_grad()
    def generate(
        self,
        video_shape: tuple[int, int, int], # (time, height, width)
        batch_size = 1,
        steps = 16
    ) -> (
        Float['b tl hl wl d'] |
        Float['b c t h w']
    ):

        self.eval()

        def fn(step_times, denoised):

            pred_flow = self.forward(
                denoised,
                return_flow_loss = False,
                input_is_video = False
            )

            return pred_flow

        output_shape = (batch_size, *video_shape, self.dim_latent)

        noise = torch.randn(output_shape)
        times = torch.linspace(0, 1, steps, device = self.device)

        trajectory = self.odeint_fn(fn, noise, times)

        sampled_latents = trajectory[-1]

        # enforce zero mean unit variance

        sampled_latents = normalize(sampled_latents)

        if not exists(self.tokenizer):
            return sampled_latents

        video = self.tokenizer.decode(sampled_latents)
        return video

    def forward(
        self,
        video_or_latents: Float['b tl hl wl d'] | Float['b c t h w'],
        context: Float['b nc dc'] | None = None,
        context_mask: Bool['b nc'] | None = None,
        times: Float['b'] | Float[''] | None = None,
        input_is_video = None,
        return_flow_loss = True
    ):

        # if tokenizer is added, assume is video

        input_is_video = default(input_is_video, exists(self.tokenizer))

        if input_is_video:
            with torch.no_grad():
                self.tokenizer.eval()
                latents = self.tokenizer.encode(video_or_latents, return_sampled = True)
        else:
            latents = video_or_latents

        # shape and device

        batch, device = latents.shape[0], latents.device

        # normalize data to zero mean, unit variance

        latents = normalize(latents)

        # flow matching is easy
        # you just noise some random amount and store the flow as data - noise, then force it to predict that velocity

        if not exists(times):

            time_shape = (batch,)

            if self.use_logit_norm_distr:
                # sample from bimodal normal distribution - section 2.2.4

                expanded_normal_mean_std = repeat(self.normal_mean_std, '... -> b ...', b = batch)
                mean, std = expanded_normal_mean_std.unbind(dim = -1)
                all_sampled = torch.normal(mean, std)

                batch_arange = torch.arange(batch, device = device)[:, None]
                sel_normal_indices = Categorical(self.mode_distr).sample(time_shape)[:, None]

                sel_samples = all_sampled[batch_arange, sel_normal_indices]
                times = sel_samples.sigmoid()

                times = rearrange(times, 'b 1 -> b')

            else:
                # else uniform
                times = torch.rand(time_shape, device = device)

            noise = torch.randn_like(latents)

            flow = latents - noise

            padded_times = rearrange(times, 'b -> b 1 1 1 1')
            tokens = noise.lerp(latents, padded_times) # read as (noise * (1. - time) + data * time)

        # handle time conditioning

        if times.ndim == 0:
            times = repeat(times, '-> b', b = batch)

        cond = self.to_time_cond(times)

        # transformer

        tokens = self.to_tokens(latents)

        attended = self.transformer(
            tokens,
            cond = cond,
            context = context,
            context_mask = context_mask
        )

        # flow matching

        pred_flow = self.to_pred_flow(attended)

        if not return_flow_loss:
            return pred_flow

        return F.mse_loss(pred_flow, flow)
