from __future__ import annotations
from gaia2_pytorch.tensor_typing import Float, Int, Bool

from functools import partial
from itertools import zip_longest

import torch
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch import nn, cat, stack, tensor, is_tensor
from torch.nn import Module, ModuleList, Linear, Conv3d, Sequential
from torch.distributions import Normal, Categorical, kl_divergence

from torchdiffeq import odeint

import torchvision
from torchvision.transforms import Resize

import torchvision.models as vision_models
VGG16_Weights = vision_models.VGG16_Weights

import einx
from einops import rearrange, repeat, reduce, pack, unpack, einsum
from einops.layers.torch import Rearrange, Reduce

from ema_pytorch import EMA

from hyper_connections import get_init_and_expand_reduce_stream_functions

from rotary_embedding_torch import RotaryEmbedding

# einstein notation

# b - batch
# h - attention heads
# n - sequence
# d - feature dimension
# t - time
# vh, vw - height width of feature map or video
# i, j - sequence (source, target)
# tl, vhl, vwl - time, height, width of latents feature map
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

def module_device(m: Module):
    return next(m.parameters()).device

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
        heads = 8,
        use_sdpa = True
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

        self.to_v_gates = nn.Sequential(
            LinearNoBias(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        )

        self.use_sdpa = use_sdpa

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

        # projections

        q = self.to_q(tokens)

        k, v = self.to_kv(kv_tokens).chunk(2, dim = -1)

        # split heads

        q, k, v = tuple(self.split_heads(t) for t in (q, k, v))

        if self.use_sdpa:

            if exists(context_mask):
                context_mask = rearrange(context_mask, 'b j -> b 1 1 j')

            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal = False,
                attn_mask = context_mask
            )
        else:
            q = q * self.scale

            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            if exists(context_mask):
                sim = einx.where('b j, b h i j,', context_mask, sim, max_neg_value(sim))

            attn = sim.softmax(dim = -1)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # alphafold-like gating, for attending to nothing
        # many paper corroborating the need for this by now

        out = out * self.to_v_gates(tokens)

        # merge heads

        out = self.merge_heads(out)

        return self.to_out(out)

# feedforward

class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return F.gelu(gates) * x

def FeedForward(dim, expansion_factor = 4.):
    # glu variant - https://arxiv.org/abs/2002.05202

    dim_inner = int(dim * expansion_factor * 2 / 3)

    return Sequential(
        Linear(dim, dim_inner * 2),
        GEGLU(),
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
        accept_cond = False,
        num_hyperconn_streams = 1,
        num_hyperconn_fracs = 4,    # Zhu et al. https://arxiv.org/abs/2503.14125
        num_register_tokens = 16    # Darcet et al. https://arxiv.org/abs/2309.16588 
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

        # prepare hyper connections

        init_hyperconn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_hyperconn_streams, num_fracs = num_hyperconn_fracs, dim = dim)

        def wrap_block(fn):
            return init_hyperconn(branch = norm_config(fn))

        # register tokens

        self.registers_space = nn.Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

        if has_time_attn:
            self.registers_time = nn.Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

        self.has_time_attn = has_time_attn

        # layers through depth

        for _ in range(depth):

            space_attn = Attention(**attn_kwargs)
            space_ff = FeedForward(**ff_kwargs)

            space_layers.append(ModuleList([
                wrap_block(space_attn),
                wrap_block(space_ff),
            ]))

            if has_time_attn:

                time_attn = Attention(**attn_kwargs)
                time_ff = FeedForward(**ff_kwargs)

                time_layers.append(ModuleList([
                    wrap_block(time_attn),
                    wrap_block(time_ff),
                ]))

            if cross_attend:
                dim_context = default(dim_cross_attended_tokens, dim)

                cross_attn = Attention(**attn_kwargs, dim_context = dim_context)
                cross_ff = FeedForward(**ff_kwargs)

                cross_attn_layers.append(ModuleList([
                    wrap_block(cross_attn),
                    wrap_block(cross_ff)
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
        cond: Float['b dim_cond'] | None = None,
        cross_attn_dropout: Bool['b'] | None = None
    ):
        batch, time, height, width, _ = tokens.shape
        assert xnor(exists(cond), self.accept_cond)

        block_kwargs = dict()

        if exists(cond):
            block_kwargs.update(cond = cond)

        tokens, inv_pack_space = pack_with_inverse(tokens, 'b t * d')

        # expand for hyper conns

        tokens = self.expand_streams(tokens)

        # register tokens

        registers_space = repeat(self.registers_space, 'n d -> b n d', b = batch * time)

        if self.has_time_attn:
            registers_time = repeat(self.registers_time, 'n d -> b n d', b = batch * height * width)

        # space / time attention layers

        for (
            space_attn,
            space_ff
        ), maybe_time_layer, maybe_cross_attn_layer in zip_longest(self.space_layers, self.time_layers, self.cross_attn_layers):

            # space attention

            tokens, inv_pack_batch = pack_with_inverse(tokens, '* n d')

            tokens, inv_pack_registers = pack_with_inverse((registers_space, tokens), 'b * d')

            tokens = space_attn(tokens, **block_kwargs)
            tokens = space_ff(tokens, **block_kwargs)

            registers_space, tokens = inv_pack_registers(tokens)

            tokens = inv_pack_batch(tokens)

            if exists(maybe_time_layer):

                time_attn, time_ff = maybe_time_layer

                # time attention

                tokens = rearrange(tokens, 'b t n d -> b n t d')
                tokens, inv_pack_batch = pack_with_inverse(tokens, '* t d')

                tokens, inv_pack_registers = pack_with_inverse((registers_time, tokens), 'b * d')

                tokens = time_attn(tokens, **block_kwargs)
                tokens = time_ff(tokens, **block_kwargs)

                registers_time, tokens = inv_pack_registers(tokens)

                tokens = inv_pack_batch(tokens)
                tokens = rearrange(tokens, 'b n t d -> b t n d')

            if exists(context):
                assert exists(maybe_cross_attn_layer), f'`cross_attend` must be set to True on Transformer to receive context'

                cross_attn, cross_ff = maybe_cross_attn_layer

                # maybe cross attention

                cross_attn_tokens, inv_time_space_pack = pack_with_inverse(tokens, 'b * d')

                cross_attn_tokens = cross_attn(cross_attn_tokens, context = context, context_mask = context_mask, **block_kwargs)
                cross_attn_tokens = cross_ff(cross_attn_tokens, **block_kwargs)

                cross_attn_tokens = inv_time_space_pack(cross_attn_tokens)

                if exists(cross_attn_dropout):
                    tokens = einx.where('b, b ..., b ...', cross_attn_dropout, tokens, cross_attn_tokens)
                else:
                    tokens = cross_attn_tokens

        tokens = inv_pack_space(tokens)

        # reduce hyper conn streams

        tokens = self.reduce_streams(tokens)

        # final norm

        tokens = self.final_norm(tokens)

        return tokens

# tokenizer modules

class LPIPSLoss(Module):
    def __init__(
        self,
        vgg: Module | None = None,
        vgg_input_image_shape = (224, 224),
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
        vgg_device = None
    ):
        super().__init__()

        self.resize = Resize(vgg_input_image_shape, antialias = True)

        if not exists(vgg):
            vgg = vision_models.vgg16(weights = vgg_weights)
            vgg.classifier = nn.Sequential(*vgg.classifier[:-2])

        if exists(vgg_device):
            vgg.to(vgg_device)

        self._vgg = [vgg]

    @property
    def vgg(self):
        return first(self._vgg)

    def forward(
        self,
        video,
        recon_video
    ):
        vgg, device, orig_device = self.vgg, video.device, module_device(self.vgg)
        vgg.to(device)

        video, recon_video = tuple(rearrange(t, 'b c t vh vw -> (b t) c vh vw') for t in (video, recon_video))

        video, recon_video = tuple(self.resize(t) for t in (video, recon_video))

        recon_vgg_embed, vgg_embed = map(vgg, (recon_video, video))

        vgg.to(orig_device)
        return F.mse_loss(vgg_embed, recon_vgg_embed)

# adversarial loss related
# should abstract this into a library at some point

def gradient_penalty(
    images,
    output,
    center = 0. # recent paper claims to have solved GAN stability issues with zero mean gp penalty "Gan is dead paper"
):
    gradients = first(torch_grad(
        outputs = output,
        inputs = images,
        grad_outputs = torch.ones_like(output),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    ))

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return ((gradients.norm(2, dim = 1) - center) ** 2).mean()

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

class ReconDiscriminator(Module):
    def __init__(
        self,
        *,
        dim,
        channels = 3,
        depth = 2,
        activation = nn.LeakyReLU(0.1)
    ):
        super().__init__()

        self.net = Sequential(
            Conv3d(channels, dim, 7, 3, padding = 1),
            activation,
            *[Sequential(
                Conv3d(dim, dim, 3, 2, 1),
                activation
            ) for _ in range(depth)],
            Conv3d(dim, 1, 1),
        )

    def forward(
        self,
        recon_videos: Float['b c t vh vw'],
        real_videos: Float['b c t vh vw'] | None = None,
        return_logits = False
    ):

        is_discr_loss = exists(real_videos)

        if is_discr_loss:
            videos, inverse_pack_fake_real = pack_with_inverse((recon_videos, real_videos), '* c t vh vw')
        else:
            videos = recon_videos

        logits = self.net(videos)

        if is_discr_loss:
            logits = inverse_pack_fake_real(logits)
            fake_logits, real_logits = logits
            loss = hinge_discr_loss(fake_logits, real_logits)
        else:
            loss = hinge_gen_loss(logits)

        if not return_logits:
            return loss

        return loss, logits

# residual down / up sampling - as proposed in https://arxiv.org/abs/2410.10733v1
# section 3.2

class ResidualDownsample(Module):
    def __init__(
        self,
        dim,
        *,
        space = False,
        time = False,
        channel_reduce_factor = None
    ):
        super().__init__()
        assert space or time

        space_factor = 2 if space else 1
        time_factor = 2 if time else 1
        channel_reduce_factor = default(channel_reduce_factor, space_factor)

        self.channel_reduce_factor = channel_reduce_factor

        self.to_residual = Rearrange(
            'b (t ft) (h fsh) (w fsw) d -> b t h w (ft fsh fsw d)',
            ft = time_factor,
            fsh = space_factor,
            fsw = space_factor,
        )

        dim_in = dim * (time_factor * space_factor ** 2)
        dim_out = dim_in // channel_reduce_factor

        self.proj = nn.Linear(dim_in, dim_out)

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        feats: Float['b vt vhl vwl d']
    ):
        residual = self.to_residual(feats)
        channel_reduced_residual = reduce(residual, '... (r d) -> ... d', 'mean', r = self.channel_reduce_factor)
        return channel_reduced_residual + self.proj(residual)

class ResidualUpsample(Module):
    def __init__(
        self,
        dim,
        *,
        space = False,
        time = False,
        repeat_channels = None
    ):
        super().__init__()
        assert space or time
        space_factor = 2 if space else 1
        time_factor = 2 if time else 1
        repeat_factor = default(repeat_channels, space_factor)

        self.repeat_factor = repeat_factor

        self.to_residual = Rearrange('b t h w (ft fsh fsw d) -> b (t ft) (h fsh) (w fsw) d',
            ft = time_factor,
            fsh = space_factor,
            fsw = space_factor,
        )

        dim_out = dim * repeat_factor

        self.proj = Linear(dim, dim_out)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        feats: Float['b tl vhl vwl d']
    ):
        residual = self.to_residual(feats)
        channel_expanded_residual = repeat(residual, '... d -> ... (r d)', r = self.repeat_factor)

        learned_residual = self.proj(feats)
        learned_residual = self.to_residual(learned_residual)

        return channel_expanded_residual + learned_residual

# video tokenizer

class VideoTokenizer(Module):
    def __init__(
        self,
        *,
        channels = 3,
        dim = 512,
        dim_latent = 64, # they do a really small latent dimension, claims this brings about improvements
        eps = 1e-6,
        dim_head = 64,
        heads = 16,
        enc_depth = 2,
        dec_depth = 2,
        enc_transformer_kwargs: dict = dict(),
        dec_transformer_kwargs: dict = dict(),
        lpips_loss_kwargs: dict = dict(),
        latent_loss_weight = 1.,
        lpips_loss_weight = 1.,
        adversarial_gen_loss_weight = 1.
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
            has_time_attn = False,
            **enc_transformer_kwargs
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
            depth = dec_depth,
            **dec_transformer_kwargs
        )

        self.to_recon = Conv3d(dim, channels, 1, bias = False)

        # loss related

        self.lpips = LPIPSLoss()

        self.lpips_loss_weight = lpips_loss_weight

        self.adversarial_gen_loss_weight = adversarial_gen_loss_weight

        self.register_buffer('zero', tensor(0.), persistent = False)

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
        recon_discr: ReconDiscriminator | Module | None = None,
        return_discr_loss = False,
        return_breakdown = False,
        return_recon_only = False,
        apply_grad_penalty = True
    ):

        orig_video = video

        latent_mean, latent_var = self.encode(video)

        latent_normal_distr = Normal(latent_mean, latent_var.sqrt())

        sampled_latents = latent_normal_distr.sample()

        recon = self.decode(sampled_latents)

        if return_discr_loss:
            assert exists(recon_discr)

            recon = recon.detach().clone()

            # maybe gradient penalty

            if apply_grad_penalty:
                recon.requires_grad_()
                orig_video.requires_grad_()

            # discriminator loss

            adv_discr_loss, (fake_logits, real_logits) = recon_discr(recon, orig_video, return_logits = True)

            gp_loss = self.zero

            if apply_grad_penalty:
                gp_loss = gradient_penalty(recon, fake_logits) + gradient_penalty(orig_video, real_logits)

            breakdown = (adv_discr_loss, gp_loss)

            total_loss = (
                adv_discr_loss +
                gp_loss
            )

            if not return_breakdown:
                return total_loss

            return total_loss, breakdown

        if return_recon_only:
            return recon

        recon_loss = F.mse_loss(orig_video, recon)

        lpips_loss = self.lpips(orig_video, recon)

        latent_loss = kl_divergence(latent_normal_distr, self.gaussian).mean()

        adv_gen_loss = self.zero

        if exists(recon_discr):
            adv_gen_loss = recon_discr(recon)

        breakdown = (recon_loss, lpips_loss, latent_loss, adv_gen_loss)

        total_loss = (
            recon_loss + 
            latent_loss * self.latent_loss_weight +
            lpips_loss * self.lpips_loss_weight + 
            adv_gen_loss * self.adversarial_gen_loss_weight
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
        context_dropout_prob = 0.25, # for classifier free guidance, Ho et al.
        logit_norm_distr = [
            (.8, (.5, 1.4)),
            (.2, (-3., 1.))
        ],
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        transformer_kwargs: dict = dict()
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
            accept_cond = True,
            **transformer_kwargs
        )

        # time conditioning

        self.to_time_cond = nn.Sequential(
            RandomFourierEmbed(dim),
            Linear(dim + 1, dim),
            nn.SiLU(),
        )

        # action / conditioning - classifier free guidance
        # consider the newer improved cfg paper if circumstances allows

        self.context_dropout_prob = context_dropout_prob

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
        steps = 16,
        cond_scale = 3.
    ) -> (
        Float['b tl hl wl d'] |
        Float['b c t h w']
    ):

        self.eval()

        def fn(step_times, denoised):

            pred_flow = self.forward_cfg(
                denoised,
                times = step_times,
                cond_scale = cond_scale,
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

    @torch.no_grad()
    def forward_cfg(
        self,
        *args,
        return_flow_loss = False,
        cond_scale = 3.,
        **kwargs,
    ):
        assert not return_flow_loss

        pred_flow = self.forward(*args, **kwargs, cross_attn_dropout = False, return_flow_loss = False)

        if cond_scale == 1.:
            return pred_flow

        null_pred_flow = self.forward(*args, **kwargs, cross_attn_dropout = True, return_flow_loss = False)

        update = pred_flow - null_pred_flow
        return pred_flow + update * (cond_scale - 1.)

    def forward(
        self,
        video_or_latents: Float['b tl vhl vwl d'] | Float['b c t vh vw'],
        context: Float['b nc dc'] | None = None,
        context_mask: Bool['b nc'] | None = None,
        times: Float['b'] | Float[''] | None = None,
        input_is_video = None,
        return_flow_loss = True,
        cross_attn_dropout: bool | Bool['b'] | None = None
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

        # classifier free guidance

        if isinstance(cross_attn_dropout, bool):
            cross_attn_dropout = tensor(cross_attn_dropout, device = device)
            cross_attn_dropout = repeat(cross_attn_dropout, ' -> b ', b = batch)

        if not exists(cross_attn_dropout):
            cross_attn_dropout = torch.rand(batch, device = device) < self.context_dropout_prob

        # transformer

        tokens = self.to_tokens(latents)

        attended = self.transformer(
            tokens,
            cond = cond,
            context = context,
            context_mask = context_mask,
            cross_attn_dropout = cross_attn_dropout
        )

        # flow matching

        pred_flow = self.to_pred_flow(attended)

        if not return_flow_loss:
            return pred_flow

        return F.mse_loss(pred_flow, flow)
