# *_*coding:utf-8 *_*
"""
adapted from: https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
"""
import copy

from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

from models.subNets.position_embedding import PositionalEmbedding

# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = dict()

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result

    return cached_fn


def fourier_encode(x, max_freq, num_bands=4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


# Note: GEGLU() is different from that (i.e., GELU()) in mbt.py
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h) # (B*h, 1, T2)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, ff_expansion=4, attn_dropout=0., ff_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, FeedForward(dim, mult=ff_expansion, dropout=ff_dropout))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x


# half MPU (mutual promotion unit)
class CrossSelfTransformer(nn.Module):
    def __init__(self, latent_dim, input_dim, depth, heads, dim_head, ff_expansion=4, attn_dropout=0., ff_dropout=0.):
        """
        :param latent_dim: dim of target (query)
        :param input_dim:  dim of source/context (key/value)
        :param depth: number of layers
        :param heads: number of attention heads
        :param dim_head: dim of each head
        :param ff_expansion: expansion factor of feed-forward layer
        :param attn_dropout:
        :param ff_dropout:
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(latent_dim, Attention(latent_dim, context_dim=input_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                        context_dim=input_dim),
                PreNorm(latent_dim, Attention(latent_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(latent_dim, FeedForward(latent_dim, mult=ff_expansion, dropout=ff_dropout))
            ]))

    def forward(self, x, context, mask=None, context_mask=None):
        """
        :param x: latent array, (B, T1, D1)
        :param context: input array, (B, T2, D2)
        :param mask: padding mask, (B, T1)
        :param context_mask: padding mask for context, (B, T2)
        :return: (B, T1, D1)
        """
        for cross_attn, self_attn, ff in self.layers:
            x = cross_attn(x, context=context, mask=context_mask) + x
            x = self_attn(x, mask=mask) + x
            x = ff(x) + x
        return x


def _get_clones(module, N, share=True):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)]) if not share else nn.ModuleList([module for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "tanh":
        return nn.Tanh
    else:
        raise NotImplementedError

class NaiveAttention(nn.Module):
    def __init__(self, dim, activation_fn='relu'):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            _get_activation_fn(activation_fn)(),
            nn.Linear(dim, 1)
        )

    def forward(self, inputs):
        """
        :param inputs: (B, T, D)
        :return: (B, D)
        """
        scores = self.attention(inputs) # (B, T, 1)
        output = torch.matmul(torch.softmax(scores, dim=1).transpose(1,2), inputs).squeeze(1)
        return output


class EMT(nn.Module):
    def __init__(self, dim, depth, heads, num_modality, learnable_pos_emb=False,
                 emb_dropout=0., attn_dropout=0., ff_dropout=0., ff_expansion=4, max_seq_len=1024,
                 mpu_share=False, modality_share=False, layer_share=False, attn_act_fn='tanh'):
        super().__init__()

        assert dim % heads == 0, 'Error: hidden dim is not divisible by number of heads'
        dim_head = dim // heads

        self.num_modality = num_modality

        self.pos_embed = PositionalEmbedding(dim, max_seq_len=max_seq_len, dropout=emb_dropout,
                                             learnable=learnable_pos_emb)
        # level 0: MPU (mutual promotion unit) ancestor
        mpu_0 = CrossSelfTransformer(latent_dim=dim, input_dim=dim, depth=1, heads=heads, dim_head=dim_head,
                                           ff_expansion=ff_expansion, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
        # level 1: bi-direction (i.e., global multimodal context <-> local unimodal features)
        mpu_1 = _get_clones(mpu_0, 2, share=mpu_share)
        # level 2: for each modality
        mpu_2 = _get_clones(mpu_1, num_modality, share=modality_share)
        # level 3: final transformers
        self.mpus = _get_clones(mpu_2, depth, share=layer_share)

        # attention-base pooling: for aggregating global multimodal contexts interacted with different unimodal features
        attn_pool_0 = NaiveAttention(num_modality * dim, activation_fn=attn_act_fn)
        self.attn_pools = _get_clones(attn_pool_0, depth, share=layer_share)

    def forward(self, gmc_tokens, modality_inputs, modality_masks):
        """
        :param gmc_tokens: global multimodal context, (B, M, D), typically, M=3
        :param modality_inputs: local unimodal features, type: list
            [(B, T1, D), (B, T2, D), ...]
        :param modality_inputs: corresponding masks, type: list
            [(B, T1), (B, T2), ...]
        :return: promoted global multimodal context and local unimodal features
        """
        batch_size, _, _ = gmc_tokens.shape

        # add position embedding
        modality_inputs = [self.pos_embed(modality_input) for modality_input in modality_inputs]

        # fusion: global multimodal context interacts with local unimodal features
        for l_idx, layer in enumerate(self.mpus):
            gmc_tokens_list = []
            for m_idx, x in enumerate(modality_inputs):
                # local unimodal features <--- global multimodal context
                x_new = layer[m_idx][0](x, context=gmc_tokens, context_mask=None, mask=modality_masks[m_idx])
                # global mutlimodal context <--- local unimodal features
                gmc_tokens_new = layer[m_idx][1](gmc_tokens, context=x, context_mask=modality_masks[m_idx], mask=None)
                gmc_tokens_list.append(gmc_tokens_new)
                # update uni-modal representations
                modality_inputs[m_idx] = x_new
            # aggregating multiple global multimodal contexts via attention pooling
            gmc_tokens = self.attn_pools[l_idx](torch.stack(gmc_tokens_list, dim=1).view(batch_size, self.num_modality, -1))
            gmc_tokens = gmc_tokens.view(batch_size, self.num_modality, -1)

        return gmc_tokens, modality_inputs


if __name__ == "__main__":
    b = 4
    dim = 32
    num_modality =3
    model = EMT(dim=dim, depth=2, heads=4, num_modality=num_modality)
    print(f"model: {model}")
    # global multimodal context
    gmc_tokens = torch.randn(b, num_modality, dim)
    # local unimodal features
    modality_inputs = [
        torch.randn(b, 50, dim), 
        torch.randn(b, 500, dim), 
        torch.randn(b, 200, dim)
    ]
    modality_masks = [
        torch.randint(2, (b,50)).bool(), 
        torch.randint(2, (b,500)).bool(), 
        torch.randint(2, (b,200)).bool()
    ]
    gmc_tokens, modality_outputs = model(gmc_tokens, modality_inputs, modality_masks)
    print(gmc_tokens.shape)