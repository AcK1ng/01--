# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :]
        .expand(slen, n_kv_heads, n_rep, head_dim)
        .reshape(slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = torch.nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = torch.nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = torch.nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = torch.nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

    def forward(
        self,
        kv: torch.Tensor, # (2, max_seq_len, kv_head, head_dim)
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        cache_k = kv[0]
        cache_v = kv[1]

        bsz, seqlen, _ = x.shape
        assert bsz == 1
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        cache_k[start_pos : start_pos + seqlen] = xk
        cache_v[start_pos : start_pos + seqlen] = xv

        keys = cache_k[: start_pos + seqlen]
        values = cache_v[: start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (cache_len + seqlen, n_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (cache_len + seqlen, n_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(0, 1)  # (n_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            0, 1
        )  # (n_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(1, 2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (n_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = torch.nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = torch.nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = torch.nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        kv: torch.Tensor,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(kv, self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = torch.nn.Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = torch.nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, kvcache: Tuple[torch.Tensor], tokens: torch.Tensor, start_pos: int):
        # tokens = tokens.unsqueeze(0)
        _bsz, seqlen = tokens.shape
        assert _bsz == 1
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = torch.full((seqlen, seqlen), float("-inf"))

        mask = torch.triu(mask, diagonal=1)

        # When performing key-value caching, we compute the attention scores
        # only for the new sequence. Thus, the matrix of scores is of size
        # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        # j > cache_len + i, since row i corresponds to token cache_len + i.
        mask = torch.hstack(
            [torch.zeros((seqlen, start_pos)), mask]
        ).type_as(h)

        for layer_idx, layer in enumerate(self.layers):
            h = layer(kvcache[layer_idx], h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output

class TransformerWrapper(nn.Module):
    def __init__(self, 
                 temperature: float,
                 top_p: float,
                 params: ModelArgs):
        super().__init__()
        self.temperature = temperature
        self.top_p = top_p
        self.model = Transformer(params)

    def forward(self,
                kvs_l0,
                kvs_l1,
                kvs_l2,
                kvs_l3,
                kvs_l4,
                kvs_l5,
                kvs_l6,
                kvs_l7,
                kvs_l8,
                kvs_l9,
                kvs_l10,
                kvs_l11,
                kvs_l12,
                kvs_l13,
                kvs_l14,
                kvs_l15,
                kvs_l16,
                kvs_l17,
                kvs_l18,
                kvs_l19,
                kvs_l20,
                kvs_l21,
                kvs_l22,
                kvs_l23,
                kvs_l24,
                kvs_l25,
                kvs_l26,
                kvs_l27,
                kvs_l28,
                kvs_l29,
                kvs_l30,
                kvs_l31,
                tokens: torch.Tensor,
                dummy_tensor__for__start_pos: torch.Tensor, # start_pos: int,
                ):
        
        kvcaches = (kvs_l0,
                    kvs_l1,
                    kvs_l2,
                    kvs_l3,
                    kvs_l4,
                    kvs_l5,
                    kvs_l6,
                    kvs_l7,
                    kvs_l8,
                    kvs_l9,
                    kvs_l10,
                    kvs_l11,
                    kvs_l12,
                    kvs_l13,
                    kvs_l14,
                    kvs_l15,
                    kvs_l16,
                    kvs_l17,
                    kvs_l18,
                    kvs_l19,
                    kvs_l20,
                    kvs_l21,
                    kvs_l22,
                    kvs_l23,
                    kvs_l24,
                    kvs_l25,
                    kvs_l26,
                    kvs_l27,
                    kvs_l28,
                    kvs_l29,
                    kvs_l30,
                    kvs_l31)
        
        local_hit = dummy_tensor__for__start_pos.shape[0]
        miss_len = tokens.shape[1]

        
        logits = self.model.forward(kvcaches, tokens, local_hit)
        probs = torch.softmax(logits[:, -1] / self.temperature, dim=-1)
        next_token = self.sample_top_p(probs, self.top_p)
        return next_token, torch.empty((miss_len + local_hit,))
    

    
    def sample_top_p(self, probs, p):
        """
        Perform top-p (nucleus) sampling on a probability distribution.
    
        Args:
            probs (torch.Tensor): Probability distribution tensor.
            p (float): Probability threshold for top-p sampling.
    
        Returns:
            torch.Tensor: Sampled token indices.
    
        Note:
            Top-p sampling selects the smallest set of tokens whose cumulative probability mass
            exceeds the threshold p. The distribution is renormalized based on the selected tokens.
        """
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token
    