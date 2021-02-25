from dataclasses import dataclass
from functools import partial

from tqdm import tqdm

import jax
import jax.numpy as jnp

import flax
from flax import linen as nn

from jax.experimental.maps import xmap


class MultiHeadSelfAttention(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x, mask, training):
        seq_len = x.shape[1]
        d_k = self.d_model // self.n_heads

        q = nn.Dense(self.d_model)(x)
        k = nn.Dense(self.d_model)(x)
        v = nn.Dense(self.d_model)(x)

        q = q.reshape((-1, seq_len, self.n_heads, d_k)).transpose((0, 2, 1, 3))
        k = k.reshape((-1, seq_len, self.n_heads, d_k)).transpose((0, 2, 1, 3))
        v = v.reshape((-1, seq_len, self.n_heads, d_k)).transpose((0, 2, 1, 3))

        a = jnp.matmul(q, k.transpose((0, 1, 3, 2))) / jnp.sqrt(d_k)

        mask = jnp.where(mask, 0, -jnp.inf)
        a += mask

        a = nn.softmax(a, axis=-1)
        a = nn.Dropout(0.1)(a, deterministic=not training)
        a = jnp.matmul(a, v)

        return a.transpose((0, 2, 1, 3)).reshape(-1, seq_len, self.d_model)


x = jnp.arange(10).reshape((2, 5))
y = xmap(jnp.vdot,
         in_axes=({0: 'left'}, {1: 'right'}),
         out_axes=['left', 'right', ...])(x, x.T)

print(y)
