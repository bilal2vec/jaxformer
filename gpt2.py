import argparse

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp

import flax
from flax import linen as nn


class MultiHeadSelfAttention(nn.Module):
    seq_len: int
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x, mask, training):
        seq_len = x.shape[0]
        d_k = self.d_model // self.n_heads

        q = nn.Dense(self.d_model)(x)
        k = nn.Dense(self.d_model)(x)
        v = nn.Dense(self.d_model)(x)

        q = q.reshape((seq_len, self.n_heads, d_k)).transpose((1, 0, 2))
        k = k.reshape((seq_len, self.n_heads, d_k)).transpose((1, 0, 2))
        v = v.reshape((seq_len, self.n_heads, d_k)).transpose((1, 0, 2))

        a = jnp.matmul(q, k.transpose((0, 2, 1))) / jnp.sqrt(d_k)

        mask = jnp.where(mask, 0, -jnp.inf)
        a += mask

        a = nn.softmax(a, axis=-1)
        a = nn.Dropout(0.1)(a, deterministic=not training)
        a = jnp.matmul(a, v)

        return a.transpose((1, 0, 2)).reshape(self.seq_len, self.d_model)


class MLP(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, x, training):
        x = nn.Dense(self.d_model * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model)(x)
        x = nn.Dropout(0.1)(x, deterministic=not training)

        return x


class Block(nn.Module):
    seq_len: int
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x, mask, training):

        x = x + MultiHeadSelfAttention(self.seq_len, self.d_model,
                                       self.n_heads)(nn.LayerNorm()(x), mask, training)
        x = x + nn.Dropout(0.1)(MLP(self.d_model)(nn.LayerNorm()
                                                  (x), training), deterministic=not training)

        return x


class GPT2(nn.Module):
    seq_len: int
    n_layers: int
    vocab_size: int
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x, training=False):
        position_ids = jnp.arange(start=0, stop=self.seq_len, step=1)
        mask = jnp.triu(jnp.ones((1, self.seq_len, self.seq_len)), k=1) == 0

        embeddings = nn.Embed(self.vocab_size, self.d_model)(
            x) + nn.Embed(self.seq_len, self.d_model)(position_ids)
        x = nn.Dropout(0.1)(embeddings)

        for _ in range(self.n_layers):
            x = Block(self.seq_len, self.d_model,
                      self.n_heads)(x, mask, training)

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.vocab_size)(x)

        return x


def main(args):
    seq_len = 128
    n_layers = 2
    vocab_size = 1024
    d_model = 768
    n_heads = 8
    d_k = d_model // n_heads

    rng = jax.random.PRNGKey(42)
    rng, dropout_rng = jax.random.split(rng)
    x = jnp.array(1)

    variables_gpt2 = GPT2(seq_len, n_layers, vocab_size, d_model, n_heads).init(
        {'params': rng, 'dropout': dropout_rng}, x, training=False)
    out = GPT2(seq_len, n_layers, vocab_size, d_model, n_heads).apply(
        variables_gpt2, x, training=False, rngs={'dropout': rng})
    print(out.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    if args.debug:
        import ptvsd
        ptvsd.enable_attach(address=('localhost', 5678),
                            redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

    main(args)
