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

        return a.transpose((0, 2, 1, 3)).reshape(-1, self.seq_len, self.d_model)


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
    batch_size = 2
    seq_len = 128
    n_layers = 2
    vocab_size = 1024
    d_model = 768
    n_heads = 8

    d_k = d_model // n_heads

    rng = jax.random.PRNGKey(42)
    x = jax.random.randint(rng, (batch_size, seq_len), 0, 1000, jnp.int32)

    variables = GPT2(seq_len, n_layers, vocab_size, d_model, n_heads).init(
        {'params': rng, 'dropout': rng}, x, training=False)
    gpt2 = GPT2(seq_len, n_layers, vocab_size, d_model, n_heads)

    def loss_fn(variables, batch, rng):
        x = batch[:, :-1]
        y = batch[:, 1:]
        y = jax.nn.one_hot(y, vocab_size)

        y_hat = gpt2.apply(variables, x, training=True, rngs={'dropout': rng})

        loss = jnp.sum(y * jax.nn.log_softmax(y_hat, axis=-1), axis=-1)
        return -jnp.mean(loss)

    # @partial(jax.pmap, axis_name='batch')
    def train_step(optimizer, batch, rng):
        rng, rng_dropout = jax.random.split(rng)

        loss, grad = jax.value_and_grad(loss_fn)(
            optimizer.target, batch, rng_dropout)

        # loss = jax.lax.pmean(loss, axis_name='batch')
        # grad = jax.lax.pmean(grad, axis_name='batch')

        optimizer = optimizer.apply_gradient(grad)

        return optimizer, loss, rng

    optimizer = flax.optim.Adam(
        learning_rate=1e-4, beta1=0.5, beta2=0.9).create(variables)
    x = jax.random.randint(rng, (batch_size, seq_len + 1), 0, 1000, jnp.int32)

    for _ in tqdm(range(20)):
        optimizer, loss, rng = train_step(optimizer, x, rng)

    print(loss)

    logits = gpt2.apply(
        optimizer.target, x[:, :-1], training=True, rngs={'dropout': rng})
    preds = jnp.argmax(nn.softmax(logits, axis=-1), axis=-1)

    print(x[:, 1:])
    print("\n")
    print(preds)


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
