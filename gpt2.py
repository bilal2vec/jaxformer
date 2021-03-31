import argparse
from dataclasses import dataclass
from functools import partial

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax.experimental.maps import xmap

import flax
from flax import linen as nn

from tokenizers import Tokenizer


@jax.custom_vjp
def f_psum(x):
    return x


def f_psum_fwd(x):
    return f_psum(x), None


def f_psum_bwd(_, g):
    return jax.lax.psum(g, "shard"),


@jax.custom_vjp
def g_psum(x):
    return jax.lax.psum(x, "shard")


def g_psum_fwd(x):
    return g_psum(x), None


def g_psum_bwd(_, g):
    return g,


f_psum.defvjp(f_psum_fwd, f_psum_bwd)
g_psum.defvjp(g_psum_fwd, g_psum_bwd)


class MultiHeadSelfAttention(nn.Module):
    config: dataclass

    @nn.compact
    def __call__(self, x, mask):
        batch, seq_len, d_model = x.shape

        d_model_mp = d_model // config.mp
        d_k = d_model_mp // config.n_heads

        x = f_psum(x)

        q = nn.Dense(d_model_mp)(x)
        k = nn.Dense(d_model_mp)(x)
        v = nn.Dense(d_model_mp)(x)

        # batch x n_heads x seq_len x d_k
        q = q.reshape((-1, seq_len, config.n_heads, d_k)
                      ).transpose((0, 2, 1, 3))
        k = k.reshape((-1, seq_len, config.n_heads, d_k)
                      ).transpose((0, 2, 1, 3))
        v = v.reshape((-1, seq_len, config.n_heads, d_k)
                      ).transpose((0, 2, 1, 3))

        a = jnp.matmul(q, k.transpose((0, 1, 3, 2))) / jnp.sqrt(d_k)

        mask = jnp.where(mask, 0, -jnp.inf)
        a += mask

        a = nn.softmax(a, axis=-1)
        a = jnp.matmul(a, v)

        # batch x seq_len x n_heads x d_k
        # batch x seq_len x d_model_mp
        a = a.transpose((0, 2, 1, 3)).reshape(-1, seq_len, d_model_mp)

        o = nn.Dense(d_model)(a)
        o = g_psum(o)

        return o


class MLP(nn.Module):
    config: dataclass

    @nn.compact
    def __call__(self, x):
        x = f_psum(x)

        x = nn.Dense((self.config.d_model // self.config.mp) * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.config.d_model)(x)

        x = g_psum(x)

        return x


class Block(nn.Module):
    config: dataclass

    @nn.compact
    def __call__(self, x, mask):

        x = x + MultiHeadSelfAttention(self.config)(nn.LayerNorm()(x), mask)
        x = x + MLP(self.config)(nn.LayerNorm()(x))

        return x


class GPT2(nn.Module):
    config: dataclass

    @nn.compact
    def __call__(self, x):
        seq_len = x.shape[-1]

        position_ids = jnp.arange(start=0, stop=seq_len, step=1)
        mask = jnp.triu(
            jnp.ones((1, seq_len, seq_len)), k=1) == 0

        content_embedding = nn.Embed(
            self.config.vocab_size, self.config.d_model)
        x = content_embedding(
            x) + nn.Embed(self.config.max_seq_len, self.config.d_model)(position_ids)

        for _ in range(self.config.n_layers):
            x = Block(config)(x, mask)

        x = nn.LayerNorm()(x)
        x = content_embedding.attend(x)

        return x


def main(config):
    tokenizer = Tokenizer.from_file('./tokenizer.json')
    with open('./wikitext-2-raw/wiki.train.raw', 'r') as f:
        text = f.read()
    tokenized = tokenizer.encode(text)
    batches = jnp.array(tokenized.ids)

    rng = jax.random.PRNGKey(42)

    @partial(xmap, in_axes=(["shard", ...], ["batch", ...]), out_axes=["shard", ...])
    def initialize(rng, init_batch):
        variables = GPT2(Config).init({'params': rng}, init_batch)
        optimizer = flax.optim.Adam(
            learning_rate=1e-4, beta1=0.5, beta2=0.9).create(variables)

        return optimizer

    def loss_fn(variables, batch):
        x = batch[:, :-1]
        y = batch[:, 1:]
        y = jax.nn.one_hot(y, config.vocab_size)

        y_hat = GPT2(config).apply(variables, x)

        loss = jnp.sum(y * jax.nn.log_softmax(y_hat, axis=-1), axis=-1)
        return -jnp.mean(loss)

    @partial(xmap, in_axes=(["shard", ...], ["batch", ...]), out_axes=(["shard", ...], ["batch", ...]))
    def train_step(optimizer, batch):
        loss, grad = jax.value_and_grad(loss_fn)(optimizer.target, batch)

        loss = jax.lax.pmean(loss, axis_name=['shard'])
        grad = jax.lax.pmean(grad, axis_name=['batch'])

        optimizer = optimizer.apply_gradient(grad)

        return optimizer, loss

    @partial(xmap, in_axes=(["shard", ...], ["batch", ...]), out_axes=(["batch", ...]))
    def eval_step(optimizer, batch):
        loss = loss_fn(optimizer.target, batch)
        loss = jax.lax.pmean(loss, axis_name='shard')

        return loss

    rngs = jax.random.split(rng, num=config.mp)
    init_batches = jnp.zeros((config.dp, 1, 128), dtype=int)

    optimizer = initialize(rngs, init_batches)

    global_step = 0
    for _ in range(config.epochs):
        for i in tqdm(range(0, batches.shape[0] // ((config.max_seq_len+1)*config.batch_size))):
            batch = batches[i*(config.max_seq_len+1)*config.batch_size:(i+1)
                            * (config.max_seq_len+1)*config.batch_size].reshape(-1, config.max_seq_len+1)
            batch = batch.reshape(config.dp, batch.shape[0], batch.shape[1])

            optimizer, loss = train_step(optimizer, batch)

            if global_step % 10 == 0:
                loss = flax.jax_utils.unreplicate(loss)
                print(loss)

            global_step += 1

            if args.fast:
                break

    with open('./wikitext-2-raw/wiki.test.raw', 'r') as f:
        text = f.read()
    tokenized = tokenizer.encode(text)
    test_batches = jnp.array(tokenized.ids)

    test_loss = 0
    for i in tqdm(range(0, test_batches.shape[0] // ((config.max_seq_len+1)*config.batch_size))):
        batch = test_batches[i*(config.max_seq_len+1)*config.batch_size:(i+1) *
                             (config.max_seq_len+1)*config.batch_size].reshape(-1, config.max_seq_len+1)
        batch = batch.reshape(config.dp, batch.shape[0], batch.shape[1])

        loss = eval_step(optimizer, batch)

        test_loss += loss

        if args.fast:
            break

    test_loss /= (i + 1)
    print(f'Test Loss: {test_loss}')
    print('\n')

    # optimizer = flax.jax_utils.unreplicate(optimizer)

    # generated = tokenizer.encode(' ').ids
    # for i in tqdm(range(config.sample_len)):
    #     rng, _ = jax.random.split(rng, 2)

    #     x = jnp.array(generated).reshape(1, -1)
    #     logits = GPT2(config).apply(
    #         optimizer.target, x, training=False, rngs={'dropout': rng})
    #     next_token = jax.random.categorical(rng, logits[0, 0])
    #     generated += [int(next_token)]

    # print(f'Dataset: {tokenizer.decode(batches[:config.max_seq_len])}')
    # print("\n")
    # print(f'Continuation: {tokenizer.decode(generated)}')


@dataclass
class Config:
    fast: bool = False

    mp: int = 1
    dp: int = 1

    batch_size: int = 1
    epochs: int = 30
    sample_len: int = 16

    max_seq_len: int = 128
    n_layers: int = 2
    vocab_size: int = 32768
    d_model: int = 768
    n_heads: int = 8


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--fast', default=False, action="store_true")

    args = parser.parse_args()

    if args.debug:
        import ptvsd
        ptvsd.enable_attach(address=('localhost', 5678),
                            redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

    config = Config(fast=args.fast)
    main(config)
