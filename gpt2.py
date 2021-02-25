import argparse
from dataclasses import dataclass
from functools import partial

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp

import flax
from flax import linen as nn

from tokenizers import Tokenizer


def shard(xs):
    return jax.tree_map(
        lambda x: x.reshape((jax.device_count(), -1) + x.shape[1:]), xs)


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
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x, mask, training):

        x = x + MultiHeadSelfAttention(self.d_model,
                                       self.n_heads)(nn.LayerNorm()(x), mask, training)
        x = x + nn.Dropout(0.1)(MLP(self.d_model)(nn.LayerNorm()
                                                  (x), training), deterministic=not training)

        return x


class GPT2(nn.Module):
    config: dataclass

    @nn.compact
    def __call__(self, x, training=False):
        seq_len = x.shape[-1]

        position_ids = jnp.arange(start=0, stop=seq_len, step=1)
        mask = jnp.triu(
            jnp.ones((1, seq_len, seq_len)), k=1) == 0

        content_embedding = nn.Embed(
            self.config.vocab_size, self.config.d_model)
        embeddings = content_embedding(
            x) + nn.Embed(self.config.max_seq_len, self.config.d_model)(position_ids)
        x = nn.Dropout(0.1)(embeddings)

        for _ in range(self.config.n_layers):
            x = Block(self.config.d_model, self.config.n_heads)(
                x, mask, training)

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

    variables = GPT2(config).init(
        {'params': rng, 'dropout': rng}, batches[:128].reshape(1, 128), training=False)

    def loss_fn(variables, batch, rng):
        x = batch[:, :-1]
        y = batch[:, 1:]
        y = jax.nn.one_hot(y, config.vocab_size)

        y_hat = GPT2(config).apply(
            variables, x, training=True, rngs={'dropout': rng})

        loss = jnp.sum(y * jax.nn.log_softmax(y_hat, axis=-1), axis=-1)
        return -jnp.mean(loss)

    @partial(jax.pmap, axis_name='batch')
    def train_step(optimizer, batch, rng):
        rng, rng_dropout = jax.random.split(rng)

        loss, grad = jax.value_and_grad(loss_fn)(
            optimizer.target, batch, rng_dropout)

        loss = jax.lax.pmean(loss, axis_name='batch')
        grad = jax.lax.pmean(grad, axis_name='batch')

        optimizer = optimizer.apply_gradient(grad)

        return optimizer, loss, rng

    @partial(jax.pmap, axis_name='batch')
    def eval_step(optimizer, batch, rng):
        rng, rng_dropout = jax.random.split(rng)

        loss = loss_fn(optimizer.target, batch, rng_dropout)
        loss = jax.lax.pmean(loss, axis_name='batch')

        return loss, rng

    optimizer = flax.optim.Adam(
        learning_rate=1e-4, beta1=0.5, beta2=0.9).create(variables)
    optimizer = flax.jax_utils.replicate(optimizer)

    rngs = jax.random.split(rng, num=jax.local_device_count())

    global_step = 0
    for _ in range(config.epochs):
        for i in tqdm(range(0, batches.shape[0] // ((config.max_seq_len+1)*config.batch_size))):
            batch = batches[i*(config.max_seq_len+1)*config.batch_size:(i+1)
                            * (config.max_seq_len+1)*config.batch_size].reshape(-1, config.max_seq_len+1)
            x = shard(batch)

            optimizer, loss, rngs = train_step(optimizer, x, rngs)

            if global_step % 100 == 0:
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
        x = shard(batch)

        loss, rngs = eval_step(optimizer, x, rngs)
        loss = flax.jax_utils.unreplicate(loss)

        test_loss += loss

        if args.fast:
            break

    test_loss /= (i + 1)
    print(f'Test Loss: {test_loss}')
    print('\n')

    optimizer = flax.jax_utils.unreplicate(optimizer)

    generated = tokenizer.encode(' ').ids
    for i in tqdm(range(config.sample_len)):
        rng, _ = jax.random.split(rng, 2)

        x = jnp.array(generated).reshape(1, -1)
        logits = GPT2(config).apply(
            optimizer.target, x, training=False, rngs={'dropout': rng})
        next_token = jax.random.categorical(rng, logits[0, 0])
        generated += [int(next_token)]

    print(f'Dataset: {tokenizer.decode(batches[:config.max_seq_len])}')
    print("\n")
    print(f'Continuation: {tokenizer.decode(generated)}')


@dataclass
class Config:
    fast: bool = False

    batch_size: int = 256
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
