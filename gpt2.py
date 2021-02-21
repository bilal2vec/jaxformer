import argparse
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

        content_embedding = nn.Embed(self.vocab_size, self.d_model)
        embeddings = content_embedding(
            x) + nn.Embed(self.seq_len, self.d_model)(position_ids)
        x = nn.Dropout(0.1)(embeddings)

        for _ in range(self.n_layers):
            x = Block(self.seq_len, self.d_model,
                      self.n_heads)(x, mask, training)

        x = nn.LayerNorm()(x)
        x = content_embedding.attend(x)

        return x


def main():
    batch_size = 1
    seq_len = 128
    n_layers = 2
    vocab_size = 30000
    d_model = 768
    n_heads = 8

    d_k = d_model // n_heads

    tokenizer = Tokenizer.from_file('./tokenizer.json')
    with open('./data.txt', 'r') as f:
        text = f.read()
    tokenized = tokenizer.encode(text)
    batches = jnp.array(tokenized.ids)

    rng = jax.random.PRNGKey(42)

    variables = GPT2(seq_len, n_layers, vocab_size, d_model, n_heads).init(
        {'params': rng, 'dropout': rng}, batches[:128].reshape(1, 128), training=False)
    gpt2 = GPT2(seq_len, n_layers, vocab_size, d_model, n_heads)

    def loss_fn(variables, batch, rng):
        x = batch[:, :-1]
        y = batch[:, 1:]
        y = jax.nn.one_hot(y, vocab_size)

        y_hat = gpt2.apply(variables, x, training=True, rngs={'dropout': rng})

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

    optimizer = flax.optim.Adam(
        learning_rate=1e-4, beta1=0.5, beta2=0.9).create(variables)
    optimizer = flax.jax_utils.replicate(optimizer)

    rngs = jax.random.split(rng, num=jax.local_device_count())

    global_step = 0
    for _ in range(1):
        for i in tqdm(range(0, batches.shape[0] // ((seq_len+1)*batch_size))):
            batch = batches[i*(seq_len+1)*batch_size:(i+1)
                            * (seq_len+1)*batch_size].reshape(-1, seq_len+1)

            x = shard(batch)

            optimizer, loss, rngs = train_step(optimizer, x, rngs)

            if global_step % 100 == 0:
                loss = flax.jax_utils.unreplicate(loss)
                print(loss)

            global_step += 1

    optimizer = flax.jax_utils.unreplicate(optimizer)
    batch = flax.jax_utils.unreplicate(x)

    logits = gpt2.apply(
        optimizer.target, batches[:seq_len].reshape(1, seq_len), training=True, rngs={'dropout': rng})
    preds = jnp.argmax(nn.softmax(logits, axis=-1), axis=-1)

    print(tokenizer.decode(batches[:seq_len]))
    print("\n")
    print(tokenizer.decode(preds.reshape(-1)))


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

    main()
