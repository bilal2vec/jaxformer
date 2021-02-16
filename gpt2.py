import argparse

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp

import flax
from flax import linen as nn


class MultiHeadAttention(nn.Module):
    seq_len: int
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, q, k, v, mask=None):
        seq_len = q.shape[0]
        d_k = self.d_model // self.n_heads

        q = nn.Dense(self.d_model)(q)
        k = nn.Dense(self.d_model)(k)
        v = nn.Dense(self.d_model)(v)

        q = q.reshape((seq_len, self.n_heads, d_k)).transpose((1, 0, 2))
        k = k.reshape((seq_len, self.n_heads, d_k)).transpose((1, 0, 2))
        v = v.reshape((seq_len, self.n_heads, d_k)).transpose((1, 0, 2))

        a = jnp.matmul(q, k.transpose((0, 2, 1))) / jnp.sqrt(d_k)

        if mask is not None:
            mask = jnp.where(mask, 0, -jnp.inf)
            a += mask

        a = jnp.matmul(nn.softmax(a, axis=-1), v)

        return a.transpose((1, 0, 2)).reshape(self.seq_len, self.d_model)


def main(args):
    print(args)

    nn.Dense()


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
