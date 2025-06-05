import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax
from typing import List, Tuple

class ConvFunc(eqx.Module):
    layers: List[eqx.Module]
    activation: callable
    height: int
    width: int
    mask: jax.Array

    def __init__(self, mask: jax.Array, channels: List[int], kernel_size: int, stride: int = 1, *, key, **kwargs):
        super().__init__(**kwargs)
        keys = jr.split(key, len(channels) - 1)
        self.height, self.width = mask.shape
        self.mask = mask
        self.layers = []
        for i in range(len(channels) - 1):
            conv_layer = eqx.nn.Conv2d(
                channels[i], channels[i + 1],
                kernel_size=kernel_size,
                stride=stride,
                padding='SAME',
                key=keys[i]
            )
            norm_layer = eqx.nn.GroupNorm(groups=1, channels=channels[i + 1])
            combined_layer = eqx.nn.Sequential([conv_layer, norm_layer])
            # self.layers.append(combined_layer)
            self.layers.append(conv_layer)
        self.activation = jnn.swish

    def __call__(self, t, yv, args):
        # un-flatten the image data
        y = yv.reshape(yv.shape[0], self.height, self.width)
        for layer in self.layers:
            # apply convolution only on valid data
            # y = layer(y * self.mask)
            y = layer(y)
            y = self.activation(y)
        y = y.reshape(y.shape[0], -1)  # flatten back to vector form
        return y

class NeuralODE(eqx.Module):
    func: ConvFunc
    mask: jax.Array
    valid: jax.Array

    def __init__(self, mask: jax.Array , channels: List[int], kernel_size: int, stride: int = 1, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mask = mask
        self.valid = mask.reshape(1, -1).nonzero()[1]
        self.func = ConvFunc(mask=mask, channels=channels, kernel_size=kernel_size, stride=stride,  key=key)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            saveat=diffrax.SaveAt(ts=[ts[-1]]),
        )
        return solution.ys
