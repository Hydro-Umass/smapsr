import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
import diffrax

class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(in_size=data_size, out_size=data_size, width_size=width_size,
                              depth=depth, activation=jnn.tanh, key=key)

    def __call__(self, t, y, args):
        return self.mlp(y)

class NeuralODE(eqx.Module):
    func: Func
    upsampler: eqx.nn.ConvTranspose2d
    mask: jax.Array
    height: int
    width: int

    def __init__(self, data_size, width_size, depth, mask, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)
        self.mask = mask
        self.upsampler = eqx.nn.ConvTranspose2d(1, 1, 9, padding='SAME', key=key)
        self.height, self.width = mask.shape

    def __call__(self, ts, y):
        y0 = self.upsampler(y.reshape(-1, self.height, self.width)).ravel()
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            #stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=[ts[-1]]),
        )
        return solution.ys


# --------------------------------------------------------------------------------------------
# V1: one shared, stationary local operator (strategy section 4.5).
#
# Pixel count sets the *state* dimension, never the *parameter* dimension: every grid point obeys the
# same equation du/dt = F_theta[u](x; s, t), so the model is defined on any crop, grid or region.
# V0 above (dense per-tile MLP) is kept unchanged as the ablation.
# --------------------------------------------------------------------------------------------

def bilinear_kernel(stride):
    """2-D bilinear (triangle) kernel of a transposed convolution with integer `stride`.

    Size 2*stride - stride%2, centred so that coarse-cell centres coincide with the centres of their
    stride-wide fine blocks (the aligned grids of the cache: stride 3 for 9 km -> 3 km and 3 km -> 1 km)."""
    size = 2 * stride - stride % 2
    centre = stride - 1 if stride % 2 else stride - 0.5
    k = 1.0 - np.abs(np.arange(size) - centre) / stride
    return np.outer(k, k).astype(np.float32)


class Upsampler(eqx.Module):
    """U_phi: cascade of stride-s transposed convolutions, one per integer scale factor.

    The first layer maps the C_in input channels (data, validity mask, optional extras) to the 1-channel
    state; each layer starts as exact bilinear interpolation of channel 0 (other channels and biases
    start at zero), so an untrained V1 reproduces bilinear interpolation of the gap-filled input.
    strides=(3,) is 9 km -> 3 km, (3, 3) is 9 km -> 1 km."""
    layers: tuple
    strides: tuple = eqx.field(static=True)

    def __init__(self, in_channels, strides, *, key):
        keys = jr.split(key, len(strides))
        layers = []
        for n, (s, k) in enumerate(zip(strides, keys)):
            cin = in_channels if n == 0 else 1
            size = 2 * s - s % 2
            layer = eqx.nn.ConvTranspose2d(cin, 1, size, stride=s, padding=s // 2, key=k)
            w = np.zeros((1, cin, size, size), dtype=np.float32)
            w[0, 0] = bilinear_kernel(s)
            layer = eqx.tree_at(lambda l: (l.weight, l.bias), layer, (jnp.asarray(w), jnp.zeros_like(layer.bias)))
            layers.append(layer)
        self.layers = tuple(layers)
        self.strides = tuple(strides)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def scale_embedding(scale_km, n_freq):
    """Fourier features of log(scale): [log s, sin(2^k log s), cos(2^k log s)], k = 0..n_freq-1."""
    x = jnp.log(jnp.asarray(scale_km, dtype=jnp.float32))
    f = 2.0 ** jnp.arange(n_freq, dtype=jnp.float32)
    return jnp.concatenate([x[None], jnp.sin(f * x), jnp.cos(f * x)])


class ConvFunc(eqx.Module):
    """F_theta: small conv stack over a (1, H, W) state with GroupNorm, SiLU and FiLM scale/time conditioning.

    Parameters depend on width/depth only (not on H, W). The last conv and every FiLM layer start at zero,
    so the initial vector field is identically zero and u(1) = u(0). Zero padding at the crop edge."""
    convs: tuple
    norms: tuple
    films: tuple
    out: eqx.nn.Conv2d

    def __init__(self, width, depth, groups, cond_dim, kernel=3, *, key):
        keys = jr.split(key, 2 * depth + 1)
        cins = [1] + [width] * (depth - 1)
        self.convs = tuple(eqx.nn.Conv2d(c, width, kernel, padding="SAME", key=keys[i]) for i, c in enumerate(cins))
        self.norms = tuple(eqx.nn.GroupNorm(groups, width) for _ in range(depth))
        films = []
        for i in range(depth):
            lin = eqx.nn.Linear(cond_dim, 2 * width, key=keys[depth + i])
            films.append(eqx.tree_at(lambda l: (l.weight, l.bias), lin, (jnp.zeros_like(lin.weight), jnp.zeros_like(lin.bias))))
        self.films = tuple(films)
        out = eqx.nn.Conv2d(width, 1, kernel, padding="SAME", key=keys[-1])
        self.out = eqx.tree_at(lambda l: (l.weight, l.bias), out, (jnp.zeros_like(out.weight), jnp.zeros_like(out.bias)))

    def __call__(self, t, u, args):
        cond = jnp.concatenate([args["scale_emb"], jnp.asarray(t, dtype=jnp.float32)[None]])
        h = u
        for conv, norm, film in zip(self.convs, self.norms, self.films):
            h = norm(conv(h))
            gamma, beta = jnp.split(film(cond), 2)
            h = jnn.silu(h * (1.0 + gamma[:, None, None]) + beta[:, None, None])
        return self.out(h)


class NeuralODEV1(eqx.Module):
    """u(0) = U_phi(x_c);  du/dt = F_theta(u, t; e(s));  output u(1). No mask, no geometry in the model.

    __call__(x, scale_km): x is (C_in, h, w) on the coarse grid, scale_km the target pixel size (only enters
    through the Fourier embedding; one scale per run in stage 1). Returns the (1, H, W) fine field, H = h * prod(strides).
    Solver: Tsitouras 5/4, constant step 1/n_steps (about 6 * n_steps + 1 function evaluations)."""
    upsampler: Upsampler
    func: ConvFunc
    n_steps: int = eqx.field(static=True)
    n_freq: int = eqx.field(static=True)

    def __init__(self, in_channels=2, strides=(3,), width=32, depth=4, groups=8, n_freq=4, n_steps=10, *, key):
        k1, k2 = jr.split(key)
        self.upsampler = Upsampler(in_channels, strides, key=k1)
        self.func = ConvFunc(width, depth, groups, cond_dim=2 * n_freq + 2, key=k2)
        self.n_steps = n_steps
        self.n_freq = n_freq

    def initial_state(self, x):
        return self.upsampler(x)

    def __call__(self, x, scale_km):
        u0 = self.initial_state(x)
        args = dict(scale_emb=scale_embedding(scale_km, self.n_freq))
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func), diffrax.Tsit5(), t0=0.0, t1=1.0, dt0=1.0 / self.n_steps, y0=u0, args=args,
            stepsize_controller=diffrax.ConstantStepSize(), saveat=diffrax.SaveAt(t1=True),
        )
        return sol.ys[0]
