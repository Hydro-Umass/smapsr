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


class PixelNorm(eqx.Module):
    """Normalise the channels of every pixel separately (LayerNorm over channels, as in ConvNeXt), then a learned
    per-channel scale and offset. Pointwise, so unlike GroupNorm the output at a pixel does not depend on the spatial
    extent it is applied to: training crops, inference tiles and the full grid see the same operator."""
    weight: jax.Array
    bias: jax.Array
    eps: float = eqx.field(static=True)

    def __init__(self, channels, eps=1e-5):
        self.weight, self.bias, self.eps = jnp.ones(channels), jnp.zeros(channels), eps

    def __call__(self, h):
        mu = h.mean(0, keepdims=True)
        var = ((h - mu) ** 2).mean(0, keepdims=True)
        return (h - mu) / jnp.sqrt(var + self.eps) * self.weight[:, None, None] + self.bias[:, None, None]


def make_norm(norm, groups, width):
    """'group' (GroupNorm over channel groups and the whole extent), 'pixel' (PixelNorm) or 'none'."""
    if norm == "group":
        return eqx.nn.GroupNorm(groups, width)
    if norm == "pixel":
        return PixelNorm(width)
    if norm == "none":
        return eqx.nn.Identity()
    raise ValueError(f"unknown norm {norm!r}")


class ConvFunc(eqx.Module):
    """F_theta: small conv stack over a (1, H, W) state with normalisation (`norm`, see make_norm), SiLU and FiLM
    scale/time conditioning.

    Parameters depend on width/depth (and the number of static channels) only, not on H, W. The last conv and every
    FiLM layer start at zero, so the initial vector field is identically zero and u(1) = u(0). Zero padding at the
    crop edge. `static_channels` > 0: fixed fine-grid fields (e.g. the label climatology) passed as args["static"]
    (static_channels, H, W) are concatenated to the state before the first conv; they do not evolve."""
    convs: tuple
    norms: tuple
    films: tuple
    out: eqx.nn.Conv2d

    def __init__(self, width, depth, groups, cond_dim, kernel=3, static_channels=0, norm="group", *, key):
        keys = jr.split(key, 2 * depth + 1)
        cins = [1 + static_channels] + [width] * (depth - 1)
        self.convs = tuple(eqx.nn.Conv2d(c, width, kernel, padding="SAME", key=keys[i]) for i, c in enumerate(cins))
        self.norms = tuple(make_norm(norm, groups, width) for _ in range(depth))
        films = []
        for i in range(depth):
            lin = eqx.nn.Linear(cond_dim, 2 * width, key=keys[depth + i])
            films.append(eqx.tree_at(lambda l: (l.weight, l.bias), lin, (jnp.zeros_like(lin.weight), jnp.zeros_like(lin.bias))))
        self.films = tuple(films)
        out = eqx.nn.Conv2d(width, 1, kernel, padding="SAME", key=keys[-1])
        self.out = eqx.tree_at(lambda l: (l.weight, l.bias), out, (jnp.zeros_like(out.weight), jnp.zeros_like(out.bias)))

    def __call__(self, t, u, args):
        cond = jnp.concatenate([args["scale_emb"], jnp.asarray(t, dtype=jnp.float32)[None]])
        h = jnp.concatenate([u, args["static"]]) if "static" in args else u
        for conv, norm, film in zip(self.convs, self.norms, self.films):
            h = norm(conv(h))
            gamma, beta = jnp.split(film(cond), 2)
            h = jnn.silu(h * (1.0 + gamma[:, None, None]) + beta[:, None, None])
        return self.out(h)


class NeuralODEV1(eqx.Module):
    """u(0) = U_phi(x_c);  du/dt = F_theta(u, t; e(s));  output u(1). No mask, no geometry in the model.

    __call__(x, scale_km, static=None): x is (C_in, h, w) on the coarse grid, scale_km the target pixel size (only
    enters through the Fourier embedding; one scale per run in stage 1), static the (static_channels, H, W) fine-grid
    fields seen by F_theta (required iff static_channels > 0). Returns the (1, H, W) fine field, H = h * prod(strides).
    Solver: Tsitouras 5/4, constant step 1/n_steps (about 6 * n_steps + 1 function evaluations), or solver="euler":
    explicit Euler with n_steps evaluations, i.e. the same-f discrete ResNet u_{k+1} = u_k + f(u_k, t_k) / n_steps."""
    upsampler: Upsampler
    func: ConvFunc
    n_steps: int = eqx.field(static=True)
    n_freq: int = eqx.field(static=True)
    solver: str = eqx.field(static=True)

    def __init__(self, in_channels=2, strides=(3,), width=32, depth=4, groups=8, n_freq=4, n_steps=10, kernel=3,
                 static_channels=0, norm="group", solver="tsit5", *, key):
        k1, k2 = jr.split(key)
        self.upsampler = Upsampler(in_channels, strides, key=k1)
        self.func = ConvFunc(width, depth, groups, cond_dim=2 * n_freq + 2, kernel=kernel, static_channels=static_channels, norm=norm, key=k2)
        self.n_steps = n_steps
        self.n_freq = n_freq
        self.solver = solver

    def initial_state(self, x):
        return self.upsampler(x)

    def __call__(self, x, scale_km, static=None):
        u0 = self.initial_state(x)
        args = dict(scale_emb=scale_embedding(scale_km, self.n_freq))
        if static is not None and static.shape[0]:
            args["static"] = static
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func), dict(tsit5=diffrax.Tsit5(), euler=diffrax.Euler())[self.solver], t0=0.0, t1=1.0, dt0=1.0 / self.n_steps, y0=u0, args=args,
            stepsize_controller=diffrax.ConstantStepSize(), saveat=diffrax.SaveAt(t1=True),
        )
        return sol.ys[0]


class ResBlock(eqx.Module):
    """EDSR residual block: conv, ReLU, conv, plus the input (no normalisation)."""
    a: eqx.nn.Conv2d
    b: eqx.nn.Conv2d

    def __init__(self, width, kernel, *, key):
        k1, k2 = jr.split(key)
        self.a = eqx.nn.Conv2d(width, width, kernel, padding="SAME", key=k1)
        self.b = eqx.nn.Conv2d(width, width, kernel, padding="SAME", key=k2)

    def __call__(self, h):
        return h + self.b(jnn.relu(self.a(h)))


class EDSRLite(eqx.Module):
    """E1 baseline: a small EDSR [features on the coarse grid, pixel-shuffle upsampling, no normalisation] on top of the
    same bilinear start as V1 (the tail starts at zero, so an untrained model = bilinear). Fine-grid static channels are
    pixel-unshuffled onto the coarse grid and join the input. Same call as NeuralODEV1; scale_km is unused, because the
    architecture is tied to one scale factor (that is the point of comparing it with V1)."""
    upsampler: Upsampler
    head: eqx.nn.Conv2d
    blocks: tuple
    body: eqx.nn.Conv2d
    tail: eqx.nn.Conv2d
    factor: int = eqx.field(static=True)

    def __init__(self, in_channels=2, strides=(3,), width=32, blocks=8, kernel=3, static_channels=0, *, key):
        self.factor = int(np.prod(strides))
        keys = jr.split(key, blocks + 4)
        self.upsampler = Upsampler(in_channels, strides, key=keys[0])
        self.head = eqx.nn.Conv2d(in_channels + static_channels * self.factor ** 2, width, kernel, padding="SAME", key=keys[1])
        self.blocks = tuple(ResBlock(width, kernel, key=k) for k in keys[4:])
        self.body = eqx.nn.Conv2d(width, width, kernel, padding="SAME", key=keys[2])
        tail = eqx.nn.Conv2d(width, self.factor ** 2, kernel, padding="SAME", key=keys[3])
        self.tail = eqx.tree_at(lambda l: (l.weight, l.bias), tail, (jnp.zeros_like(tail.weight), jnp.zeros_like(tail.bias)))

    def initial_state(self, x):
        return self.upsampler(x)

    def __call__(self, x, scale_km, static=None):
        f = self.factor
        _, h, w = x.shape
        z = x
        if static is not None and static.shape[0]:
            s = static.reshape(static.shape[0], h, f, w, f).transpose(0, 2, 4, 1, 3).reshape(-1, h, w)     # pixel unshuffle
            z = jnp.concatenate([x, s])
        z = self.head(z)
        y = z
        for block in self.blocks:
            y = block(y)
        y = self.tail(self.body(y) + z)
        return self.initial_state(x) + y.reshape(f, f, h, w).transpose(2, 0, 3, 1).reshape(1, h * f, w * f)  # pixel shuffle
