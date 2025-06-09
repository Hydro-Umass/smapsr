import jax
import jax.nn as jnn
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
    mask: jax.Array
    height: int
    width: int

    def __init__(self, data_size, width_size, depth, mask, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)
        self.mask = mask
        self.height, self.width = mask.shape

    def __call__(self, ts, y0):
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
