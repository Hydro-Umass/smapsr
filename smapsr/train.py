import xarray as xr
import numpy as np
import pandas as pd
import time
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
from rasterio.enums import Resampling
from models import NeuralODE

def prepare_data(sl, sh, region):
    """Prepare data for super-resolution.
    - `sl`: coarse-scale soil moisture
    - `sh`: fine-scale soil moisture
    - `region`: list of bounding box [xul, yul, xlr, ylr]
    """
    slr = sl.sel(x=slice(region[0], region[2]), y=slice(region[1], region[3]))
    shr = sh.sel(x=slice(region[0], region[2]), y=slice(region[1], region[3]))
    shr_ = shr.rio.write_nodata(np.nan).rio.interpolate_na()
    slr_ = slr.rio.write_nodata(np.nan).rio.interpolate_na()
    slri_ = slr_.rio.reproject_match(shr_, resampling=Resampling.bilinear).rio.interpolate_na()
    slri = slri_.where(shr.notnull())
    ys = jnp.stack([slri_.data.reshape((1, -1)), shr_.data.reshape((1, -1))])
    return slr, shr, slri, ys

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size

def train(sl, sh, region, train_period, width_size=64, depth=3, lr=1e-3, steps=100, batch_size=32, seed=5678, print_every=20):
    """Train Neural ODE model for super-resolution of SMAP data.

    :param sl: coarse resolution data (xr.DataArray)
    :param sh: fine resolution data (xr.DataArray)
    :param region: region of interest bounding box (list)
    :param train_period: training period date range (pd.DatetimeIndex)
    :param width_size: number of neurons per layer (int)
    :param depth: number of neural network layers (int)
    :param lr: learning rate (float)
    :param steps: number of training epochs (int)
    :param batch_size: batch size for training (int)
    :param seed: random seed (int)
    :param print_every: epoch frequency to print training information (int)

    """
    data = []
    for t in train_period:
        if t in sl.time and t in sh.time:
            slr_, shr_, slri_, ys = prepare_data(sl.sel(time=t), sh.sel(time=t), region)
            if slr_.notnull().any() and shr_.notnull().any():
                data.append(ys[:, 0, :].T)
    data = jnp.array(data)
    key = jr.PRNGKey(seed)
    model_key, loader_key = jr.split(key, 2)
    ts = jnp.linspace(0, 1, 100)
    _, data_size, _ = data.shape
    model = NeuralODE(data_size, width_size, depth, key=model_key)
    optim = optax.adabelief(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, :, 0])
        return jnp.mean((yi[:, :, -1] - y_pred) ** 2)
    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    for step, (yi,) in zip(range(steps), dataloader((data,), batch_size, key=loader_key)):
        start = time.time()
        loss, model, opt_state = make_step(ts, yi, model, opt_state)
        end = time.time()
        if ((step + 1) % print_every) == 0 or step == steps - 1:
            print(f"Step: {step+1}, Loss: {loss}, Time elapsed: {end - start}")
    return model


if __name__ == '__main__':
    s9 = xr.open_dataset("../smap9km.nc").sm.rio.write_crs("epsg:4326")
    s3 = xr.open_dataset("../smap3km.nc").sm.rio.write_crs("epsg:4326")
