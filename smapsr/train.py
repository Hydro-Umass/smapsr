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
from smapsr.models import NeuralODE
from smapsr.losses import masked_ssim_loss, mae_loss, gradient_loss

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
    y0 = slri_.rio.interpolate_na().data.reshape((1, -1))
    ys = shr_.data.reshape((1, -1))
    return slr, shr, slri, y0, ys

def dataloader(arrays, batch_size, *, key, shuffle=True):
    """Generate batches of data from input arrays. This generator yields batches of data from the provided arrays, supporting both shuffling and non-shuffled data iteration. It is designed to handle cases where the dataset size may be smaller than or equal to the batch size.

    :param arrays: A tuple of JAX arrays to create batches from. Each array must have the same shape[0] (dataset_size).
    :param batch_size: The desired number of samples per batch.
    :param key: A JAX random key for shuffling.
    :param shuffle: Whether or not to shuffle data. Default is True.

    Yields:
        tuple: A tuple containing one batch from each input array, aligned by index."""
    dataset_size = arrays[0].shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        start = 0
        if shuffle:
            perm = jr.permutation(key, indices)
        else:
            perm = indices
        (key,) = jr.split(key, 1)
        end = min(start + batch_size, dataset_size)
        while start < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = min(start + batch_size, dataset_size)

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
            # number of pixels spatially aligned between coarse and fine resolution images
            spatial_match_thresh = 90
            spatial_match = slr_.where(shr_.rio.reproject_match(slr_,).notnull()).notnull().sum()
            if slr_.notnull().any() and shr_.notnull().any() and spatial_match > spatial_match_thresh:
                data0.append(y0.T)
                data.append(ys.T)
    if len(data) == 0:
        raise ValueError("No valid data found for this region")
    data = jnp.array(data)
    key = jr.PRNGKey(seed)
    model_key, loader_key = jr.split(key, 2)
    ts = jnp.linspace(0, 1, 100)
    _, data_size, _ = data.shape
    mask = jnp.array((sh.sel(x=slice(region[0], region[2]), y=slice(region[1], region[3])).notnull().sum('time') > 0).data)
    model = NeuralODE(data_size, width_size, depth, mask, key=model_key)
    optim = optax.adabelief(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, :, 0])
        yi_img = yi[:, :, -1].reshape(yi.shape[0], model.height, model.width)
        yp_img = y_pred[:, 0, :].reshape(y_pred.shape[0], model.height, model.width)
        g_loss = gradient_loss(yi_img, yp_img)
        l1_loss = mae_loss(yi_img, yp_img, model.mask)
        s_loss = masked_ssim_loss(yi_img, yp_img, model.mask)
        return 0.4*l1_loss + 0.0*g_loss + 0.6*s_loss
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
